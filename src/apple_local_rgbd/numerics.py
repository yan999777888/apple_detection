"""Bounded same-batch numerical recovery; failed attempts never update weights.

Only floating-point loss/gradient failures are retried. Data, objective, learning
rate, optimizer and successful-update budget are unchanged.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import torch

from .local_training import TrainingContractError


NUMERICAL_POLICY = {
    "version": "same_batch_finite_update_v2",
    "amp_attempts": 3,
    "scale_backoff": 0.5,
    "fp32_fallback_attempts": 1,
    "restore_registered_buffers_and_rng_before_retry": True,
    "failed_batch_skip_allowed": False,
    "clip_error_if_nonfinite": True,
}
AMP_SCALE_POLICY = "no scale growth inside fixed budget; bounded same-batch retry then one FP32 fallback"


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_initialized():
            raise TrainingContractError("CUDA RNG restore requires an initialized training device")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def gradient_failure_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    """Expensive FP64 diagnosis only on failed attempts; JSON-safe even with Inf."""
    bad, finite_norms = [], []
    total_nan = total_inf = parameters_with_grad = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        parameters_with_grad += 1
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        n_nan = int(torch.isnan(gradient).sum().item())
        n_inf = int(torch.isinf(gradient).sum().item())
        total_nan += n_nan
        total_inf += n_inf
        if n_nan or n_inf:
            bad.append({"name": name, "nan_elements": n_nan, "inf_elements": n_inf})
        else:
            finite_norms.append(float(torch.linalg.vector_norm(gradient.to(torch.float64)).item()))
    finite_only_norm = math.hypot(*finite_norms)
    return {"parameters_with_grad": parameters_with_grad,
            "nonfinite_parameter_count": len(bad), "nan_elements": total_nan, "inf_elements": total_inf,
            "first_nonfinite_parameters": bad[:10],
            "all_gradient_elements_finite": not bad,
            "finite_parameters_norm_fp64": finite_only_norm if math.isfinite(finite_only_norm) else None,
            "global_norm_fp64": finite_only_norm if not bad and math.isfinite(finite_only_norm) else None}


class _NonfiniteAttempt(RuntimeError):
    pass


def finite_optimizer_step(*, model: torch.nn.Module, optimizer: Any, scaler: Any,
                          closure: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
                          autocast_context: Callable[[bool], Any],
                          on_retry: Callable[[dict[str, Any]], None],
                          gradient_audit: Callable[[torch.nn.Module], Any] | None = None,
                          clip_norm: float = 10.0) -> dict[str, Any]:
    """Commit exactly one finite update, or fail after bounded same-data attempts.

    The caller owns global-step counters. The FP32 fallback bypasses scaling, but
    retains the reduced GradScaler state for subsequent normal AMP batches.
    """
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rng = capture_rng()
    parameters = list(model.parameters())
    failures = []

    def rollback() -> None:
        current = dict(model.named_buffers())
        if current.keys() != buffers.keys():
            raise TrainingContractError("Registered buffer set changed during numerical attempt")
        with torch.no_grad():
            for name, value in buffers.items():
                current[name].copy_(value)
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)

    modes = [True] * NUMERICAL_POLICY["amp_attempts"] + [False]
    for index, amp_enabled in enumerate(modes):
        optimizer.zero_grad(set_to_none=True)
        loss_value = None
        scale_before = float(scaler.get_scale())
        try:
            with autocast_context(amp_enabled):
                loss, loss_log = closure()
            # Initialize GradScaler even when the first forward is nonfinite, so
            # update(new_scale=...) can reset its per-optimizer attempt state.
            backward_loss = scaler.scale(loss) if amp_enabled else loss
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise _NonfiniteAttempt("Nonfinite forward loss")
            backward_loss.backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            try:
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True))
            except RuntimeError as error:
                if "non-finite" not in str(error).lower() and "nonfinite" not in str(error).lower():
                    raise
                raise _NonfiniteAttempt(str(error)) from error
        except _NonfiniteAttempt as error:
            diagnostics = gradient_failure_diagnostics(model)
            rollback()
            next_scale = scale_before
            if amp_enabled:
                next_scale = scale_before * NUMERICAL_POLICY["scale_backoff"]
                if not math.isfinite(next_scale) or next_scale < torch.finfo(torch.float32).tiny:
                    raise TrainingContractError("AMP recovery scale underflow") from error
                scaler.update(new_scale=next_scale)
            retry = {"attempt": index + 1, "amp_enabled": amp_enabled,
                "fp32_fallback": not amp_enabled, "old_scale": scale_before, "new_scale": next_scale,
                "error": str(error), "loss": loss_value if loss_value is not None and math.isfinite(loss_value) else None,
                "loss_repr": str(loss_value), "optimizer_updated": False,
                "registered_buffers_and_rng_restored": True, "gradient_diagnostics": diagnostics,
                "will_retry_same_batch": index + 1 < len(modes)}
            failures.append(retry)
            on_retry(retry)
            if index + 1 == len(modes):
                raise TrainingContractError("Same batch remains nonfinite after 3 AMP attempts and 1 FP32 fallback; no update") from error
            continue
        except BaseException:
            # OOM, data/programming failures and interrupts are never disguised
            # as numerical retries; undo their forward-only buffer effects.
            rollback()
            raise

        audit = gradient_audit(model) if gradient_audit is not None else None
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise TrainingContractError("Unexpected AMP skipped update after finite gradient check")
        else:
            optimizer.step()
        return {"loss": loss_value, "loss_log": loss_log, "grad_norm_before_clip": norm,
                "gradient_groups": audit, "amp_scale": float(scaler.get_scale()),
                "numerical_attempts": index + 1, "numerical_retries": len(failures),
                "fp32_fallback_used": not amp_enabled, "optimizer_updates": 1}
    raise AssertionError("Unreachable numerical attempt loop")
