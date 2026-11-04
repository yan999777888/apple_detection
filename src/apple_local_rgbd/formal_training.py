"""Bounded PApple V-only fine-tuning with epoch-boundary resumable checkpoints.

The original Base/Small models and annotations are never loaded for mutation.
This module reuses the already-tested raster adapter and its loss arithmetic.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .local_training import (
    CHECKPOINT_SCHEMA, NativeVisibleDataset, TrainingContractError,
    build_visible_model, collate_native, compute_e2e_loss, configure_f_runtime,
    gradient_audit, load_local_checkpoint, require_f_path, sha256_file, write_json,
)
from .numerics import (AMP_SCALE_POLICY, NUMERICAL_POLICY, capture_rng,
                       finite_optimizer_step, restore_rng)

FORMAL_SCHEMA = "local_apple_formal_training_v1"
CRITERION_FIELDS = ("updates", "total", "o2m", "o2o", "o2m_copy", "final_o2m")
LEGACY_EPOCH1_SHA256 = "3b80d4a866ffa9394134d031edee95cc72f3e55afe6cf7e72bab69b6bbeffa01"
LEGACY_FORMAL_SHA256 = "6ff345aa6170da9603e5e57440ec8c017c0b0e5a76eabbc26804b1a706ea95e1"
LEGACY_ADAPTER_SHA256 = "4aa52fbc1d212c436294b35a970f910b936ce9308b45d4a8064be28bf1d7a037"
LEGACY_SCALE_POLICY = "no scale growth inside this fixed update budget; nonfinite gradients fail"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes() -> dict[str, str]:
    return {"formal_training.py": sha256_file(__file__),
            "local_training.py": sha256_file(Path(__file__).with_name("local_training.py")),
            "numerics.py": sha256_file(Path(__file__).with_name("numerics.py"))}


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """Epoch 0 based, independent of checkpoint saves and global RNG calls."""
    generator = torch.Generator().manual_seed(seed + epoch * 1000003)
    return torch.randperm(count, generator=generator).tolist()


def criterion_state(criterion: Any) -> dict[str, Any]:
    return {key: getattr(criterion, key) for key in CRITERION_FIELDS}


def restore_criterion(criterion: Any, state: dict[str, Any]) -> None:
    if set(state) != set(CRITERION_FIELDS):
        raise TrainingContractError("Incomplete E2ELoss schedule state")
    for key in CRITERION_FIELDS:
        setattr(criterion, key, state[key])


def cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def recipe_sha256(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_resume(checkpoint: dict[str, Any], config: dict[str, Any], *,
                    upgrade_numerical_policy: bool = False,
                    checkpoint_sha256: str | None = None) -> dict[str, Any] | None:
    state = checkpoint.get("formal_training", {})
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or state.get("schema") != FORMAL_SCHEMA:
        raise TrainingContractError("Resume requires a formal checkpoint, not a smoke checkpoint")
    if not state.get("at_epoch_boundary"):
        raise TrainingContractError("Only epoch-boundary checkpoints are resumable")
    expected = state["completed_epochs"] * state["steps_per_epoch"]
    if state["global_step"] != expected or state["criterion"]["updates"] != state["completed_epochs"]:
        raise TrainingContractError("Checkpoint epoch/update/schedule counters disagree")
    old = state.get("recipe")
    if old == config:
        if upgrade_numerical_policy:
            raise TrainingContractError("Numerical-policy upgrade is only for the allowlisted legacy epoch-1 checkpoint")
        return None
    if not upgrade_numerical_policy:
        raise TrainingContractError("Resume recipe differs: dataset/source/optimizer/schedule must stay fixed")
    legacy_sources = {"formal_training.py": LEGACY_FORMAL_SHA256,
                      "local_training.py": LEGACY_ADAPTER_SHA256}
    if (checkpoint_sha256 != LEGACY_EPOCH1_SHA256 or state["completed_epochs"] != 1
            or state["steps_per_epoch"] != 1152 or state["global_step"] != 1152
            or not isinstance(old, dict) or old.get("sources") != legacy_sources
            or old.get("amp_scale_policy") != LEGACY_SCALE_POLICY or "numerical_policy" in old
            or config.get("numerical_policy") != NUMERICAL_POLICY
            or config.get("amp_scale_policy") != AMP_SCALE_POLICY
            or config.get("sources") != source_hashes()
            or config["sources"].get("local_training.py") != LEGACY_ADAPTER_SHA256
            or checkpoint.get("config") != old):
        raise TrainingContractError("Numerical-policy migration does not match the fixed checkpoint/source allowlist")
    comparable = copy.deepcopy(config)
    comparable.pop("numerical_policy")
    comparable["amp_scale_policy"] = LEGACY_SCALE_POLICY
    comparable["sources"] = legacy_sources
    if comparable != old:
        raise TrainingContractError("Numerical-policy migration changed fields outside the explicit allowlist")
    return {"schema": "local_formal_numerical_migration_v1", "created_utc": now(),
            "checkpoint_sha256": checkpoint_sha256, "old_recipe_sha256": recipe_sha256(old),
            "new_recipe_sha256": recipe_sha256(config), "old_sources": old["sources"],
            "new_sources": config["sources"], "numerical_policy": config["numerical_policy"],
            "allowed_changed_fields": ["sources.formal_training.py", "sources.numerics.py (added)",
                                       "amp_scale_policy", "numerical_policy (added)"],
            "all_other_recipe_fields_equal": True, "dataset_and_optimizer_schedule_unchanged": True}


def verify_preflight(report_path: Path, dataset: NativeVisibleDataset) -> dict[str, Any]:
    audit = json.loads(report_path.read_text(encoding="utf-8-sig"))
    expected = {"manifest_sha256": dataset.manifest_sha256, "imgsz": dataset.imgsz,
                "mask_ratio": dataset.mask_ratio, "image_count": len(dataset),
                "instance_count": sum(x["instance_count"] for x in dataset.rows)}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise TrainingContractError("Full-train preflight does not match the current manifest/shape/counts")
    if audit.get("passed") is not True or audit.get("failures") or audit.get("GT_dropped", 0) != 0:
        raise TrainingContractError("Full-train preflight failed or dropped ground truth")
    return {"path": str(report_path), "sha256": sha256_file(report_path), **expected,
            "semantic_p3_vanished_instance_count": audit.get("semantic_p3_vanished_instance_count"),
            "passed": True}


class PreparedVisibleDataset(NativeVisibleDataset):
    """Optional lossless disk cache of adapter tensors; never caches a dataset in RAM."""
    ARRAY_KEYS = ("img", "masks", "bboxes", "cls", "sem_masks")

    def __init__(self, *args: Any, prepared_cache: Path | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.cache_directory = None
        if prepared_cache is not None:
            key = hashlib.sha256(json.dumps({"manifest": self.manifest_sha256,
                "adapter": source_hashes()["local_training.py"], "imgsz": self.imgsz,
                "mask_ratio": self.mask_ratio, "schema": "lossless_tensors_v1"},
                sort_keys=True).encode()).hexdigest()
            self.cache_directory = require_f_path(prepared_cache) / key
            self.cache_directory.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_directory is None:
            return super().__getitem__(index)
        row = self.rows[index]
        if index not in self._verified:
            for key, digest in (("rgb_path", "rgb_sha256"), ("mask_path", "mask_sha256")):
                if sha256_file(row[key]) != row[digest]:
                    raise TrainingContractError(f"Source SHA mismatch before cache access: {row[key]}")
            self._verified.add(index)
        path = self.cache_directory / f"{index:06d}.npz"
        receipt_path = path.with_suffix(".json")
        if path.exists() or receipt_path.exists():
            if not path.exists() or not receipt_path.exists():
                raise TrainingContractError(f"Incomplete prepared-cache pair: {path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("sha256") != sha256_file(path) or receipt.get("rgb_sha256") != row["rgb_sha256"]
                    or receipt.get("mask_sha256") != row["mask_sha256"]):
                raise TrainingContractError(f"Prepared cache checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as saved:
                item = {key: torch.from_numpy(saved[key].copy()) for key in self.ARRAY_KEYS}
                item["metadata"] = json.loads(str(saved["metadata_json"].item()))
            if item["metadata"]["image_id"] != row["image_id"]:
                raise TrainingContractError("Prepared cache identity mismatch")
            return item
        item = super().__getitem__(index)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{key: item[key].numpy() for key in self.ARRAY_KEYS},
                                metadata_json=np.array(json.dumps(item["metadata"], ensure_ascii=False)))
        temporary.replace(path)
        write_json(receipt_path, {"sha256": sha256_file(path), "rgb_sha256": row["rgb_sha256"],
                                  "mask_sha256": row["mask_sha256"], "lossless": True})
        return item


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: Any, scaler: Any,
                    criterion: Any, recipe: dict[str, Any], initialization: dict[str, Any],
                    completed_epochs: int, global_step: int, steps_per_epoch: int,
                    at_epoch_boundary: bool) -> dict[str, Any]:
    payload = {"schema": CHECKPOINT_SCHEMA,
        "state_dict": cpu_snapshot(model.state_dict()), "model_yaml": copy.deepcopy(model.yaml),
        "model_args": {"task": "segment", "overlap_mask": False, "epochs": recipe["epochs"]},
        "names": {0: "apple"}, "config": recipe, "initialization": initialization,
        "steps": global_step, "metrics": None,
        "formal_training": {"schema": FORMAL_SCHEMA, "created_utc": now(), "recipe": recipe,
            "completed_epochs": completed_epochs, "global_step": global_step,
            "steps_per_epoch": steps_per_epoch, "at_epoch_boundary": at_epoch_boundary,
            "criterion": criterion_state(criterion), "rng": capture_rng(),
            "sampler": {"type": "deterministic_epoch_permutation", "seed": recipe["seed"],
                        "next_epoch": completed_epochs, "epoch_is_zero_based": True},
            "optimizer": cpu_snapshot(optimizer.state_dict()), "scaler": scaler.state_dict(),
            "resume_scope": "epoch boundaries only; same recipe and runtime"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    receipt = {"path": str(path), "sha256": sha256_file(path), "completed_epochs": completed_epochs,
               "global_step": global_step, "at_epoch_boundary": at_epoch_boundary, "created_utc": now()}
    write_json(path.with_suffix(".json"), receipt)
    return receipt


def run_formal(*, repo: Path, manifest: Path, weights: Path, output: Path,
               epochs: int = 10, until_epoch: int | None = None, batch_size: int = 2,
               imgsz: int = 640, mask_ratio: int = 2, seed: int = 20261104,
               lr: float = 1e-4, amp_init_scale: float = 1024, device: str = "cuda:0",
               preflight_report: Path | None = None, prepared_cache: Path | None = None,
               resume: Path | None = None, max_updates: int | None = None,
               log_interval: int = 20, upgrade_numerical_policy: bool = False) -> dict[str, Any]:
    if not 1 <= epochs <= 100 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise TrainingContractError("Invalid bounded epochs/batch/lr")
    if not math.isfinite(amp_init_scale) or amp_init_scale <= 0:
        raise TrainingContractError("AMP scale must be finite and positive")
    until_epoch = epochs if until_epoch is None else until_epoch
    if not 0 <= until_epoch <= epochs or log_interval < 1:
        raise TrainingContractError("until_epoch/log_interval invalid")
    if until_epoch == 0 and (resume is not None or max_updates is not None):
        raise TrainingContractError("Epoch zero is a fresh initialization-only node")
    if max_updates is not None and not 1 <= max_updates <= 32:
        raise TrainingContractError("The engineering max_updates override is restricted to 1..32")
    if upgrade_numerical_policy and resume is None:
        raise TrainingContractError("Numerical-policy upgrade requires the allowlisted epoch-1 resume checkpoint")
    configure_f_runtime(repo)
    repo, manifest, weights, output = map(require_f_path, (repo, manifest, weights, output))
    if resume is not None:
        resume = require_f_path(resume)
    if output.exists() and any(output.iterdir()):
        raise TrainingContractError("Choose a new segment output directory, including when resuming")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = repo / "reports/local_training_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(descriptor, json.dumps({"pid": os.getpid(), "output": str(output), "created_utc": now()}).encode())
    os.close(descriptor)
    completed_epochs, global_step, current_epoch, completed_in_segment = 0, 0, 0, 0
    last_boundary = None
    status = {"phase": "preflight", "pid": os.getpid(), "created_utc": now(),
              "target_epochs": epochs, "segment_until_epoch": until_epoch, "engineering_max_updates": max_updates}
    write_json(output / "pipeline_status.json", status)
    events_path = output / "events.jsonl"
    def event(kind: str, **values: Any) -> None:
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": now(), "kind": kind, **values}, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(2); cv2.setNumThreads(1)
        dataset = PreparedVisibleDataset(manifest, imgsz=imgsz, mask_ratio=mask_ratio,
                                         prepared_cache=prepared_cache)
        if len(dataset) != 2304 or sum(x["instance_count"] for x in dataset.rows) != 8579:
            raise TrainingContractError("Formal training requires the complete original 2304-image/8579-GT train split")
        for row in dataset.rows:
            require_f_path(row["rgb_path"]); require_f_path(row["mask_path"])
        if preflight_report is None:
            audit = dataset.audit()
            write_json(output / "full_train_preflight.json", audit)
            preflight_report = output / "full_train_preflight.json"
        preflight = verify_preflight(require_f_path(preflight_report), dataset)
        write_json(output / "preflight_receipt.json", preflight)
        selected = torch.device(device)
        if selected.type != "cuda" or not torch.cuda.is_available():
            raise TrainingContractError("Formal training requires the root-controlled CUDA slot")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import ultralytics
        recipe = {"mode": "formal_visible_apple_training", "epochs": epochs, "batch_size": batch_size,
            "imgsz": imgsz, "mask_ratio": mask_ratio, "seed": seed, "lr": lr,
            "optimizer": "AdamW", "weight_decay": 1e-4, "lr_schedule": "constant",
            "amp": True, "amp_init_scale": amp_init_scale,
            "amp_growth_interval": math.ceil(len(dataset) / batch_size) * epochs + 1,
            "amp_scale_policy": AMP_SCALE_POLICY, "numerical_policy": copy.deepcopy(NUMERICAL_POLICY),
            "gradient_clip_norm": 10.0,
            "workers": 0, "pin_memory": False, "augmentation": "letterbox only",
            "classes": {"0": "apple"}, "A_in_loss": False, "GT_changed": False,
            "manifest_sha256": dataset.manifest_sha256, "official_weights_sha256": sha256_file(weights),
            "sources": source_hashes(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cudnn_deterministic": True, "cudnn_benchmark": False}
        write_json(output / "recipe.json", recipe)
        checkpoint = None
        if resume is None:
            model, initialization = build_visible_model(weights, epochs=epochs)
        else:
            resume_sha256 = sha256_file(resume)
            if upgrade_numerical_policy and resume_sha256 != LEGACY_EPOCH1_SHA256:
                raise TrainingContractError("Numerical-policy migration checkpoint SHA mismatch before loading")
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            migration = validate_resume(checkpoint, recipe,
                upgrade_numerical_policy=upgrade_numerical_policy, checkpoint_sha256=resume_sha256)
            if migration is not None:
                migration["checkpoint_path"] = str(resume)
                write_json(output / "migration_receipt.json", migration)
                event("numerical_policy_migration", **migration)
            model = load_local_checkpoint(resume)
            initialization = checkpoint["initialization"]
            frozen = set(initialization["frozen_parameter_names"])
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name not in frozen)
            completed_epochs = checkpoint["formal_training"]["completed_epochs"]
            global_step = checkpoint["formal_training"]["global_step"]
            if completed_epochs >= until_epoch:
                raise TrainingContractError("Resume checkpoint has already reached this segment target")
        write_json(output / "initialization.json", initialization)
        model.to(selected).train()
        model.criterion = model.init_criterion()
        criterion = model.criterion
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=amp_init_scale,
                                     growth_interval=recipe["amp_growth_interval"])
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        if checkpoint is not None:
            saved = checkpoint["formal_training"]
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            restore_criterion(criterion, saved["criterion"])
            restore_rng(saved["rng"])
            last_boundary = {"path": str(resume), "sha256": sha256_file(resume),
                             "completed_epochs": completed_epochs, "global_step": global_step}
            del checkpoint
            event("resumed", checkpoint=last_boundary)
        else:
            last_boundary = save_checkpoint(output / "checkpoints/epoch_000.pt", model=model,
                optimizer=optimizer, scaler=scaler, criterion=criterion, recipe=recipe, initialization=initialization,
                completed_epochs=0, global_step=0, steps_per_epoch=steps_per_epoch, at_epoch_boundary=True)
            event("initialized", checkpoint=last_boundary)
        if until_epoch == 0:
            status.update(phase="initialization_complete", updated_utc=now(), completed_epochs=0,
                global_step=0, last_boundary_checkpoint=last_boundary, validation_evaluated_here=False)
            write_json(output / "pipeline_status.json", status)
            return status
        torch.cuda.reset_peak_memory_stats(selected)
        run_started = time.perf_counter()
        for epoch in range(completed_epochs, until_epoch):
            current_epoch = epoch
            order = epoch_order(len(dataset), seed, epoch)
            order_sha = hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest()
            loader = DataLoader(dataset, batch_size=batch_size, sampler=order, num_workers=0,
                pin_memory=False, persistent_workers=False, collate_fn=collate_native, drop_last=False,
                generator=torch.Generator().manual_seed(seed + epoch))
            epoch_started = time.perf_counter()
            loss_sum, images_seen, instances_seen = 0.0, 0, 0
            for batch_index, batch in enumerate(loader):
                metadata = batch.pop("metadata")
                images_seen += len(metadata)
                instances_seen += sum(len(x["instance_ids"]) for x in metadata)
                torch.cuda.synchronize(selected)
                tick = time.perf_counter()
                batch = {key: value.to(selected, non_blocking=False) for key, value in batch.items()}
                def closure() -> tuple[torch.Tensor, dict[str, Any]]:
                    predictions = model(batch["img"])
                    return compute_e2e_loss(criterion, predictions, batch)
                def retry_event(values: dict[str, Any]) -> None:
                    record = {"utc": now(), "epoch": epoch + 1,
                        "batch": batch_index + 1, "global_step_completed": global_step,
                        "samples": metadata, "image_ids": [x["image_id"] for x in metadata], **values}
                    with (output / "numerical_retries.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    event("numerical_retry", **{key: value for key, value in record.items() if key != "utc"})
                    write_json(output / "last_numerical_retry.json", record)
                step = finite_optimizer_step(model=model, optimizer=optimizer, scaler=scaler,
                    closure=closure,
                    autocast_context=lambda enabled: torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled),
                    on_retry=retry_event,
                    gradient_audit=gradient_audit if completed_in_segment < 2 or batch_index + 1 == steps_per_epoch else None,
                    clip_norm=recipe["gradient_clip_norm"])
                global_step += 1; completed_in_segment += 1
                scalar_loss = step["loss"]; loss_sum += scalar_loss
                torch.cuda.synchronize(selected)
                record = {"utc": now(), "global_step": global_step, "epoch": epoch + 1,
                    "batch": batch_index + 1, "total_loss": scalar_loss, **step["loss_log"],
                    **{key: step[key] for key in ("grad_norm_before_clip", "amp_scale", "numerical_attempts",
                        "numerical_retries", "fp32_fallback_used", "optimizer_updates")},
                    "step_compute_ms_excluding_loader": (time.perf_counter() - tick) * 1000,
                    "image_ids": [x["image_id"] for x in metadata], "instances": sum(len(x["instance_ids"]) for x in metadata),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected)}
                if step["gradient_groups"] is not None:
                    record["gradient_groups"] = step["gradient_groups"]
                    record["gradient_audit_stage"] = "after_unscale_and_clip"
                with (output / "steps.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                status.update(phase="training", updated_utc=now(), completed_epochs=completed_epochs,
                    current_epoch=epoch + 1, batch_in_epoch=batch_index + 1, batches_per_epoch=steps_per_epoch,
                    global_step=global_step, last_loss=scalar_loss, last_boundary_checkpoint=last_boundary,
                    peak_allocated_bytes=record["peak_allocated_bytes"],
                    eta_current_epoch_seconds=((time.perf_counter() - epoch_started) / (batch_index + 1)
                        * (steps_per_epoch - batch_index - 1)) if batch_index >= 9 else None,
                    eta_segment_seconds=((time.perf_counter() - epoch_started) / (batch_index + 1)
                        * ((until_epoch - epoch) * steps_per_epoch - batch_index - 1)) if batch_index >= 9 else None)
                write_json(output / "pipeline_status.json", status)
                if global_step % log_interval == 0 or completed_in_segment <= 2:
                    print(json.dumps({key: record[key] for key in ("epoch", "batch", "global_step", "total_loss", "amp_scale")}), flush=True)
                if max_updates is not None and completed_in_segment >= max_updates:
                    partial = save_checkpoint(output / f"checkpoints/engineering_partial_{global_step:06d}.pt",
                        model=model, optimizer=optimizer, scaler=scaler, criterion=criterion, recipe=recipe,
                        initialization=initialization, completed_epochs=completed_epochs, global_step=global_step,
                        steps_per_epoch=steps_per_epoch, at_epoch_boundary=False)
                    status.update(phase="engineering_gate_complete", updated_utc=now(), checkpoint=partial,
                                  full_training_complete=False, partial_checkpoint_resumable=False)
                    write_json(output / "pipeline_status.json", status)
                    event("engineering_gate_complete", checkpoint=partial)
                    return status
            if images_seen != len(dataset) or instances_seen != 8579:
                raise TrainingContractError("Epoch did not consume every original train image/instance exactly once")
            completed_epochs = epoch + 1
            criterion.update()
            last_boundary = save_checkpoint(output / f"checkpoints/epoch_{completed_epochs:03d}.pt",
                model=model, optimizer=optimizer, scaler=scaler, criterion=criterion, recipe=recipe,
                initialization=initialization, completed_epochs=completed_epochs, global_step=global_step,
                steps_per_epoch=steps_per_epoch, at_epoch_boundary=True)
            epoch_result = {"epoch": completed_epochs, "images": images_seen, "instances": instances_seen,
                "optimizer_updates": steps_per_epoch, "global_step": global_step, "mean_batch_loss": loss_sum / steps_per_epoch,
                "epoch_seconds_including_loader_and_checkpoint": time.perf_counter() - epoch_started,
                "sampler_order_sha256": order_sha, "next_criterion_state": criterion_state(criterion),
                "checkpoint": last_boundary, "precision_recall_evaluated_here": False}
            write_json(output / f"epoch_{completed_epochs:03d}.json", epoch_result)
            event("epoch_complete", **epoch_result)
        status.update(phase="training_complete" if completed_epochs == epochs else "epoch_node_complete",
            updated_utc=now(), completed_epochs=completed_epochs, global_step=global_step,
            segment_seconds=time.perf_counter() - run_started, last_boundary_checkpoint=last_boundary,
            validation_evaluated_here=False, automatic_model_promotion=False)
        write_json(output / "pipeline_status.json", status)
        event(status["phase"], completed_epochs=completed_epochs, global_step=global_step)
        return status
    except BaseException as error:
        status.update(phase="failed" if not isinstance(error, KeyboardInterrupt) else "interrupted",
            updated_utc=now(), completed_epochs=completed_epochs, attempted_epoch=current_epoch + 1,
            global_step=global_step, last_boundary_checkpoint=last_boundary,
            error_type=type(error).__name__, error=str(error),
            resume_policy="resume last completed epoch; discard and replay incomplete epoch updates")
        write_json(output / "pipeline_status.json", status)
        event(status["phase"], error_type=type(error).__name__, error=str(error), global_step=global_step)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
