"""CPU GradScaler tests: numerical retries preserve one-batch semantics."""
from contextlib import contextmanager
import copy
import random
import unittest

import numpy as np
import torch

from apple_local_rgbd.local_training import TrainingContractError
from apple_local_rgbd.numerics import (
    capture_rng, finite_optimizer_step, gradient_failure_diagnostics, restore_rng,
)


class NumericalRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(71); np.random.seed(71); torch.manual_seed(71)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                         torch.nn.Dropout(.25), torch.nn.Linear(4, 1))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.scaler = torch.amp.GradScaler("cpu", init_scale=1024, growth_interval=1000)
        self.x = torch.randn(5, 3)
        self.records, self.observations = [], []
        self.mode = True

    @contextmanager
    def context(self, enabled):
        self.mode = enabled
        # CPU tests deliberately use FP32 math with a real GradScaler. An Inf
        # gradient is injected to exercise precisely the overflow transaction.
        yield

    def closure(self):
        draws = (random.random(), np.random.rand())
        prediction = self.model(self.x)
        self.observations.append((draws, prediction.detach().clone(), self.mode))
        loss = prediction.square().mean()
        return loss, {"loss_test": float(loss.detach())}

    def run_step(self, closure=None):
        return finite_optimizer_step(model=self.model, optimizer=self.optimizer, scaler=self.scaler,
            closure=closure or self.closure, autocast_context=self.context, on_retry=self.records.append)

    def assert_one_committed_update(self, attempts):
        self.assertEqual(int(self.model[1].num_batches_tracked), 1)
        for state in self.optimizer.state.values():
            self.assertEqual(int(state["step"]), 1)
        self.assertEqual(len(self.observations), attempts)
        for draws, output, _ in self.observations[1:]:
            self.assertEqual(draws, self.observations[0][0])
            self.assertTrue(torch.equal(output, self.observations[0][1]))

    def test_normal_amp_does_one_update_without_retry(self):
        result = self.run_step()
        self.assertEqual(result["numerical_attempts"], 1)
        self.assertFalse(result["fp32_fallback_used"])
        self.assertEqual(result["amp_scale"], 1024)
        self.assertEqual(self.records, [])
        self.assert_one_committed_update(1)

    def test_gradient_overflow_replays_identical_bn_rng_and_one_optimizer_update(self):
        initial_model = copy.deepcopy(self.model.state_dict())
        initial_rng = capture_rng()
        calls = 0
        def overflow_once(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, float("inf")) if calls == 1 else gradient
        hook = self.model[-1].weight.register_hook(overflow_once)
        result = self.run_step()
        hook.remove()
        actual_model = copy.deepcopy(self.model.state_dict())
        actual_next_rng = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(result["numerical_attempts"], 2)
        self.assertEqual(result["amp_scale"], 512)
        self.assert_one_committed_update(2)
        self.assertEqual(self.records[0]["gradient_diagnostics"]["nonfinite_parameter_count"], 1)
        self.assertFalse(self.records[0]["optimizer_updated"])
        # A clean one-pass update from the same state at the recovered scale is
        # bitwise identical, including the next RNG draws and BatchNorm buffers.
        self.model.load_state_dict(initial_model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.scaler = torch.amp.GradScaler("cpu", init_scale=512, growth_interval=1000)
        restore_rng(initial_rng)
        self.run_step()
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, actual_model[name]), name)
        expected_next_rng = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(actual_next_rng[:2], expected_next_rng[:2])
        self.assertTrue(torch.equal(actual_next_rng[2], expected_next_rng[2]))

    def test_three_amp_failures_use_one_fp32_fallback_then_resume_amp(self):
        hook = self.model[-1].weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf")) if self.mode else gradient)
        result = self.run_step()
        hook.remove()
        self.assertTrue(result["fp32_fallback_used"])
        self.assertEqual(result["numerical_attempts"], 4)
        self.assertEqual(result["amp_scale"], 128)
        self.assertEqual([r["new_scale"] for r in self.records], [512, 256, 128])
        self.assert_one_committed_update(4)
        next_result = self.run_step()
        self.assertFalse(next_result["fp32_fallback_used"])
        self.assertEqual(next_result["numerical_attempts"], 1)
        for state in self.optimizer.state.values():
            self.assertEqual(int(state["step"]), 2)

    def test_nonfinite_forward_initializes_scaler_and_retries_same_batch(self):
        def loss_overflow_once():
            loss, log = self.closure()
            return (loss * float("inf") if len(self.observations) == 1 else loss), log
        result = self.run_step(loss_overflow_once)
        self.assertEqual(result["numerical_attempts"], 2)
        self.assertEqual(self.records[0]["loss"], None)
        self.assert_one_committed_update(2)

    def test_final_failure_rolls_back_and_never_updates_or_silently_skips(self):
        initial = copy.deepcopy(self.model.state_dict())
        rng = capture_rng()
        hook = self.model[-1].weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        with self.assertRaisesRegex(TrainingContractError, "no update"):
            self.run_step()
        hook.remove()
        self.assertEqual(len(self.records), 4)
        self.assertFalse(self.records[-1]["will_retry_same_batch"])
        self.assertTrue(self.records[-1]["fp32_fallback"])
        self.assertEqual(self.optimizer.state, {})
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(initial[name], value), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        actual = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(rng)
        expected = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_unrelated_runtime_error_is_not_retried(self):
        def broken():
            self.closure()
            raise RuntimeError("unrelated engineering failure")
        with self.assertRaisesRegex(RuntimeError, "engineering failure"):
            self.run_step(broken)
        self.assertEqual(len(self.observations), 1)
        self.assertEqual(self.records, [])
        self.assertEqual(int(self.model[1].num_batches_tracked), 0)
        self.assertEqual(self.optimizer.state, {})

    def test_fp64_diagnosis_distinguishes_finite_element_norm_overflow(self):
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 1e30)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, error_if_nonfinite=True)
        diagnosis = gradient_failure_diagnostics(self.model)
        self.assertTrue(diagnosis["all_gradient_elements_finite"])
        self.assertGreater(diagnosis["global_norm_fp64"], 1e30)


if __name__ == "__main__":
    unittest.main()
