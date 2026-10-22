"""Small scheduling/cursor tests plus explicitly enabled real mixed-data smoke."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from apple_seg.training import CyclingOrder, SOURCE_ORDER, _rng_state, _restore_rng, _same, criterion_in_fp32, lr_factor, run_training


class UpdatePlanTests(unittest.TestCase):
    def test_point_matching_math_explicitly_disables_autocast_and_retains_gradient(self):
        import torch
        source = torch.ones((2, 2), dtype=torch.bfloat16, requires_grad=True)
        def criterion(outputs, targets, generator=None):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertTrue(all(value.dtype == torch.float32 for value in outputs.values()))
            return {"contract_objective": sum(value.square().mean() for value in outputs.values())}
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = criterion_in_fp32(criterion, {"class_queries_logits": source}, [])
        losses["contract_objective"].backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(bool(torch.isfinite(source.grad).all()))

    def test_successful_update_schedule(self):
        self.assertEqual(lr_factor(0, 600, 50), .02)
        self.assertEqual(lr_factor(49, 600, 50), 1.)
        self.assertAlmostEqual(lr_factor(599, 600, 50), .05)
        # An AMP skip does not increment this successful-update argument.
        successful = 17
        before = lr_factor(successful, 600, 50)
        self.assertEqual(lr_factor(successful, 600, 50), before)

    def test_independent_shuffled_cycle_resume_has_exact_next_items(self):
        original = CyclingOrder(7, 123)
        first = [original.next() for _ in range(7)]
        self.assertEqual(set(first), set(range(7)))
        saved = deepcopy(original.state_dict())
        restored = CyclingOrder(7, 999)
        restored.load_state_dict(saved)
        self.assertEqual([original.next() for _ in range(19)], [restored.next() for _ in range(19)])
        self.assertEqual(SOURCE_ORDER.count("PApple"), 2)
        self.assertEqual(SOURCE_ORDER.count("MinneApple"), 2)

    def test_optimizer_and_all_RNG_cursor_state_replay_exact_next_step(self):
        # Tiny optimizer/RNG mechanics fixture only; no synthetic dataset/GT is
        # substituted for the separately enabled real-data mixed smoke below.
        import numpy as np
        import torch
        if torch.cuda.is_initialized():
            self.fail("Unit replay must remain on CPU")
        torch.manual_seed(71); random.seed(72); np.random.seed(73)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, foreach=False)
        generator = torch.Generator(device="cpu").manual_seed(74)
        samplers = {name: CyclingOrder(7 + i, 75 + i) for i, name in enumerate(("PApple", "MinneApple"))}

        def step(module, opt, orders, gen, successful):
            indices = [orders[name].next() for name in SOURCE_ORDER]
            values = torch.randn((4, 3), generator=gen) + random.random() + float(np.random.random())
            opt.param_groups[0]["lr"] = 5e-6 * lr_factor(successful, 600, 50)
            opt.zero_grad(set_to_none=True)
            loss = module(values).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(module.parameters(), .1); opt.step()
            return indices, values, float(loss.detach())

        step(model, optimizer, samplers, generator, 0)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng": _rng_state(torch, np, generator, False),
                   "samplers": {name: order.state_dict() for name, order in samplers.items()}}
        stream = io.BytesIO(); torch.save(payload, stream)
        expected = step(model, optimizer, samplers, generator, 1)
        expected_state = deepcopy((model.state_dict(), optimizer.state_dict()))
        stream.seek(0); loaded = torch.load(stream, map_location="cpu", weights_only=False)
        restored = torch.nn.Linear(3, 1)
        restored.load_state_dict(loaded["model"], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1., foreach=False)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        restored_orders = {name: CyclingOrder(7 + i, 999) for i, name in enumerate(("PApple", "MinneApple"))}
        for name, order in restored_orders.items():
            order.load_state_dict(loaded["samplers"][name])
        _restore_rng(loaded["rng"], torch, np, generator, False)
        actual = step(restored, restored_optimizer, restored_orders, generator, 1)
        self.assertTrue(_same(expected, actual, torch, np))
        self.assertTrue(_same(expected_state, (restored.state_dict(), restored_optimizer.state_dict()), torch, np))
        self.assertFalse(torch.cuda.is_initialized())


@unittest.skipUnless(os.environ.get("APPLE_SEG_MIXED_TEST_CONFIG") and os.environ.get("APPLE_SEG_MIXED_TEST_OUTPUT"),
                     "Set real portable config and fresh output paths for actual data/model smoke")
class RealMixedTrainingSmoke(unittest.TestCase):
    def test_real_two_source_update_unknown_A_and_own_checkpoint_roundtrip(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            self.fail("CPU smoke requires CUDA_VISIBLE_DEVICES=-1 before importing Torch")
        repo = Path(__file__).absolute().parents[1]
        output = Path(os.environ["APPLE_SEG_MIXED_TEST_OUTPUT"])
        process = subprocess.run([sys.executable, "-B", str(repo / "scripts/train_mixed.py"),
                                  "--config", os.environ["APPLE_SEG_MIXED_TEST_CONFIG"],
                                  "--device", "cpu", "--smoke", "--max-updates", "1",
                                  "--input-size", "64", "--output", str(output)],
                                 cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        self.assertIn('"event": "launch"', (output / "runtime.log").read_text(encoding="utf-8"))
        self.assertEqual(result["phase"], "smoke_complete")
        self.assertEqual(result["updates"], {"attempted": 1, "successful": 1, "skipped": 0})
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse(result["full_validation_executed"])
        self.assertFalse(result["formal_training_resumable"])
        self.assertTrue(result["checkpoint"]["own_roundtrip_verified"])
        row = result["update_history"][0]
        self.assertTrue(row["finite_gradients"])
        self.assertTrue(all(value > 0 for value in row["smoke_parameter_max_abs_change"].values()))
        self.assertTrue(all(value > 0 for value in row["smoke_readout_gradient_abs_sum"].values()))
        self.assertEqual([m["source"]["dataset"] for m in row["micro_batches"]], list(SOURCE_ORDER))
        for micro in row["micro_batches"]:
            self.assertTrue(micro["GT_unchanged"])
            if micro["source"]["dataset"] == "MinneApple":
                self.assertTrue(micro["unknown_A_gradient_absent_verified"])
                self.assertIsNone(micro["GT_fingerprint"]["amodal_masks"])
                self.assertFalse(any("amodal" in key for key in micro["losses"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
