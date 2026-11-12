"""CPU checks for bounded BN diagnosis; no model weights, images or CUDA."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
MODULE = Path(__file__).resolve().parents[1] / "apple_research_v2/diagnose_joint_bn_20261112.py"
spec = importlib.util.spec_from_file_location("joint_bn_diag", MODULE)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class BNDiagnosticTests(unittest.TestCase):
    def test_parameter_and_non_target_state_invariants(self):
        result = diag.cpu_self_test()
        self.assertTrue(result["all_parameters_unchanged"])
        self.assertTrue(result["head_BN_unchanged"])
        self.assertFalse(result["CUDA_initialized"])

    def test_cumulative_statistics_are_per_batch_not_exponential(self):
        import torch
        from torch import nn
        torch.manual_seed(6)
        model = nn.Sequential(nn.BatchNorm2d(3)).eval().requires_grad_(False)
        inputs = [torch.randn(4, 3, 6, 6) * scale + shift for scale, shift in ((1, 3), (2, -2), (3, 6))]
        expected_mean = torch.stack([x.mean((0, 2, 3)) for x in inputs]).mean(0)
        expected_var = torch.stack([x.var((0, 2, 3), unbiased=True) for x in inputs]).mean(0)
        audit = diag.recalibrate_bn(model, model, inputs, model)
        torch.testing.assert_close(model[0].running_mean, expected_mean)
        torch.testing.assert_close(model[0].running_var, expected_var)
        self.assertEqual(int(model[0].num_batches_tracked), 3)
        self.assertTrue(audit["non_target_state_unchanged"])
        self.assertEqual(model[0].momentum, .1)

    def test_invalid_batch_restores_modes_and_momentum(self):
        import torch
        from torch import nn
        model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(.5)).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "Invalid calibration batch"):
            diag.recalibrate_bn(model, model, [torch.zeros(3, 3, 4, 4)], model)
        self.assertFalse(any(module.training for module in model.modules()))
        self.assertEqual(model[0].momentum, .1)

    def test_train_only_selection_is_deterministic_and_balanced(self):
        rows = []
        for source in ("stage", "defect"):
            for split in ("train", "val"):
                for i in range(80):
                    row = {"sample_id": f"{source}/{split}/{i}", "source": source, "split": split,
                           "image_path": f"{source}/{split}/{i}.jpg", "image_sha256": "image",
                           "label_path": f"{source}/{split}/{i}.txt", "label_sha256": "label"}
                    rows.append(row)
        selected = {source: diag.select_rows(rows, source) for source in ("stage", "defect")}
        self.assertEqual(selected["stage"], diag.select_rows(list(reversed(rows)), "stage"))
        batches = diag.batch_plan(selected, "balanced64_64")
        self.assertEqual(len(batches), 32)
        self.assertEqual([batch[0]["source"] for batch in batches], ["stage", "defect"] * 16)
        self.assertTrue(all(row["split"] == "train" for batch in batches for row in batch))
        self.assertEqual(len({row["sample_id"] for batch in batches for row in batch}), 128)
        self.assertTrue(all(len(batch) == 4 for batch in batches))


if __name__ == "__main__":
    unittest.main()
