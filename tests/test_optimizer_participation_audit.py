"""CPU integration checks against the retained local epoch1/epoch2 evidence.

These fixtures are local-only and intentionally excluded from Git. No checkpoint
is modified: negative cases alter an in-memory optimizer counter and restore it.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_recovery_audit", ROOT / "scripts/audit_local_recovery.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OptimizerParticipationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source_path = ROOT / "runs/local_visible_papple_20261104/train_to_001/checkpoints/epoch_001.pt"
        cls.current_path = ROOT / "runs/local_visible_papple_20261104/recovery_v3/train_to_002/checkpoints/epoch_002.pt"
        proof_path = ROOT / "reports/local_formal_20261104/recovery_v3/optimizer_participation_audit.json"
        if not all(p.is_file() for p in (cls.source_path, cls.current_path, proof_path)):
            raise unittest.SkipTest("Retained local epoch1/epoch2 CPU participation fixtures are unavailable")
        cls.old_sha = audit.digest(cls.source_path)
        cls.new_sha = audit.digest(cls.current_path)
        cls.source = torch.load(cls.source_path, map_location="cpu", weights_only=False)
        cls.current = torch.load(cls.current_path, map_location="cpu", weights_only=False)
        cls.steps = audit.read_jsonl(cls.current_path.parents[1] / "steps.jsonl")
        cls.proof = audit.read_json(proof_path)

    def result(self):
        return audit.optimizer_participation_audit(self.source, self.current, self.steps, self.proof,
            source_sha=self.old_sha, current_sha=self.new_sha)

    def test_actual_epoch2_exact_participation_passes(self):
        r = self.result()
        self.assertTrue(r["passed"], r["errors"])
        self.assertEqual(len(r["parameters"]), 442)
        self.assertEqual(r["box_head_parameters"], 48)
        self.assertEqual(r["other_parameters"], 394)

    def reject_delta(self, parameter_id, delta):
        counter = self.current["formal_training"]["optimizer"]["state"][parameter_id]["step"]
        original = counter.clone()
        try:
            counter.add_(delta)
            r = self.result()
            self.assertFalse(r["passed"])
            self.assertTrue(any("Exact parameter step mismatch" in x for x in r["errors"]))
        finally:
            counter.copy_(original)

    def test_extra_update_skip_on_regular_parameter_is_rejected(self):
        self.reject_delta(0, -1)

    def test_extra_update_skip_on_box_parameter_is_rejected(self):
        self.reject_delta(234, -1)

    def test_forcing_box_counter_to_global_step_is_rejected(self):
        self.reject_delta(234, 1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "old_sha"):
            assert audit.digest(cls.source_path) == cls.old_sha
            assert audit.digest(cls.current_path) == cls.new_sha
            assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    unittest.main()
