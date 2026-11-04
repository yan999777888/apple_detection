"""CPU-only contracts for bounded stages, lossless cache and boundary resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from apple_local_rgbd.formal_training import (
    PreparedVisibleDataset, capture_rng, cpu_snapshot, criterion_state, epoch_order,
    restore_criterion, restore_rng, save_checkpoint, validate_resume, verify_preflight,
    LEGACY_EPOCH1_SHA256, LEGACY_FORMAL_SHA256, LEGACY_ADAPTER_SHA256,
    LEGACY_SCALE_POLICY, recipe_sha256, source_hashes,
)
from apple_local_rgbd.numerics import AMP_SCALE_POLICY, NUMERICAL_POLICY
from apple_local_rgbd.local_training import (
    MANIFEST_SCHEMA, NativeVisibleDataset, TrainingContractError, sha256_file, tight_boxes,
)


class FormalTrainingTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[1] / "tmp/local_formal_tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        masks = np.zeros((1, 64, 64), dtype=np.uint8)
        masks[0, 2:24, 2:24] = 1
        masks[0, 40:60, 40:60] = 1
        image = self.root / "rgb.png"
        Image.fromarray(np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)).save(image)
        mask = self.root / "mask.npz"
        np.savez_compressed(mask, visible_packed=np.packbits(masks, axis=-1, bitorder="little"),
            amodal_packed=np.packbits(masks, axis=-1, bitorder="little"), original_hw=np.array([64, 64]),
            instance_ids=np.array(["19"]), labels=np.array([0]), boxes_visible_xyxy=tight_boxes(masks),
            boxes_amodal_xyxy=tight_boxes(masks))
        row = {"split": "train", "image_id": image.name, "rgb_path": str(image), "mask_path": str(mask),
               "rgb_sha256": sha256_file(image), "mask_sha256": sha256_file(mask),
               "original_hw": [64, 64], "instance_count": 1}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "split": "train", "role": "train", "samples": [row]}))
        return manifest

    def test_cache_preserves_all_adapter_tensors_and_metadata_exactly(self):
        path = self.manifest()
        plain = NativeVisibleDataset(path, imgsz=64, mask_ratio=2)[0]
        cached = PreparedVisibleDataset(path, imgsz=64, mask_ratio=2, prepared_cache=self.root / "cache")
        written = cached[0]
        reread = PreparedVisibleDataset(path, imgsz=64, mask_ratio=2, prepared_cache=self.root / "cache")[0]
        for key in cached.ARRAY_KEYS:
            self.assertTrue(torch.equal(plain[key], written[key]), key)
            self.assertTrue(torch.equal(plain[key], reread[key]), key)
            self.assertEqual(plain[key].dtype, reread[key].dtype)
        self.assertEqual(plain["metadata"], reread["metadata"])

    def test_corrupt_prepared_cache_is_rejected(self):
        dataset = PreparedVisibleDataset(self.manifest(), imgsz=64, prepared_cache=self.root / "cache")
        dataset[0]
        path = next(dataset.cache_directory.glob("*.npz"))
        with path.open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(TrainingContractError, "checksum mismatch"):
            dataset[0]

    def test_epoch_order_is_complete_and_resume_independent(self):
        order = epoch_order(2304, 20261104, 4)
        self.assertEqual(sorted(order), list(range(2304)))
        torch.rand(100)
        self.assertEqual(order, epoch_order(2304, 20261104, 4))
        self.assertNotEqual(order, epoch_order(2304, 20261104, 5))

    def test_preflight_rejects_stale_dimensions_or_failed_contract(self):
        dataset = NativeVisibleDataset(self.manifest(), imgsz=64, mask_ratio=2)
        report = {"manifest_sha256": dataset.manifest_sha256, "imgsz": 64, "mask_ratio": 2,
                  "image_count": 1, "instance_count": 1, "passed": True, "failures": [], "GT_dropped": 0}
        path = self.root / "audit.json"
        path.write_text(json.dumps(report))
        self.assertTrue(verify_preflight(path, dataset)["passed"])
        report["imgsz"] = 32
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(TrainingContractError, "does not match"):
            verify_preflight(path, dataset)
        report["imgsz"] = 64; report["passed"] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(TrainingContractError, "failed"):
            verify_preflight(path, dataset)

    def test_rng_restores_python_numpy_and_torch_cpu(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(5))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(5))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertIsNone(state["torch_cuda"])

    def test_cpu_optimizer_dropout_resume_matches_uninterrupted_next_update(self):
        torch.manual_seed(103)
        def new_model():
            model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
            model.yaml = {"toy_cpu_test": True}
            return model
        model = new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = SimpleNamespace(updates=1, total=1.0, o2m=.7, o2o=.3, o2m_copy=.8, final_o2m=.1)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            x, y = torch.randn(4, 3), torch.randn(4, 2)
            loss = (model(x) - y).square().mean()
            loss.backward(); optimizer.step()
            return float(loss.detach())
        update(model, optimizer); update(model, optimizer)
        path = self.root / "epoch_001.pt"
        recipe = {"epochs": 10, "seed": 20261104}
        save_checkpoint(path, model=model, optimizer=optimizer,
            scaler=SimpleNamespace(state_dict=lambda: {"scale": 1024}), criterion=criterion,
            recipe=recipe, initialization={}, completed_epochs=1, global_step=2,
            steps_per_epoch=2, at_epoch_boundary=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        expected_loss = update(model, optimizer)
        expected_parameters = cpu_snapshot(model.state_dict())
        resumed = new_model()
        resumed.load_state_dict(saved["state_dict"], strict=True)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_optimizer.load_state_dict(saved["formal_training"]["optimizer"])
        resumed_criterion = SimpleNamespace()
        restore_criterion(resumed_criterion, saved["formal_training"]["criterion"])
        restore_rng(saved["formal_training"]["rng"])
        actual_loss = update(resumed, resumed_optimizer)
        self.assertEqual(expected_loss, actual_loss)
        for key, value in resumed.state_dict().items():
            self.assertTrue(torch.equal(expected_parameters[key], value), key)
        self.assertEqual(criterion_state(criterion), criterion_state(resumed_criterion))
        validate_resume(saved, recipe)
        partial = copy.deepcopy(saved)
        partial["formal_training"]["at_epoch_boundary"] = False
        with self.assertRaisesRegex(TrainingContractError, "epoch-boundary"):
            validate_resume(partial, recipe)
        with self.assertRaisesRegex(TrainingContractError, "recipe differs"):
            validate_resume(saved, {**recipe, "epochs": 11})

    def test_numerical_migration_is_explicit_hash_allowlisted_and_otherwise_strict(self):
        from apple_local_rgbd.formal_training import CHECKPOINT_SCHEMA, FORMAL_SCHEMA
        old = {"epochs": 10, "lr": 1e-4, "manifest_sha256": "fixed",
               "amp_scale_policy": LEGACY_SCALE_POLICY,
               "sources": {"formal_training.py": LEGACY_FORMAL_SHA256,
                           "local_training.py": LEGACY_ADAPTER_SHA256}}
        new = {**old, "amp_scale_policy": AMP_SCALE_POLICY,
               "numerical_policy": copy.deepcopy(NUMERICAL_POLICY), "sources": source_hashes()}
        saved = {"schema": CHECKPOINT_SCHEMA, "config": old, "formal_training": {
            "schema": FORMAL_SCHEMA, "at_epoch_boundary": True, "recipe": old,
            "completed_epochs": 1, "global_step": 1152, "steps_per_epoch": 1152,
            "criterion": {"updates": 1}}}
        with self.assertRaisesRegex(TrainingContractError, "recipe differs"):
            validate_resume(saved, new)
        receipt = validate_resume(saved, new, upgrade_numerical_policy=True,
                                  checkpoint_sha256=LEGACY_EPOCH1_SHA256)
        self.assertEqual(receipt["old_recipe_sha256"], recipe_sha256(old))
        self.assertEqual(receipt["new_recipe_sha256"], recipe_sha256(new))
        for changes in ({"lr": 2e-4}, {"manifest_sha256": "changed"}, {"epochs": 11}):
            with self.assertRaisesRegex(TrainingContractError, "outside the explicit allowlist"):
                validate_resume(saved, {**new, **changes}, upgrade_numerical_policy=True,
                                checkpoint_sha256=LEGACY_EPOCH1_SHA256)
        with self.assertRaisesRegex(TrainingContractError, "checkpoint/source allowlist"):
            validate_resume(saved, new, upgrade_numerical_policy=True, checkpoint_sha256="unapproved")
        wrong_source = copy.deepcopy(saved)
        wrong_source["formal_training"]["recipe"]["sources"]["local_training.py"] = "changed"
        with self.assertRaisesRegex(TrainingContractError, "checkpoint/source allowlist"):
            validate_resume(wrong_source, new, upgrade_numerical_policy=True,
                            checkpoint_sha256=LEGACY_EPOCH1_SHA256)
        saved["config"] = new
        saved["formal_training"]["recipe"] = new
        self.assertIsNone(validate_resume(saved, new))
        with self.assertRaisesRegex(TrainingContractError, "recipe differs"):
            validate_resume(saved, {**new, "lr": 2e-4})


if __name__ == "__main__":
    unittest.main()
