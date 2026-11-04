"""CPU contract tests; no weights, GPU, training or external dataset needed."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from apple_local_rgbd.local_training import (
    MANIFEST_SCHEMA, NativeVisibleDataset, TrainingContractError, collate_native,
    decode_packed_masks, prepare_raster_sample, sha256_file, tight_boxes, run_smoke,
)


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root, prefix="local_training_test_")
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, *, empty=False, split="train", role="train"):
        v = np.zeros((0 if empty else 1, 64, 64), dtype=np.uint8)
        if not empty:
            v[0, 4:24, 4:24] = 1
            v[0, 10:16, 10:16] = 0  # hole must not be filled
            v[0, 40:52, 40:60] = 1  # second fragment, same apple
        a = v.copy()
        image = self.path / ("empty.png" if empty else "fruit.png")
        Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(image)
        mask = image.with_suffix(".npz")
        np.savez_compressed(mask, visible_packed=np.packbits(v, axis=-1, bitorder="little"),
                            amodal_packed=np.packbits(a, axis=-1, bitorder="little"),
                            original_hw=np.array([64, 64]), instance_ids=np.array([] if empty else ["appleA"], dtype=str),
                            labels=np.zeros(len(v), dtype=np.int64), boxes_visible_xyxy=tight_boxes(v),
                            boxes_amodal_xyxy=tight_boxes(a))
        row = {"split": split, "image_id": image.stem, "rgb_path": str(image), "mask_path": str(mask),
               "rgb_sha256": sha256_file(image), "mask_sha256": sha256_file(mask),
               "original_hw": [64, 64], "instance_count": len(v)}
        manifest = self.path / f"{image.stem}_{split}.json"
        manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "split": split, "role": role, "samples": [row]}))
        return manifest, row, v

    def test_multifragment_hole_and_tight_visible_box_preserved(self):
        manifest, _, v = self.fixture()
        item = NativeVisibleDataset(manifest, imgsz=64, mask_ratio=1)[0]
        np.testing.assert_array_equal(item["masks"].numpy(), v)
        self.assertEqual(item["masks"].shape[0], 1)
        np.testing.assert_array_equal(tight_boxes(v), [[4, 4, 60, 52]])
        self.assertEqual(int(item["masks"][0, 12, 12]), 0)
        self.assertEqual(int(item["masks"][0, 30, 30]), 0)  # no artificial bridge
        self.assertEqual(item["metadata"]["instance_ids"], ["appleA"])
        self.assertFalse(item["metadata"]["amodal_used_in_loss"])

    def test_empty_image_retained_and_collation_indexes_instances(self):
        manifest1, _, _ = self.fixture(empty=True)
        manifest2, _, _ = self.fixture()
        a = NativeVisibleDataset(manifest1, imgsz=64)[0]
        b = NativeVisibleDataset(manifest2, imgsz=64)[0]
        batch = collate_native([a, b])
        self.assertEqual(tuple(batch["img"].shape), (2, 3, 64, 64))
        self.assertEqual(batch["batch_idx"].tolist(), [1])
        self.assertEqual(tuple(batch["masks"].shape), (1, 32, 32))
        self.assertEqual(tuple(batch["sem_masks"].shape), (2, 32, 32))
        self.assertEqual(batch["sem_masks"].dtype, torch.int64)

    def test_validation_and_test_never_accepted_for_training(self):
        for split in ("val", "test"):
            manifest, _, _ = self.fixture(split=split, role="development")
            with self.assertRaisesRegex(TrainingContractError, "VAL and test"):
                NativeVisibleDataset(manifest)

    def test_corrupted_mask_hash_rejected(self):
        manifest, row, _ = self.fixture()
        with Path(row["mask_path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(TrainingContractError, "SHA256 mismatch"):
            NativeVisibleDataset(manifest, imgsz=64)[0]

    def test_disappearing_small_instance_rejected_not_dropped(self):
        v = np.zeros((1, 64, 64), dtype=np.uint8)
        v[0, 1, 1] = 1
        d = {"visible": v, "amodal": v, "instance_ids": ["tiny"], "original_hw": [64, 64]}
        with self.assertRaisesRegex(TrainingContractError, "tiny.*no GT was dropped"):
            prepare_raster_sample(np.zeros((64, 64, 3), dtype=np.uint8), d, imgsz=64, mask_ratio=2,
                                  metadata={"image_id": "tiny_sample"})

    def test_semantic_aux_vanishing_is_audited_separately(self):
        v = np.zeros((1, 64, 64), dtype=np.uint8)
        v[0, 2, 2] = 1  # survives ratio2 but disappears on P3/8 grid
        d = {"visible": v, "amodal": v, "instance_ids": ["tiny"], "original_hw": [64, 64]}
        item = prepare_raster_sample(np.zeros((64, 64, 3), dtype=np.uint8), d, imgsz=64, mask_ratio=2,
                                     metadata={"image_id": "tiny_sample"})
        self.assertEqual(int(item["masks"].sum()), 1)
        self.assertEqual(item["metadata"]["semantic_p3_vanished_ids"], ["tiny"])

    def test_mismatched_native_bbox_is_rejected(self):
        _, row, _ = self.fixture()
        mask = Path(row["mask_path"])
        with np.load(mask, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        fields["boxes_visible_xyxy"][0, 2] -= 1
        np.savez_compressed(mask, **fields)
        with self.assertRaisesRegex(TrainingContractError, "Stored V box differs"):
            decode_packed_masks(mask)

    def test_non_square_letterbox_uses_identical_mask_image_geometry(self):
        v = np.zeros((1, 32, 64), dtype=np.uint8)
        v[0, 8:24, 16:48] = 1
        rgb = np.zeros((32, 64, 3), dtype=np.uint8)
        rgb[8:24, 16:48, 0] = 255
        d = {"visible": v, "amodal": v, "instance_ids": ["wide"], "original_hw": [32, 64]}
        item = prepare_raster_sample(rgb, d, imgsz=64, mask_ratio=1, metadata={"image_id": "wide"})
        self.assertEqual(item["metadata"]["letterbox"]["pad_top_left"], [16, 0])
        np.testing.assert_array_equal(tight_boxes(item["masks"].numpy()), [[16, 24, 48, 40]])
        self.assertEqual(item["bboxes"].tolist(), [[0.5, 0.5, 0.5, 0.25]])
        self.assertEqual(float(item["img"][0, 24, 16]), 1.0)
        self.assertAlmostEqual(float(item["img"][0, 0, 0]), 114 / 255)
        self.assertEqual(item["metadata"]["visible_connected_components_8"]["native"], [1])

    def test_invalid_amp_scale_rejected_before_any_runtime_or_gpu_activity(self):
        for value in (float("nan"), float("inf"), -1.0, 0.0):
            with self.subTest(scale=value), self.assertRaisesRegex(TrainingContractError, "finite and positive"):
                run_smoke(repo=self.path, manifest=self.path / "not_read.json", weights=self.path / "not_read.pt",
                          output=self.path / "not_created", amp_init_scale=value)
        self.assertFalse((self.path / "not_created").exists())


if __name__ == "__main__":
    unittest.main()
