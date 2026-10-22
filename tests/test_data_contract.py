"""CPU data checks; real assets are optional in a source-only Git checkout.

The portable bundle uses assets/ by default. Environment variables may point
these tests at the same original assets elsewhere, without embedding a drive.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
from PIL import Image
import torch

from apple_seg.data import (DataContractError, MINNE_TILES, MinneAppleDataset,
                            PAppleDataset, collate_single, flip_item_horizontal)

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("APPLE_SEG_ASSETS", REPO / "assets"))
PAPPLE = Path(os.environ.get("APPLE_SEG_PAPPLE_ROOT", ASSETS / "papple"))
MINNE = Path(os.environ.get("APPLE_SEG_MINNE_ROOT", ASSETS / "minneapple"))
MANIFEST = Path(os.environ.get("APPLE_SEG_MINNE_MANIFEST", ASSETS / "minne_manifest.json"))
PROVENANCE = Path(os.environ.get("APPLE_SEG_PAPPLE_PROVENANCE", ASSETS / "papple_val_provenance.json"))


def digest(value):
    return hashlib.sha256(memoryview(value.contiguous().numpy())).hexdigest()


class SmallContractTests(unittest.TestCase):
    def test_unpaired_ID_union_is_diagnosed_not_intersected_or_fabricated(self):
        temporary_base = Path(os.environ.get("APPLE_SEG_TEST_TMP", REPO / "runs/cpu_data_test_tmp"))
        temporary_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_base) as directory:
            root = Path(directory)
            (root / "images/train").mkdir(parents=True)
            (root / "gt_json/train").mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(root / "images/train/fixture.png")
            def via(instance_id):
                return {"fixture": {"filename": "fixture.png", "regions": [{
                    "region_attributes": {"apple_ID": instance_id},
                    "shape_attributes": {"all_points_x": [1, 4, 1], "all_points_y": [1, 1, 4]}}]}}
            for branch, instance_id in (("instance", "a"), ("amodal", "b")):
                (root / f"gt_json/train/via_region_data_{branch}.json").write_text(json.dumps(via(instance_id)), encoding="utf-8")
            dataset = PAppleDataset(root, input_size=64)
            self.assertEqual(dataset.samples[0]["instance_ids"], ["a", "b"])
            with self.assertRaisesRegex(DataContractError, "Unpaired"):
                dataset[0]

    def test_fixed_grid_is_not_selected_from_GT(self):
        self.assertEqual(MINNE_TILES, ((0, 0, 512, 512), (208, 0, 720, 512),
                                      (0, 384, 512, 896), (208, 384, 720, 896),
                                      (0, 768, 512, 1280), (208, 768, 720, 1280)))
        with self.assertRaises(ValueError):
            collate_single([])


@unittest.skipUnless(PAPPLE.is_dir() and MINNE.is_dir() and MANIFEST.is_file() and PROVENANCE.is_file(),
                     "Provide original assets or APPLE_SEG_* dataset paths")
class RealDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("CPU data tests require CUDA_VISIBLE_DEVICES=-1")
        torch.set_num_threads(2)
        cls.provenance = json.loads(PROVENANCE.read_text(encoding="utf-8-sig"))

    def assert_item(self, item, size=768):
        self.assertEqual(item["pixel_values"].shape, (3, size, size))
        self.assertEqual(item["pixel_values"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(item["pixel_values"]).all())
        self.assertTrue(item["pixel_mask"].all())
        target = item["target"]
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["labels"].device.type, "cpu")
        self.assertTrue((target["labels"] == 0).all())
        self.assertEqual(target["visible_masks"].shape, (len(target["instance_ids"]), *target["original_hw"]))
        self.assertEqual(target["visible_masks"].dtype, torch.uint8)
        self.assertTrue(Path(target["source"]["rgb_path"]).is_file())
        self.assertEqual(set(target["source"]["annotation_sha256"]), {"visible", "amodal"})
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_PApple_train_and_VAL_original_mask_hashes_and_anomaly(self):
        train = PAppleDataset(PAPPLE)
        self.assertEqual(len(train), 2304)
        self.assert_item(train[0])
        val = PAppleDataset(PAPPLE, split="val")
        self.assertEqual(len(val), 814)
        self.assertEqual(val.metadata()["GT_instances"], 3449)
        self.assertEqual(val.metadata()["empty_GT_images"], 13)
        names = [sample["filename"] for sample in val.samples]
        anomaly = next(row for row in self.provenance["ground_truth_instances"]
                       if row["occlusion_stratum"] == "annotation_visible_area_exceeds_amodal")
        for name in dict.fromkeys((names[0], anomaly["image_id"])):
            item = val[names.index(name)]
            self.assert_item(item)
            target = item["target"]
            rows = [row for row in self.provenance["ground_truth_instances"] if row["image_id"] == name]
            self.assertEqual(target["instance_ids"], [row["apple_id"] for row in rows])
            self.assertEqual(target["annotation_validity"], {"visible": True, "amodal": True})
            for index, row in enumerate(rows):
                self.assertEqual(digest(target["visible_masks"][index]), row["visible_mask_uint8_sha256"])
                self.assertEqual(digest(target["amodal_masks"][index]), row["amodal_mask_uint8_sha256"])
            if name == anomaly["image_id"]:
                g = target["instance_ids"].index(anomaly["apple_id"])
                self.assertGreater(int(target["visible_masks"][g].sum()), int(target["amodal_masks"][g].sum()))
        empty = next(i for i, sample in enumerate(val.samples) if not sample["instance_ids"])
        self.assertEqual(tuple(val[empty]["target"]["visible_masks"].shape), (0, 1300, 1300))

    def test_real_Minne_six_tiles_keep_onepixel_ID_and_exact_union(self):
        dataset = MinneAppleDataset(MINNE, MANIFEST)
        self.assertEqual(len(dataset), 570 * 6)
        name = "20150921_131234_image451.png"
        original = next(i for i, sample in enumerate(dataset.samples) if sample["source_filename"] == name)
        raw, _ = dataset._idmap(dataset.samples[original])
        reconstructed = np.zeros_like(raw)
        seen = set()
        found_onepixel = False
        for tile, crop in enumerate(MINNE_TILES):
            item = dataset[original * 6 + tile]
            self.assert_item(item)
            target = item["target"]
            self.assertIsNone(target["amodal_masks"])
            self.assertEqual(target["annotation_validity"], {"visible": True, "amodal": False})
            self.assertEqual(target["source"]["crop_xyxy"], list(crop))
            x1, y1, x2, y2 = crop
            for index, instance_id in enumerate(target["instance_ids"]):
                mask = target["visible_masks"][index].numpy()
                np.testing.assert_array_equal(mask, raw[y1:y2, x1:x2] == int(instance_id))
                reconstructed[y1:y2, x1:x2][mask.astype(bool)] = int(instance_id)
                seen.add(int(instance_id))
                if instance_id == "12":
                    self.assertEqual(int(mask.sum()), 1)
                    found_onepixel = True
            twice = flip_item_horizontal(flip_item_horizontal(item))
            self.assertTrue(torch.equal(twice["pixel_values"], item["pixel_values"]))
            self.assertTrue(torch.equal(twice["target"]["visible_masks"], target["visible_masks"]))
            self.assertIsNone(twice["target"]["amodal_masks"])
            self.assertFalse(twice["target"]["source"]["horizontal_flip"])
        self.assertTrue(found_onepixel)
        self.assertEqual(seen, set(int(v) for v in np.unique(raw) if v != 0))
        np.testing.assert_array_equal(reconstructed, raw)

    def test_real_Minne_dense_validation_stays_whole_and_GT_unfiltered(self):
        dataset = MinneAppleDataset(MINNE, MANIFEST, split="val", tiled=False)
        self.assertEqual(len(dataset), 100)
        index = max(range(len(dataset.samples)), key=lambda i: dataset.samples[i]["instances"])
        item = dataset[index]
        self.assert_item(item)
        self.assertEqual(len(item["target"]["labels"]), 123)
        self.assertEqual(item["target"]["original_hw"], [1280, 720])
        self.assertIsNone(item["target"]["amodal_masks"])
        self.assertEqual(item["target"]["source"]["crop_xyxy"], [0, 0, 720, 1280])
        papple = PAppleDataset(PAPPLE)[0]
        twice = flip_item_horizontal(flip_item_horizontal(papple))
        self.assertTrue(torch.equal(twice["target"]["amodal_masks"], papple["target"]["amodal_masks"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
