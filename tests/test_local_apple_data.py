"""CPU contract tests for disconnected native-raster instance export."""
import importlib.util
import os
from pathlib import Path
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_local_apple_data", ROOT / "scripts/prepare_local_apple_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class LocalAppleDataTests(unittest.TestCase):
    def test_disconnected_instance_and_hole_survive_packing(self):
        v = np.zeros((1, 15, 19), np.uint8)
        v[0, 1:7, 1:7] = 1
        v[0, 3:5, 3:5] = 0
        v[0, 10:13, 13:17] = 1
        arrays = prepare.encode_masks(v, v, ["7"])
        recovered = np.unpackbits(arrays["visible_packed"], axis=-1, count=19, bitorder="little")
        np.testing.assert_array_equal(recovered, v)
        self.assertEqual(recovered.shape, (1, 15, 19))
        self.assertEqual(prepare.topology(recovered[0])["components_8"], 2)
        self.assertEqual(prepare.topology(recovered[0])["holes_4"], 1)
        np.testing.assert_array_equal(arrays["boxes_visible_xyxy"], [[1, 1, 17, 13]])

    def test_empty_frame_has_valid_zero_instance_schema(self):
        v = np.zeros((0, 15, 19), np.uint8)
        arrays = prepare.encode_masks(v, v, [])
        self.assertEqual(arrays["visible_packed"].shape, (0, 15, 3))
        self.assertEqual(arrays["boxes_visible_xyxy"].shape, (0, 4))
        self.assertEqual(arrays["labels"].dtype, np.int64)

    def test_visible_outside_amodal_is_preserved_not_repaired(self):
        v = np.zeros((1, 10, 10), np.uint8)
        v[0, 1:8, 1:8] = 1
        a = np.zeros_like(v)
        a[0, 2:7, 2:7] = 1
        out = prepare.encode_masks(v, a, ["1"])
        vr = np.unpackbits(out["visible_packed"], axis=-1, count=10, bitorder="little")
        ar = np.unpackbits(out["amodal_packed"], axis=-1, count=10, bitorder="little")
        self.assertEqual(int(vr.sum()), 49)
        self.assertEqual(int(ar.sum()), 25)
        self.assertEqual(prepare.visibility_stratum(49 / 25), "annotation_visible_area_exceeds_amodal")

    def test_original_polygon_union_keeps_one_ID_and_two_fragments(self):
        polys = [np.array([[1, 1], [3, 1], [3, 3], [1, 3]]), np.array([[7, 7], [9, 7], [9, 9], [7, 9]])]
        mask = prepare._rasterize(polys, 12, 12)
        self.assertEqual(prepare.topology(mask)["components_8"], 2)
        arrays = prepare.encode_masks(mask[None], mask[None], ["7"])
        self.assertEqual(arrays["instance_ids"].tolist(), ["7"])

    def test_identity_includes_year_and_capture_excludes_crop_index(self):
        self.assertEqual(prepare.apple_identity("_MG_2821_07.png", "9"), ("2018_01_009", "2018_2821"))
        self.assertEqual(prepare.apple_identity("_MG_6821_07.png", "9"), ("2020_01_009", "2020_6821"))
        self.assertNotEqual(prepare.apple_identity("_MG_2821_07.png", "9")[0], prepare.apple_identity("_MG_6821_07.png", "9")[0])

    def test_bad_ID_count_fails_instead_of_dropping_targets(self):
        with self.assertRaises(prepare.DataContractError):
            prepare.encode_masks(np.ones((2, 4, 4), np.uint8), np.ones((2, 4, 4), np.uint8), ["1"])

    def test_stratum_boundaries_match_original_strong_model_contract(self):
        self.assertEqual(prepare.visibility_stratum(.399999), "heavy_visible_lt_0.40")
        self.assertEqual(prepare.visibility_stratum(.4), "moderate_visible_0.40_to_0.70")
        self.assertEqual(prepare.visibility_stratum(.7), "light_visible_0.70_to_0.90")
        self.assertEqual(prepare.visibility_stratum(.9), "mostly_visible_0.90_to_1.00")
        self.assertEqual(prepare.visibility_stratum(1.), "mostly_visible_0.90_to_1.00")
        self.assertEqual(prepare.visibility_stratum(1.000001), "annotation_visible_area_exceeds_amodal")


if __name__ == "__main__":
    unittest.main()
