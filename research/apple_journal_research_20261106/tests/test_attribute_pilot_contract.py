"""Metadata-only contract fixtures; no image creation, model inference or training."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "apple_research_v2/train_attribute_pilot.py"
SPEC = importlib.util.spec_from_file_location("attribute_pilot", SCRIPT)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def row(name, label="appearance_a", split="train", group=None, source=None):
    return {"image_path": name + ".png", "bbox_xyxy": [1, 2, 11, 12], "label": label,
            "source_image_id": source or name, "group_id": group or name, "split": split}


class ManifestContract(unittest.TestCase):
    def validate(self, records, **meta):
        return PILOT.validate_manifest(dict(records=records, label_semantics="Metadata-only fixture categories; never training data", **meta), Path.cwd())

    def base(self):
        return [row("a"), row("b", "appearance_b")]

    def test_real_label_mapping_and_no_implicit_validation(self):
        records = self.base() + [row("v", split="val")]
        _, train, val, names, independent = self.validate(records)
        self.assertEqual(names, ["appearance_a", "appearance_b"])
        self.assertEqual(len(train), 2)
        self.assertEqual(val, [])
        self.assertFalse(independent)

    def test_documented_disjoint_holdout(self):
        result = self.validate(self.base() + [row("v", split="val")],
            split_policy={"independent_group_holdout": True,
                          "evidence": "fixture grouping evidence", "group_definition": "fixture source groups"})
        self.assertEqual(len(result[2]), 1)
        self.assertTrue(result[4])

    def test_test_records_rejected_before_any_file_access(self):
        with self.assertRaisesRegex(ValueError, "test"):
            self.validate(self.base() + [row("nonexistent_test", split="test")])

    def test_label_semantics_required(self):
        with self.assertRaisesRegex(ValueError, "label_semantics"):
            PILOT.validate_manifest({"records": self.base()}, Path.cwd())

    def test_missing_label(self):
        records = self.base(); del records[0]["label"]
        with self.assertRaisesRegex(ValueError, "missing label"):
            self.validate(records)

    def test_numeric_label_requires_names(self):
        with self.assertRaisesRegex(ValueError, "class_names"):
            self.validate([row("a", 0), row("b", 1)])

    def test_numeric_labels_are_not_reinterpreted(self):
        result = self.validate([row("a", 0), row("b", 1)], class_names=["provided_A", "provided_B"])
        self.assertEqual(result[3], ["provided_A", "provided_B"])
        self.assertEqual([r["label_id"] for r in result[1]], [0, 1])

    def test_group_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "group crosses"):
            self.validate(self.base() + [row("v", split="val", group="a")])

    def test_source_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "source image crosses"):
            self.validate(self.base() + [row("v", split="val", source="a")])

    def test_duplicate_roi_even_if_conflicting_label(self):
        with self.assertRaisesRegex(ValueError, "duplicated ROI"):
            self.validate(self.base() + [row("a", "appearance_b")])

    def test_invalid_bbox(self):
        for box in ([2, 2, 1, 3], [-1, 0, 2, 2], [0, 0, float("nan"), 2]):
            records = self.base(); records[0]["bbox_xyxy"] = box
            with self.assertRaises(ValueError):
                self.validate(records)

    def test_training_class_must_exist(self):
        with self.assertRaisesRegex(ValueError, "real training"):
            self.validate(self.base(), class_names=["appearance_a", "appearance_b", "unobserved"])

    def test_mean_and_p95_not_mean_instantaneous_fps(self):
        result = PILOT.latency_summary(range(1, 41))
        self.assertEqual(result["mean_ms"], 20.5)
        self.assertAlmostEqual(result["p95_ms"], 38.05)
        self.assertAlmostEqual(result["ROIs_per_second"], 1000 / 20.5)
        for values in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                PILOT.latency_summary(values)


if __name__ == "__main__":
    unittest.main()
