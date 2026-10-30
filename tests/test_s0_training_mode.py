"""PApple-only isolation, complete sampler state and checkpoint selection checks."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apple_seg.training import (CyclingOrder, PAPPLE_SCHEMA, SCHEMA, SOURCE_ORDER,
                                _datasets, _papple_capacity, load_config, select_validation_point)


@contextmanager
def directory():
    # All temporary test files remain on the repository's F: drive.
    parent = Path(__file__).absolute().parents[1] / "reports"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s0_contract_test_", dir=parent) as path:
        yield Path(path)


def config_at(root, mode="papple_only", **recipe):
    paths = {"papple_root": "PApple", "parent_checkpoint": "weights/query.pt",
             "papple_provenance": "provenance.json", "output": "runs/s0"}
    if mode == "mixed":
        paths.update(minne_root="MinneApple", minne_manifest="minne.json")
    path = root / "recipe.json"
    path.write_text(json.dumps({"path_base": ".", "paths": paths,
                               "parent_sha256": "a" * 64,
                               "training": {"data_mode": mode, **recipe}}), encoding="utf-8")
    return path


class S0DataModeTests(unittest.TestCase):
    def test_papple_only_requires_no_minne_paths_and_binds_four_full_samples(self):
        with directory() as root:
            config = load_config(config_at(root))
        self.assertEqual(config["source_order"], ("PApple",) * 4)
        self.assertEqual(config["checkpoint_schema"], PAPPLE_SCHEMA)
        self.assertNotIn("minne_root", config["paths"])
        self.assertNotIn("minne_manifest", config["paths"])

    def test_existing_mixed_plan_stays_two_sources_and_separate_schema(self):
        with directory() as root:
            config = load_config(config_at(root, "mixed"))
        self.assertEqual(config["source_order"], SOURCE_ORDER)
        self.assertEqual(config["checkpoint_schema"], SCHEMA)
        self.assertNotEqual(PAPPLE_SCHEMA, SCHEMA)
        self.assertIn("minne_root", config["paths"])

    def test_papple_loader_never_constructs_minne_in_s0(self):
        with patch("apple_seg.data.PAppleDataset", return_value=object()) as papple, \
             patch("apple_seg.data.MinneAppleDataset", side_effect=AssertionError("Minne accessed")) as minne:
            loaded = _datasets({"papple_root": Path("F:/PApple")}, 768, "papple_only")
        self.assertEqual(set(loaded), {"PApple"})
        papple.assert_called_once_with(Path("F:/PApple"), split="train", input_size=768)
        minne.assert_not_called()

    def test_fixed_stress_indices_cannot_be_used_for_formal_training(self):
        with directory() as root:
            path = config_at(root, smoke_sample_indices=[0, 1, 4, 9])
            with self.assertRaisesRegex(ValueError, "never formal"):
                load_config(path)
            config = load_config(path, smoke=True)
            self.assertEqual(config["recipe"]["smoke_sample_indices"], [0, 1, 4, 9])
            path = config_at(root, "mixed", smoke_sample_indices=[0, 1, 4, 9])
            with self.assertRaisesRegex(ValueError, "PApple-only"):
                load_config(path, smoke=True)

    def test_seed_shuffle_and_smoke_indices_validation(self):
        for arguments in ({"seed": -1}, {"seed": True}, {"shuffle": "false"},
                          {"smoke_sample_indices": [0, 1, True, 3]},
                          {"model_selection": "test_best"}):
            with self.subTest(arguments=arguments), directory() as root:
                with self.assertRaises(ValueError):
                    load_config(config_at(root, **arguments), smoke=True)

    def test_unshuffled_sampler_roundtrip_preserves_next_update_and_cycle(self):
        original = CyclingOrder(7, 123, shuffle=False)
        self.assertEqual([original.next() for _ in range(5)], list(range(5)))
        saved = deepcopy(original.state_dict())
        restored = CyclingOrder(7, 999, shuffle=False)
        restored.load_state_dict(saved)
        self.assertEqual([original.next() for _ in range(16)], [restored.next() for _ in range(16)])
        self.assertEqual(original.cycles, restored.cycles)
        with self.assertRaisesRegex(ValueError, "shuffle"):
            CyclingOrder(7, 123, shuffle=True).load_state_dict(saved)

    def test_capacity_audit_retains_empty_and_over_capacity_instances(self):
        class Inventory:
            samples = [{"instance_ids": []}, {"instance_ids": list(range(101))}]
            def __len__(self):
                return 2
        result = _papple_capacity(Inventory())
        self.assertEqual(result["GT_instances"], 101)
        self.assertEqual(result["empty_images"], 1)
        self.assertEqual(result["over_capacity_indices"], [1])
        self.assertFalse(result["formal_training_compatible"])
        self.assertFalse(result["GT_truncated_or_skipped"])


class S0ValidationSelectionTests(unittest.TestCase):
    def evaluation(self):
        return {"confidence": .31, "score": .92,
                "summary": {"precision": .92, "recall": .93, "f1": .925,
                            "images": 814, "gt_count": 3449, "empty_gt_images": 13,
                            "visible_instance_mask_pr_iou50": {"precision": .8}}}

    def test_r90_selection_rejects_high_precision_with_insufficient_recall_and_separates_masks(self):
        rows = [dict(confidence=.8, precision=.99, recall=.89, f1=.937, tp=3069, fp=31, fn=380, gt_count=3449),
                dict(confidence=.7, precision=.945, recall=.90, f1=.922, tp=3105, fp=181, fn=344, gt_count=3449),
                dict(confidence=.6, precision=.945, recall=.91, f1=.927, tp=3139, fp=183, fn=310, gt_count=3449)]
        with directory() as root:
            (root / "box_threshold_curve.json").write_text(json.dumps(rows), encoding="utf-8")
            diagnostics = [{"scores": [.8, .65, .4], "visible_mask_gt_indices_by_prediction": [0, -1, 1],
                            "amodal_mask_gt_indices_by_prediction": [0, 1, -1]}]
            (root / "mask_pair_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
            result = select_validation_point(self.evaluation(), root, "precision_at_recall90")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["confidence"], .6)
        self.assertEqual(result["score"], .945)
        visible = result["summary"]["visible_instance_mask_pr_iou50"]
        amodal = result["summary"]["amodal_instance_mask_pr_iou50"]
        self.assertEqual((visible["tp"], visible["fp"]), (1, 1))
        self.assertEqual((amodal["tp"], amodal["fp"]), (2, 0))
        self.assertEqual(visible["confidence"], .6)
        self.assertEqual(amodal["confidence"], .6)
        self.assertFalse(visible["is_mask_ap"])
        self.assertEqual(len(result["mask_diagnostics_sha256"]), 64)

    def test_unreached_r90_is_ineligible_not_extrapolated(self):
        with directory() as root:
            (root / "box_threshold_curve.json").write_text(json.dumps([dict(recall=.89)]), encoding="utf-8")
            result = select_validation_point(self.evaluation(), root, "precision_at_recall90")
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["max_recall"], .89)

    def test_balanced_selection_keeps_original_mask_workpoint(self):
        evaluated = self.evaluation()
        result = select_validation_point(evaluated, Path("F:/unused"), "balanced_min_pr")
        self.assertEqual(result["summary"], evaluated["summary"])
        self.assertEqual(result["confidence"], .31)


if __name__ == "__main__":
    unittest.main(verbosity=2)
