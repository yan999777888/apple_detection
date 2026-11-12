"""Small synthetic CPU tests for the frozen replay semantics; no model/data imports."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/"apple_research_v2"/"replay_budgeted_review.py"
spec = importlib.util.spec_from_file_location("budget_replay", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NAMES = {"0": "a", "1": "b"}


def counts(tp, fp, fn):
    return {"support": tp+fn, "TP": tp, "FP": fp, "FN": fn, "FP_plus_FN": fp+fn,
            "per_class": {
                "0": {"class_name": "a", "support": tp+fn, "TP": tp, "FP": fp, "FN": fn},
                "1": {"class_name": "b", "support": 0, "TP": 0, "FP": 0, "FN": 0},
            }}


def row(identifier, confidence, low, high, low_ms=10, high_ms=20):
    return {"image_id": identifier, "coarse_min_confidence": confidence, "known_group_id": "g_"+identifier,
            "resolutions": {"640": {"box_matching": low, "native_prediction_call_ms": low_ms},
                            "960": {"box_matching": high, "native_prediction_call_ms": high_ms}}}


def samples():
    return [
        row("a", .9, counts(0, 0, 1), counts(1, 0, 0)),  # gain +1
        row("b", 0.0, counts(1, 0, 0), counts(0, 1, 1)), # gain -2
        row("c", .6, counts(1, 0, 0), counts(1, 0, 0)),  # gain 0
        row("d", .6, counts(0, 2, 1), counts(1, 0, 0)),  # gain +3
    ]


class BudgetTests(unittest.TestCase):
    def test_exact_floor_budgets(self):
        self.assertEqual([m.action_count(72, f) for f in m.FRACTIONS], [0, 7, 18, 36, 72])
        self.assertEqual([m.action_count(113, f) for f in m.FRACTIONS], [0, 11, 28, 56, 113])
        self.assertEqual(m.action_count(10, ".3"), 3)

    def test_invalid_budgets(self):
        for f in ("-0.1", "1.1"):
            with self.assertRaises(ValueError):
                m.action_count(10, f)

    def test_no_detections_first_and_image_id_ties(self):
        self.assertEqual(m.rank_ids(list(reversed(samples())), "coarse_min_confidence"), ["b", "c", "d", "a"])

    def test_random_permutation_order_invariant_and_nested(self):
        for seed in m.SEEDS:
            ranked = m.rank_ids(samples(), "random", seed)
            self.assertEqual(ranked, m.rank_ids(list(reversed(samples())), "random", seed))
            self.assertEqual(set(ranked), {"a", "b", "c", "d"})
            self.assertEqual(len(ranked[:m.action_count(4, ".5")]), 2)
            self.assertEqual(ranked[:1], ranked[:2][:1])

    def test_random_unfrozen_seed_rejected(self):
        with self.assertRaises(ValueError):
            m.rank_ids(samples(), "random", 0)

    def test_oracle_positive_only_descending_gain(self):
        self.assertEqual(m.rank_ids(samples(), "GT_oracle_positive_gain"), ["d", "a"])
        self.assertEqual(m.rank_ids(samples(), "GT_oracle_positive_gain")[:1], ["d"])

    def test_oracle_uses_fewer_than_budget(self):
        report, _ = m.evaluate(samples(), NAMES, "GT_oracle_positive_gain",
                              m.rank_ids(samples(), "GT_oracle_positive_gain"), "1")
        self.assertEqual((report["action_budget_K"], report["sequential_rerun_actions"]), (4, 2))
        self.assertEqual(report["risk_reduction_from_640"], 4)
        self.assertTrue(report["uses_GT_to_select"])
        self.assertFalse(report["selection_deployable"])

    def test_sequential_cost_adds_all_coarse_plus_selected_high(self):
        report, records = m.evaluate(samples(), NAMES, "coarse_min_confidence", ["b"], ".25")
        self.assertEqual(report["offline_replay_cost_ms"]["sum"], 4*10+20)
        self.assertEqual([r["replay_cost_ms"] for r in records], [10, 30, 10, 10])

    def test_direct_and_rerun_have_same_counts_different_cost(self):
        direct, _ = m.evaluate(samples(), NAMES, "direct_960", [], direct=True)
        rerun, _ = m.evaluate(samples(), NAMES, "rerun_all", ["a", "b", "c", "d"])
        self.assertEqual(direct["counts"], rerun["counts"])
        self.assertEqual(direct["offline_replay_cost_ms"]["sum"], 80)
        self.assertEqual(rerun["offline_replay_cost_ms"]["sum"], 120)
        self.assertEqual(direct["sequential_rerun_actions"], 0)
        self.assertEqual(direct["output_960_images"], 4)

    def test_negative_benefit_is_retained(self):
        report, _ = m.evaluate(samples(), NAMES, "coarse_min_confidence", ["b"], ".25")
        self.assertEqual(report["selected_rerun_risk_reduction"], -2)
        self.assertEqual(report["risk_reduction_from_640"], -2)
        self.assertEqual(report["selected_gain_sign_counts"], {"positive": 0, "zero": 0, "negative": 1})

    def test_all_gt_and_absent_class_remain(self):
        for selected in ([], ["b"], ["a", "b", "c", "d"]):
            report, _ = m.evaluate(samples(), NAMES, "synthetic", selected)
            self.assertEqual(report["counts"]["support"], 4)
            self.assertEqual(report["counts"]["TP"]+report["counts"]["FN"], 4)
            self.assertEqual(set(report["counts"]["per_class"]), {"0", "1"})
            self.assertIsNone(report["counts"]["per_class"]["1"]["recall"])

    def test_per_class_wrong_class_error_not_hidden(self):
        data = samples()
        data[0]["resolutions"]["640"]["box_matching"]["per_class"]["1"]["FP"] = 2
        data[0]["resolutions"]["640"]["box_matching"]["FP"] += 2
        data[0]["resolutions"]["640"]["box_matching"]["FP_plus_FN"] += 2
        report, _ = m.evaluate(data, NAMES, "baseline_640", [])
        self.assertEqual(report["counts"]["per_class"]["1"]["FP"], 2)
        self.assertEqual(report["counts"]["per_class"]["1"]["support"], 0)

    def test_budget_duplicate_unknown_id_rejected(self):
        for selected, fraction in ((["a", "b"], ".25"), (["a", "a"], "1"), (["z"], "1")):
            with self.assertRaises(ValueError):
                m.evaluate(samples(), NAMES, "x", selected, fraction)

    def test_nearest_rank_not_interpolation(self):
        self.assertEqual(m.nearest_rank([40, 10, 30, 20], .5), 20)
        self.assertEqual(m.nearest_rank([40, 10, 30, 20], .95), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)

