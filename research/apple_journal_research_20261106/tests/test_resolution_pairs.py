"""CPU-only semantic tests; no model import, GPU, dataset mutation or downloads."""
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1]/"apple_research_v2/evaluate_resolution_pairs.py"
spec = importlib.util.spec_from_file_location("resolution_pairs",MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NAMES = {0:"immature",1:"semi",2:"mature"}


def gt(identifier="a",label=0,box=None):
    return {"GT_id":identifier,"class_id":label,"box_xyxy":box or [0,0,10,10]}


def pred(score=.9,label=0,box=None):
    return {"class_id":label,"score":score,"box_xyxy":box or [0,0,10,10]}


class MatchTests(unittest.TestCase):
    def test_exact_tp_and_absent_class_support(self):
        r=m.match_boxes([pred()],[gt()],NAMES)
        self.assertEqual((r["TP"],r["FP"],r["FN"]),(1,0,0))
        self.assertEqual(r["per_class"][1]["support"],0)
        self.assertEqual(r["per_class"][2]["support"],0)

    def test_wrong_class_counts_fp_and_fn(self):
        r=m.match_boxes([pred(label=2)],[gt(label=0)],NAMES)
        self.assertEqual((r["TP"],r["FP"],r["FN"],r["FP_plus_FN"]),(0,1,1,2))
        self.assertEqual(r["per_class"][2]["FP"],1)
        self.assertEqual(r["per_class"][0]["FN"],1)

    def test_two_predictions_cannot_share_gt(self):
        r=m.match_boxes([pred(.7),pred(.9)],[gt()],NAMES)
        self.assertEqual((r["TP"],r["FP"],r["FN"]),(1,1,0))
        self.assertEqual(r["matches"][0]["prediction_index"],1)

    def test_one_prediction_cannot_share_gt(self):
        r=m.match_boxes([pred()],[gt("a"),gt("b")],NAMES)
        self.assertEqual((r["TP"],r["FP"],r["FN"]),(1,0,1))
        self.assertEqual(r["matches"][0]["GT_id"],"a")

    def test_inclusive_score_and_iou_boundaries(self):
        r=m.match_boxes([pred(.25,box=[0,0,5,10])],[gt()],NAMES)
        self.assertEqual(r["TP"],1)
        self.assertEqual(r["matches"][0]["IoU"],.5)
        r=m.match_boxes([pred(.249999999)],[gt()],NAMES)
        self.assertEqual((r["TP"],r["FP"],r["FN"]),(0,0,1))

    def test_score_tie_uses_prediction_order(self):
        r=m.match_boxes([pred(.8),pred(.8)],[gt()],NAMES)
        self.assertEqual(r["matches"][0]["prediction_index"],0)

    def test_best_unused_same_class_gt(self):
        r=m.match_boxes([pred(box=[2,0,12,10]),pred(box=[0,0,10,10])],
                        [gt("a"),gt("b",box=[2,0,12,10])],NAMES)
        self.assertEqual([v["GT_id"] for v in r["matches"]],["b","a"])

    def test_empty_image_and_empty_predictions(self):
        a=m.match_boxes([pred()],[],NAMES)
        b=m.match_boxes([],[gt()],NAMES)
        c=m.match_boxes([],[],NAMES)
        self.assertEqual((a["TP"],a["FP"],a["FN"]),(0,1,0))
        self.assertEqual((b["TP"],b["FP"],b["FN"]),(0,0,1))
        self.assertEqual(c["FP_plus_FN"],0)

    def test_negative_delta_and_gt_changes_preserved(self):
        low=m.match_boxes([pred()],[gt()],NAMES)
        high=m.match_boxes([pred(label=2)],[gt()],NAMES)
        d=m.pair_delta(low,high)
        self.assertEqual(d["sign"],"negative")
        self.assertEqual(d["risk_reduction_640_minus_960"],-2)
        self.assertEqual(d["lost_GT_ids"],["a"])
        self.assertTrue(d["GT_oracle_uses_labels_not_deployable"])

    def test_positive_and_zero_delta(self):
        low=m.match_boxes([],[gt()],NAMES)
        high=m.match_boxes([pred()],[gt()],NAMES)
        self.assertEqual(m.pair_delta(low,high)["recovered_GT_ids"],["a"])
        self.assertEqual(m.pair_delta(low,high)["sign"],"positive")
        self.assertEqual(m.pair_delta(high,high)["sign"],"zero")

    def test_totals_equal_class_support_and_do_not_drop_absent_classes(self):
        a=m.match_boxes([pred()],[gt()],NAMES)
        b=m.match_boxes([pred(label=2)],[gt(label=1)],NAMES)
        s=m.counts_summary([a,b],NAMES)
        self.assertEqual((s["support"],s["TP"],s["FP"],s["FN"]),(2,1,1,1))
        self.assertEqual(s["per_class"]["1"]["support"],1)
        self.assertIsNone(s["per_class"]["2"]["recall"])
        self.assertEqual(s["per_class"]["2"]["FP"],1)

    def test_backend_float32_threshold_preserves_inclusive_quarter(self):
        import struct
        floor=m.float32_predecessor(.25)
        self.assertLess(floor,.25)
        bits=struct.unpack(">I",struct.pack(">f",floor))[0]
        next_value=struct.unpack(">f",struct.pack(">I",bits+1))[0]
        self.assertEqual(next_value,.25)
        self.assertFalse(floor > m.BACKEND_CONF)
        self.assertTrue(.25 > m.BACKEND_CONF)
        with self.assertRaises(ValueError):
            m.float32_predecessor(0)

    def test_test_paths_are_rejected_before_read(self):
        with self.assertRaises(ValueError):
            m.reject_test_path(Path("F:/data/test/images/a.jpg"))
        m.reject_test_path(Path("F:/data/val/images/a.jpg"))

    def test_duplicate_gt_ids_rejected(self):
        with self.assertRaises(ValueError):
            m.match_boxes([pred()],[gt("a"),gt("a")],NAMES)

    def test_bad_inputs_rejected(self):
        for item in (pred(float('nan')),pred(label=99),pred(box=[0,0,0,10])):
            with self.assertRaises(ValueError):
                m.match_boxes([item],[gt()],NAMES)
        with self.assertRaises(ValueError):
            m.names_dict({0:"a",2:"b"})


if __name__ == "__main__":
    unittest.main()
