"""CPU-only metric contracts, independent of GPU and model accuracy."""
import unittest
import numpy as np

from apple_local_rgbd.local_evaluation import (
    box_iou, confidence_metrics, greedy_match, native_mask_iou, native_mask_areas_boxes,
    stratified_gt_metrics,
)


class LocalEvaluationTests(unittest.TestCase):
    def test_mask_intersection_roi_equals_full_native_logic(self):
        rng = np.random.default_rng(74)
        pred = (rng.random((5, 31, 37)) > .8).astype(np.uint8)
        gt = (rng.random((4, 31, 37)) > .7).astype(np.uint8)
        pred[0] = 0
        pred[1, 12:] = 0
        gt[0, :16] = 0
        expected = np.zeros((5, 4))
        for i in range(5):
            for j in range(4):
                union = np.count_nonzero(pred[i] | gt[j])
                expected[i, j] = np.count_nonzero(pred[i] & gt[j]) / union if union else 0
        np.testing.assert_array_equal(native_mask_iou(pred, gt), expected)

    def test_disconnected_masks_and_holes_use_all_pixels(self):
        v = np.zeros((1, 12, 15), dtype=np.uint8)
        v[0, 1:5, 1:5] = 1
        v[0, 2:4, 2:4] = 0
        v[0, 8:10, 12:14] = 1
        area, boxes = native_mask_areas_boxes(v)
        self.assertEqual(area.tolist(), [16])
        self.assertEqual(boxes.tolist(), [[1, 1, 14, 10]])
        self.assertEqual(native_mask_iou(v, v).tolist(), [[1.]])

    def test_empty_images_all_detections_are_false_positives(self):
        matched, iou = greedy_match(np.array([.8, .4]), np.zeros((2, 0)))
        self.assertEqual(matched.tolist(), [-1, -1])
        self.assertEqual(iou.tolist(), [0., 0.])
        metrics = confidence_metrics([.8, .4], [False, False], 0)
        self.assertEqual(metrics["fixed"]["FP"], 2)
        self.assertEqual(metrics["AP50_101point"], 0)
        self.assertIsNone(metrics["P_at_R90"])

    def test_matching_is_score_first_and_never_duplicates_gt(self):
        # Highest IoU alone would pick prediction 0, but prediction 1 has a higher score.
        iou = np.array([[.95, .05], [.60, .1], [.1, .8]])
        matched, quality = greedy_match(np.array([.5, .9, .6]), iou)
        self.assertEqual(matched.tolist(), [-1, 0, 1])
        self.assertEqual(quality.tolist(), [0., .6, .8])

    def test_threshold_is_inclusive_and_float16_ties_not_split(self):
        # A tie containing one TP and one FP cannot be evaluated at precision=1.
        metrics = confidence_metrics([.5, .5], [True, False], 1, fixed_threshold=.5)
        self.assertEqual(metrics["fixed"]["precision"], .5)
        self.assertEqual(len(metrics["curve"]), 1)
        self.assertEqual(metrics["P_at_R90"]["precision"], .5)
        self.assertEqual(metrics["P_at_R90"]["recall"], 1.)
        self.assertEqual(metrics["AP50_101point"], .5)

    def test_recall90_unavailable_when_prediction_floor_misses_gt(self):
        metrics = confidence_metrics([.9, .8], [True, False], 2)
        self.assertEqual(metrics["maximum_recall"], .5)
        self.assertIsNone(metrics["P_at_R90"])
        self.assertAlmostEqual(metrics["AP50_101point"], 51 / 101)

    def test_perfect_ap_and_no_prediction_case(self):
        self.assertEqual(confidence_metrics([.9, .8], [True, True], 2)["AP50_101point"], 1.)
        empty = confidence_metrics([], [], 2)
        self.assertEqual(empty["fixed"]["FN"], 2)
        self.assertEqual(empty["AP50_101point"], 0.)

    def test_heavy_group_fn_is_not_hidden_by_matched_mean(self):
        gt = [{"key": "a", "visibility_stratum": "heavy"}, {"key": "b", "visibility_stratum": "heavy"},
              {"key": "c", "visibility_stratum": "light"}]
        pred = [{"score": .9, "gt": "a", "iou": .7}, {"score": .2, "gt": "b", "iou": .9},
                {"score": .8, "gt": "c", "iou": .95}]
        result = stratified_gt_metrics(gt, pred, .25, "gt", "iou")
        self.assertEqual(result["heavy"]["TP"], 1)
        self.assertEqual(result["heavy"]["FN"], 1)
        self.assertEqual(result["heavy"]["recall"], .5)
        self.assertEqual(result["heavy"]["matched_mean_iou"], .7)
        self.assertEqual(result["heavy"]["missed_GT_keys"], ["b"])

    def test_visible_bbox_metric_uses_half_open_area(self):
        np.testing.assert_allclose(box_iou(np.array([[0, 0, 2, 2]]), np.array([[1, 1, 3, 3]])), [[1/7]])
        self.assertEqual(box_iou(np.zeros((0, 4)), np.zeros((2, 4))).shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
