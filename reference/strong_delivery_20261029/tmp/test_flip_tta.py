import unittest

import numpy as np

import flip_tta


class FlipBoxesTest(unittest.TestCase):
    def test_horizontal_exclusive_xyxy(self):
        boxes = np.array([[1., 2., 4., 6.]])
        expected = np.array([[6., 2., 9., 6.]])
        np.testing.assert_allclose(flip_tta.flip_boxes(boxes, (20, 10), "h"), expected)

    def test_vertical_exclusive_xyxy(self):
        boxes = np.array([[1., 2., 4., 6.]])
        expected = np.array([[1., 14., 4., 18.]])
        np.testing.assert_allclose(flip_tta.flip_boxes(boxes, (20, 10), "v"), expected)

    def test_both_axes_exclusive_xyxy(self):
        boxes = np.array([[1., 2., 4., 6.]])
        expected = np.array([[6., 14., 9., 18.]])
        np.testing.assert_allclose(flip_tta.flip_boxes(boxes, (20, 10), "hv"), expected)

    def test_round_trip(self):
        boxes = np.array([[0., 1., 4., 6.], [7., 8., 10., 20.]])
        for axis in ("h", "v", "hv"):
            flipped = flip_tta.flip_boxes(boxes, (20, 10), axis)
            np.testing.assert_allclose(flip_tta.flip_boxes(flipped, (20, 10), axis), boxes)


class FusionTest(unittest.TestCase):
    @staticmethod
    def entry(boxes, scores):
        return {"pred": {"boxes": boxes, "scores": scores}}

    def test_cross_view_box_average_and_mean_score(self):
        layers = [self.entry([[0., 0., 10., 10.]], [0.8]),
                  self.entry([[2., 0., 12., 10.]], [0.6])]
        boxes, scores = flip_tta.fuse_entries(layers, .55, "mean", "score_weighted")
        np.testing.assert_allclose(boxes, [[2. * .6 / 1.4, 0., 10. + 2. * .6 / 1.4, 10.]])
        np.testing.assert_allclose(scores, [.7])

    def test_unmatched_view_prediction_is_retained(self):
        layers = [self.entry([[0., 0., 10., 10.]], [.8]),
                  self.entry([[30., 30., 40., 40.]], [.7])]
        boxes, scores = flip_tta.fuse_entries(layers, .55, "mean", "score_weighted")
        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_allclose(scores, [.8, .7])

    def test_at_most_one_member_per_view(self):
        layers = [self.entry([[0., 0., 10., 10.], [1., 0., 11., 10.]], [.9, .8]),
                  self.entry([[1., 0., 11., 10.]], [.85])]
        boxes, scores = flip_tta.fuse_entries(layers, .5, "mean", "representative")
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], .875)
        self.assertAlmostEqual(scores[1], .8)

    def test_coverage_floor_only_penalises_singletons(self):
        layers = [self.entry([[0., 0., 10., 10.], [20., 20., 30., 30.]], [.8, .9]),
                  self.entry([[0., 0., 10., 10.]], [.6])]
        _, default = flip_tta.fuse_entries(layers, .5, "mean", "score_weighted")
        _, floored = flip_tta.fuse_entries(layers, .5, "mean", "score_weighted", .7)
        np.testing.assert_allclose(np.sort(default), [.7, .9])
        np.testing.assert_allclose(np.sort(floored), [.9 * .7, .7])

    def test_empty_views(self):
        boxes, scores = flip_tta.fuse_entries([self.entry([], []), self.entry([], [])])
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))

    def test_exact_metric_integration(self):
        base = [{"image_id": "a", "gt": {"boxes": [[0., 0., 10., 10.]]},
                 "pred": {"boxes": [[0., 0., 10., 10.]], "scores": [.8]}}]
        result = flip_tta.evaluate(base)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["min_precision_recall"], 1.)


if __name__ == "__main__":
    unittest.main()
