import unittest

import numpy as np

import crop_tta


class CropGeometryTest(unittest.TestCase):
    def test_tiles_cover_view_with_overlap(self):
        windows = crop_tta.tile_windows(1024, 1024, .625)
        self.assertEqual(windows, [(0, 0, 640, 640), (384, 0, 1024, 640),
                                   (0, 384, 640, 1024), (384, 384, 1024, 1024)])
        for x in (0, 512, 1023):
            for y in (0, 512, 1023):
                self.assertTrue(any(x1 <= x < x2 and y1 <= y < y2
                                    for x1, y1, x2, y2 in windows))

    def test_invalid_fraction(self):
        for value in (.49, 1.0, 1.2):
            with self.assertRaises(ValueError):
                crop_tta.tile_windows(100, 100, value)

    def test_map_crop_boxes_to_original(self):
        boxes = np.array([[0., 0., 1024., 1024.]])
        mapped = crop_tta.map_crop_boxes(boxes, (192, 384, 832, 1024),
                                         (1024, 1024), (1024, 1024), (1300, 1300))
        np.testing.assert_allclose(mapped, [[192. * 1300 / 1024, 384. * 1300 / 1024,
                                             832. * 1300 / 1024, 1024. * 1300 / 1024]])

    def test_subset_is_unique_and_even(self):
        indices = crop_tta.subset_indices(814, 200)
        self.assertEqual(len(indices), 200)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 813)
        self.assertEqual(len(set(indices)), len(indices))


if __name__ == "__main__":
    unittest.main()
