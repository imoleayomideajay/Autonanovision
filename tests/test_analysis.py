import unittest

import numpy as np

from autonanovision.analysis import connected_components, label_overlay, threshold_mask, to_grayscale


class AnalysisTests(unittest.TestCase):
    def test_to_grayscale_preserves_shape(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        gray = to_grayscale(rgb)
        self.assertEqual(gray.shape, (3, 4))

    def test_threshold_mask(self):
        gray = np.array([[0, 10], [20, 30]], dtype=np.float32)
        mask = threshold_mask(gray, percentile=50)
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[1, 1])

    def test_connected_components_stats(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:6, 3:7] = True  # 4x4 square
        labels, components = connected_components(mask, min_pixels=1)

        self.assertEqual(len(components), 1)
        comp = components[0]
        self.assertEqual(int(comp["area_px"]), 16)
        self.assertEqual(int(comp["bbox_width_px"]), 4)
        self.assertEqual(int(comp["bbox_height_px"]), 4)
        self.assertGreater(comp["circularity"], 0)
        self.assertEqual(int(labels.max()), 1)

    def test_label_overlay_marks_border(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 2:5] = 1
        overlaid = label_overlay(image, labels, 1)
        self.assertTrue(np.any((overlaid[:, :, 0] == 255) & (overlaid[:, :, 1] == 0) & (overlaid[:, :, 2] == 0)))


if __name__ == "__main__":
    unittest.main()
