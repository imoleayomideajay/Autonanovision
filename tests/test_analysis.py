import unittest

import numpy as np

from autonanovision.analysis import (
    components_csv,
    connected_components,
    enrich_components,
    label_overlay,
    otsu_threshold,
    threshold_mask,
    to_grayscale,
)


class AnalysisTests(unittest.TestCase):
    def test_to_grayscale_preserves_shape(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        gray = to_grayscale(rgb)
        self.assertEqual(gray.shape, (3, 4))

    def test_threshold_mask_percentile(self):
        gray = np.array([[0, 10], [20, 30]], dtype=np.float32)
        mask = threshold_mask(gray, percentile=50, method="percentile")
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[1, 1])

    def test_otsu_threshold_returns_valid_range(self):
        gray = np.zeros((10, 10), dtype=np.float32)
        gray[:, 5:] = 255
        threshold = otsu_threshold(gray)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 255)

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

    def test_enrich_components_with_calibration(self):
        rows = enrich_components(
            [
                {
                    "label": 1.0,
                    "area_px": 100.0,
                    "perimeter_px": 20.0,
                    "bbox_width_px": 10.0,
                    "bbox_height_px": 5.0,
                    "aspect_ratio": 2.0,
                    "circularity": 0.8,
                }
            ],
            microns_per_pixel=0.5,
        )
        self.assertAlmostEqual(rows[0]["area_um2"], 25.0)
        self.assertAlmostEqual(rows[0]["perimeter_um"], 10.0)


    def test_enrich_components_invalid_calibration_does_not_crash(self):
        rows = enrich_components(
            [{"label": 1.0, "area_px": 9.0, "perimeter_px": 12.0, "bbox_width_px": 3.0, "bbox_height_px": 3.0}],
            microns_per_pixel="invalid",
        )
        self.assertEqual(rows[0]["area_px"], 9)
        self.assertNotIn("area_um2", rows[0])

    def test_csv_export(self):
        csv_text = components_csv([{"rank": 1, "label": 1, "area_px": 10}])
        self.assertIn("rank,label,area_px", csv_text)
        self.assertIn("1,1,10", csv_text)

    def test_label_overlay_marks_border(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 2:5] = 1
        overlaid = label_overlay(image, labels, 1)
        red = (overlaid[:, :, 0] == 255) & (overlaid[:, :, 1] == 0) & (overlaid[:, :, 2] == 0)
        self.assertTrue(np.any(red))


if __name__ == "__main__":
    unittest.main()
