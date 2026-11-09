import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from insulator_anomaly.crop_experiment import CropSettings, prepare_crop_experiment
from insulator_anomaly.preprocessing_study import MODES, render_variants, paired_effects


class PreprocessingStudyTests(unittest.TestCase):
    def test_background_controls_keep_object_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            image = np.full((32, 32, 3), 30, np.uint8)
            image[8:24, 8:24] = 220
            mask = np.zeros((32, 32), np.uint8)
            mask[8:24, 8:24] = 255
            cv2.imwrite(str(path), mask)
            outputs, info = render_variants(image, path, CropSettings(output_size=32, margin=.5))
            for mode in MODES:
                np.testing.assert_array_equal(outputs[mode][8:24, 8:24], image[8:24, 8:24])
            self.assertTrue((outputs["gray"][0, 0] == 127).all())
            self.assertTrue((outputs["whole"][0, 0] == 30).all())
            self.assertTrue((outputs["context"][7, 16] == 30).all())
            self.assertTrue((outputs["gray"][7, 16] == 127).all())
            self.assertEqual({tuple(info[m][k] for k in ("x0", "y0", "x1", "y1")) for m in MODES}, {(0, 0, 32, 32)})
            cv2.imwrite(str(path), np.zeros_like(mask))
            outputs, info = render_variants(image, path, CropSettings(output_size=32))
            for mode in MODES:
                np.testing.assert_array_equal(outputs[mode], image)
                self.assertTrue(info[mode]["segmentation_failed"])
            cv2.imwrite(str(path), np.zeros((8, 8), np.uint8))
            with self.assertRaisesRegex(ValueError, "geometria"):
                render_variants(image, path, CropSettings())

    def test_study_retains_failed_sources_and_pairs_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_index = {}
            for folder, values in (("train/good", [10, 20, 30]), ("test/good", [40]), ("test/anomaly", [50])):
                target = root / "source" / folder
                target.mkdir(parents=True)
                for value in values:
                    relative = f"{folder}/{value}.png"
                    cv2.imwrite(str(root / "source" / relative), np.full((32, 32, 3), value, np.uint8))
                    path = root / f"mask{value}.png"
                    mask = np.zeros((32, 32), np.uint8)
                    if value != 50:
                        mask[8:24, 8:24] = 255
                    cv2.imwrite(str(path), mask)
                    mask_index[relative] = str(path)
            frame = prepare_crop_experiment(root / "source", root / "run", mask_index=mask_index)
            self.assertEqual(set(frame["mode"]), set(MODES))
            self.assertTrue((frame.groupby("source_id").size() == len(MODES)).all())
            self.assertTrue((frame.groupby("source_id").split.nunique() == 1).all())
            self.assertTrue(frame[frame.y_true == 1].segmentation_failed.all())

    def test_paired_bootstrap_identical_and_coverage_guard(self):
        rows = []
        for mode in ("whole", "gray"):
            for i in range(20):
                rows.append(dict(mode=mode, split="test", source_id=f"t{i}", group_id=f"g{i//2}",
                                 y_true=i % 2, pred_score=float(i % 2)))
            for i in range(5):
                rows.append(dict(mode=mode, split="val", source_id=f"v{i}", group_id=f"v{i}",
                                 y_true=0, pred_score=.1))
        scores = pd.DataFrame(rows)
        result = paired_effects(scores, n_boot=100)
        np.testing.assert_array_equal(result[["delta", "ci95_low", "ci95_high"]], 0)
        with self.assertRaisesRegex(ValueError, "mesmas origens"):
            paired_effects(scores.iloc[:-1], n_boot=100)


if __name__ == "__main__":
    unittest.main()
