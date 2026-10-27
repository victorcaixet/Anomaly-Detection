import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from insulator_anomaly.session_experiment import (
    build_manifest, cluster_dates, evaluate, prepare_groups,
)


def make_source(root):
    rng = np.random.default_rng(0)
    for mode in ("original", "gray"):
        for date, level in (("01-12-2020", 60), ("02-12-2020", 200)):
            for dji in [10 * b * 10 + i for b in range(12) for i in range(4)]:
                folder = root / mode / "train/good"
                folder.mkdir(parents=True, exist_ok=True)
                image = np.clip(rng.normal(level, 10, (32, 32, 3)), 0, 255).astype(np.uint8)
                cv2.imwrite(str(folder / f"train__good__Fotos_{date}_DJI_{dji:04d}_x.png"), image)
            for dji in (1050, 1051):
                folder = root / mode / "test/anomaly"
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"test__anomaly__Fotos_{date}_DJI_{dji:04d}_x.png"),
                            np.full((32, 32, 3), level, np.uint8))
        (root / mode / "test/good").mkdir(parents=True, exist_ok=True)


class SessionExperimentTests(unittest.TestCase):
    def test_bursts_never_cross_splits_and_split_is_grouping_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            make_source(Path(directory))
            manifest = build_manifest(directory, "gray", val=0.3, test=0.3)
            self.assertEqual(manifest[manifest.y_true == 0].groupby("burst").split.nunique().max(), 1)
            self.assertTrue((manifest[manifest.y_true == 1].split == "test").all())
            self.assertEqual(set(manifest.split), {"train", "val", "test"})
            pd.testing.assert_frame_equal(manifest, build_manifest(directory, "gray", val=0.3, test=0.3))

    def test_clusters_separate_dates_merge_small_and_prepare_links(self):
        with tempfile.TemporaryDirectory() as directory:
            make_source(Path(directory))
            manifest = build_manifest(directory, "gray", val=0.25, test=0.25)
            groups = cluster_dates(manifest, k=2, min_good=1)
            self.assertNotEqual(groups["01-12-2020"], groups["02-12-2020"])
            merged = cluster_dates(manifest, k=2, min_good=10_000)
            self.assertEqual(len(set(merged.values())), 1)
            prepare_groups(manifest, groups, Path(directory) / "exp")
            n = sum(1 for _ in (Path(directory) / "exp").rglob("*.png"))
            self.assertEqual(n, len(manifest))
            with self.assertRaises(FileExistsError):
                prepare_groups(manifest, groups, Path(directory) / "exp")

    def test_evaluate_uses_validation_threshold_per_model(self):
        rows = []
        for group, scale in (("a", 1.0), ("b", 10.0)):
            for split, label, base in (("val", 0, 0.5), ("test", 0, 0.5), ("test", 1, 2.0)):
                for i in range(20):
                    rows.append(dict(model_group=group, split=split, y_true=label, date=group,
                                     max=scale * (base + i * 0.01), top=scale * (base + i * 0.01)))
        table = evaluate(pd.DataFrame(rows), {"a": "g0", "b": "g0"})
        row = table[(table.score == "max") & (table.group == "ALL")].iloc[0]
        self.assertEqual(row.AUROC, 1.0)
        self.assertEqual(row.recall, 1.0)
        self.assertLessEqual(row.FPR, 0.1)


if __name__ == "__main__":
    unittest.main()
