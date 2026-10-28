import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from insulator_anomaly.figures import load_results, make_report, select_cases


class ReportTests(unittest.TestCase):
    def test_report_writes_all_figures_and_pdf(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(40):
                truth = int(i % 4 == 0)
                score = float(rng.uniform(0.4, 1) if truth else rng.uniform(0, 0.7))
                path = root / f"Fotos_0{1 + i % 3}-12-2020_DJI_{i:04d}.png"
                Image.new("RGB", (32, 32)).save(path)
                rows.append(dict(image_path=str(path), y_true=truth, pred_score=score, pred_label=int(score > 0.5)))
            frame = pd.DataFrame(rows)
            metrics = dict(auroc=0.8, average_precision=0.5,
                           checkpoint=dict(tn=20, fp=10, fn=3, tp=7, recall=0.7, false_positive_rate=0.33))
            curve = pd.DataFrame(dict(threshold=[0.1, 0.5], precision=[0.3, 0.6], recall=[1.0, 0.5]))
            for mode in ("original", "gray"):
                base = root / "results" / mode / "precision_recall_test"
                base.mkdir(parents=True)
                frame.to_csv(base / "predictions.csv", index=False)
                curve.to_csv(base / "precision_recall_curve.csv", index=False)
                (base / "metrics.json").write_text(json.dumps(metrics))
            results = load_results(root / "results", ("original", "gray"))
            cases = select_cases(frame, per_group=3)
            self.assertLessEqual(cases.groupby("group").size().max(), 3)
            maps = {p: rng.random((32, 32)) for p in cases.image_path}
            sessions = pd.DataFrame(dict(sistema=["global", "por_grupo"] * 2, score="max",
                                         group=["ALL", "ALL", "g0", "g0"], AUROC=0.8, AP=0.5))
            files = make_report(results, root / "out", "gray", cases, maps, sessions)
            names = {f.name for f in files}
            self.assertIn("resultados.pdf", names)
            self.assertEqual(len([n for n in names if n.endswith(".png")]), 7)


if __name__ == "__main__":
    unittest.main()
