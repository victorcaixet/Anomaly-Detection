import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from insulator_anomaly import seg_models as sm
from insulator_anomaly.seg_annotate import read_mask
from insulator_anomaly.seg_data import finalize, mask_to_yolo_polygons
from insulator_anomaly.seg_models import evaluate, load_predictor, train_torch


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root, count=10, size=(48, 64)):
    """Imagens com um retângulo claro e máscara correspondente; 2 imagens por rajada."""
    rng = np.random.default_rng(0)
    review, pool = [], []
    for i in range(count):
        image = rng.integers(0, 60, (*size, 3), dtype=np.uint8)
        mask = np.zeros(size, np.uint8)
        y, x = rng.integers(4, 12), rng.integers(4, 16)
        mask[y:y + 24, x:x + 32] = 255
        image[mask > 0] = 200
        path = f"train/good/img{i}.jpg"
        (root / "data/train/good").mkdir(parents=True, exist_ok=True)
        (root / "run").mkdir(exist_ok=True)
        cv2.imwrite(str(root / "data" / path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        mask_path = root / "run" / f"mask{i}.png"
        cv2.imwrite(str(mask_path), mask)
        review.append(dict(path=path, mask_path=str(mask_path), status="aceita",
                           image_sha256=hashlib.sha256((root / "data" / path).read_bytes()).hexdigest(),
                           mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()))
        pool.append(dict(path=path, burst_id=f"b{i // 2}"))
    write_csv(root / "run/review.csv", review)
    write_csv(root / "run/pool.csv", pool)
    return review


class SegModelsTests(unittest.TestCase):
    def test_polygons_are_normalized_and_skip_specks(self):
        mask = np.zeros((20, 40), bool)
        mask[5:15, 10:30] = True
        mask[0, 0] = True
        lines = mask_to_yolo_polygons(mask)
        self.assertEqual(len(lines), 1)
        values = [float(v) for v in lines[0].split()[1:]]
        self.assertTrue(all(0 <= v <= 1 for v in values))
        self.assertEqual(lines[0].split()[0], "0")
        self.assertGreaterEqual(len(values), 8)

    def test_finalize_splits_by_burst_and_blocks_unreviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = make_dataset(root)
            rows = finalize(root / "run/review.csv", root / "run/pool.csv", root / "data", root / "final")
            by_split = {s: {r["burst_id"] for r in rows if r["split"] == s} for s in ("train", "val")}
            self.assertFalse(by_split["train"] & by_split["val"])
            self.assertEqual(len(rows), 10)
            self.assertEqual(len(list((root / "final/val/images").iterdir())), 2)
            label = (root / "final/train/labels/img0.txt")
            self.assertTrue(label.read_text().startswith("0 ") or label.read_text() == "")
            self.assertTrue(read_mask(root / "final/train/masks/img0.png").any()
                            or read_mask(root / "final/val/masks/img0.png").any())
            self.assertIn("val: val/images", (root / "final/data.yaml").read_text())
            pending = [dict(r, status="pendente" if i == 0 else r["status"]) for i, r in enumerate(review)]
            write_csv(root / "run/pending.csv", pending)
            with self.assertRaisesRegex(ValueError, "sem revisão"):
                finalize(root / "run/pending.csv", root / "run/pool.csv", root / "data", root / "f2")
            changed = [dict(r, image_sha256="0" * 64) if i == 3 else r for i, r in enumerate(review)]
            write_csv(root / "run/changed.csv", changed)
            with self.assertRaisesRegex(ValueError, "alterada"):
                finalize(root / "run/changed.csv", root / "run/pool.csv", root / "data", root / "f3")

    def test_train_predict_evaluate_unet_and_segformer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root)
            finalize(root / "run/review.csv", root / "run/pool.csv", root / "data", root / "final")
            for kind in ("unet", "segformer"):
                weights = train_torch(kind, root / "final", root / "out", epochs=2, batch=4, size=64,
                                      device="cpu", pretrained=False, segformer=None)
                self.assertTrue(weights.exists())
                predictor = load_predictor(kind, weights, "cpu")
                image = cv2.imread(str(next((root / "final/val/images").iterdir())))
                pred = predictor(image)
                self.assertEqual(pred.shape, image.shape[:2])
                self.assertEqual(pred.dtype, bool)
                rows, summary = evaluate(predictor, root / "final/val", root / f"{kind}.csv")
                self.assertEqual(summary["n"], 2)
                self.assertTrue(0 <= summary["iou"] <= 1 and 0 <= summary["fail_rate"] <= 1)
                self.assertEqual(len(rows), 2)
            with self.assertRaises(FileExistsError):
                train_torch("unet", root / "final", root / "out", epochs=1, size=64, device="cpu",
                            pretrained=False)

    def test_train_models_runs_in_sequence_isolates_failure_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root)
            finalize(root / "run/review.csv", root / "run/pool.csv", root / "data", root / "final")
            real, calls = sm.train_torch, []

            def fast(kind, data_dir, output, epochs, device, seed, encoder, segformer):
                calls.append(kind)
                return real(kind, data_dir, output, epochs=1, batch=4, size=64, device="cpu", seed=seed,
                            pretrained=False, encoder=encoder, segformer=None)

            def broken(*args, **kwargs):
                raise RuntimeError("falha simulada")

            kinds = ["yolo", "unet", "segformer"]
            with patch.object(sm, "train_torch", fast), patch.object(sm, "train_yolo", broken):
                results, errors = sm.train_models(kinds, root / "final", root / "out")
                self.assertEqual([r["model"] for r in results], ["unet", "segformer"])
                self.assertEqual(list(errors), ["yolo"])
                self.assertEqual(calls, ["unet", "segformer"])
                self.assertIn("falha simulada", (root / "out/yolo_erro.txt").read_text(encoding="utf-8"))
                with open(root / "out/comparison.csv", encoding="utf-8") as file:
                    self.assertEqual([r["model"] for r in csv.DictReader(file)], ["unet", "segformer"])
                sm.train_models(kinds, root / "final", root / "out")  # retomada
                self.assertEqual(calls, ["unet", "segformer"])
                (root / "out/yolo").mkdir()
                with self.assertRaises(FileExistsError):
                    sm.train_models(["yolo"], root / "final", root / "out")
                _, errors = sm.train_models(["yolo"], root / "final", root / "out", retry=True)
                self.assertEqual(list(errors), ["yolo"])
                self.assertFalse((root / "out/yolo").exists())


if __name__ == "__main__":
    unittest.main()
