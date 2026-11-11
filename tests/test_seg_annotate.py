import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from insulator_anomaly.seg_annotate import (
    auto_box, box_iou, check_report, generate_masks, mask_features, pick_audit, range_reasons,
    reference_ranges, stability, write_detection_dataset, xyxy_to_yolo,
)


class Tensor:
    """Imita o tensor do ultralytics: detach().cpu().numpy()."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __len__(self):
        return len(self.array)


class FakeSam:
    """Com caixa: caixa encolhida 10% por lado (fill 0,64). Só com ponto: metade central."""

    def predict(self, source, verbose, device, bboxes=None, points=None, labels=None):
        height, width = source.shape[:2]
        mask = np.zeros((height, width), np.float32)
        if bboxes is None:
            mask[round(height * .25):round(height * .75), round(width * .25):round(width * .75)] = 1
        else:
            x0, y0, x1, y1 = bboxes
            dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
            mask[round(y0 + dy):round(y1 - dy), round(x0 + dx):round(x1 - dx)] = 1
        masks = type("M", (), {"data": Tensor(mask[None])})
        return [type("R", (), {"masks": masks})]


class ConstSam(FakeSam):
    """Ignora o prompt: máscara perfeitamente estável."""

    def predict(self, source, verbose, device, bboxes=None, **_):
        return super().predict(source, verbose, device)


def make_images(root, names):
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / name), np.full((40, 80, 3), 90, np.uint8))


class SegAnnotateTests(unittest.TestCase):
    def test_box_helpers(self):
        self.assertEqual(xyxy_to_yolo((10, 10, 30, 20), 100, 50), (0.2, 0.3, 0.2, 0.2))
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_audit_is_deterministic_and_subset(self):
        paths = [f"p{i}" for i in range(40)]
        first = pick_audit(paths, 0.25)
        self.assertEqual(first, pick_audit(reversed(paths), 0.25))
        self.assertEqual(len(first), 10)
        self.assertTrue(first <= set(paths))

    def test_detection_dataset_splits_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"train/good/a{i}.jpg" for i in range(10)]
            make_images(root, names)
            boxes = {name: [8, 4, 40, 20] for name in names}
            yaml, val = write_detection_dataset(root, boxes, root / "out", val_fraction=0.2)
            self.assertEqual(len(val), 2)
            train = {p.stem for p in (root / "out/images/train").iterdir()}
            valid = {p.stem for p in (root / "out/images/val").iterdir()}
            self.assertEqual(len(train) + len(valid), 10)
            self.assertFalse(train & valid)
            label = next((root / "out/labels/val").iterdir()).read_text().split()
            self.assertEqual(label, ["0", "0.300000", "0.300000", "0.400000", "0.400000"])
            self.assertIn("images/val", yaml.read_text())
            make_images(root, ["train/good/small.jpg"])  # 80x40; caixa passa da borda
            write_detection_dataset(root, {"train/good/small.jpg": [-5, -5, 200, 100]}, root / "clamp")
            label = next((root / "clamp/labels").rglob("small.txt")).read_text().split()
            self.assertEqual(label, ["0", "0.500000", "0.500000", "1.000000", "1.000000"])
            with self.assertRaisesRegex(ValueError, "repetidos"):
                write_detection_dataset(root, {"a/x.jpg": [0, 0, 1, 1], "b/x.jpg": [0, 0, 1, 1]},
                                        root / "dup")

    def test_generate_masks_review_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"train/good/i{i}.jpg" for i in range(6)]
            make_images(root, names)
            boxes = {name: [10, 5, 70, 35] for name in names[:5]}  # a última fica sem caixa
            boxes[names[0]] = [10, 5, 12, 6]  # caixa minúscula: o arredondamento preenche tudo
            rows = [dict(path=n, subset="propagate", box_source="detector", box_conf="0.9")
                    for n in names]
            rows[1]["box_conf"] = "0.3"
            table = generate_masks(FakeSam(), root, rows, boxes, root / "out", audit=0.5, device="cpu",
                                   auto_point=None, stability_min=0.0)
            status = {e["path"]: e["status"] for e in table}
            self.assertEqual(status[names[5]], "sem_caixa")
            self.assertEqual(sum(s == "pendente" for s in status.values()), 5)
            self.assertTrue(table[1]["suspect"])
            self.assertIn("caixa incerta", table[1]["suspect_reasons"])
            self.assertIn("preenche a caixa", table[0]["suspect_reasons"])
            self.assertFalse(any(e["suspect"] and e["audit"] for e in table))
            self.assertEqual(sum(e["audit"] for e in table), round(3 * 0.5))
            self.assertTrue((root / "out/masks/train/good/i2.png").exists())
            flagged = json.loads((root / "out/suspeitas.json").read_text(encoding="utf-8"))
            self.assertEqual(set(flagged), {e["path"] for e in table if e["suspect"]})
            self.assertTrue(all(isinstance(v, list) and v for v in flagged.values()))
            with open(root / "out/review.csv", encoding="utf-8") as file:
                review = list(csv.DictReader(file))
            self.assertEqual(len(review), 6)
            self.assertEqual(review[5]["mask_path"], "")
            with self.assertRaises(FileExistsError):
                generate_masks(FakeSam(), root, rows, boxes, root / "out")

    def test_features_ranges_and_stability(self):
        mask = np.zeros((20, 40), bool)
        mask[2:18, 4:36] = True
        mask[8:12, 15:25] = False  # buraco interno
        features = mask_features(mask)
        self.assertEqual(features["n_components"], 1)
        self.assertAlmostEqual(features["holes_fraction"], 40 / (16 * 32 - 40))
        self.assertAlmostEqual(features["bbox_fill"], (16 * 32 - 40) / (16 * 32))
        self.assertAlmostEqual(features["log_aspect"], float(np.log(2.0)))
        self.assertIsNone(mask_features(np.zeros((5, 5), bool)))
        ranges = reference_ranges([dict(features, area_fraction=0.2 + i / 100) for i in range(20)])
        self.assertFalse(any("area_fraction" in r for r in range_reasons(dict(features, area_fraction=0.25), ranges)))
        self.assertTrue(any("area_fraction" in r for r in range_reasons(dict(features, area_fraction=0.9), ranges)))
        image = np.full((40, 80, 3), 90, np.uint8)
        box = [10, 5, 70, 35]
        const = ConstSam().predict(image, False, "cpu")[0].masks.data.numpy()[0] > 0.5
        shrunk = FakeSam().predict(image, False, "cpu", bboxes=box)[0].masks.data.numpy()[0] > 0.5
        self.assertEqual(stability(ConstSam(), image, box, const, device="cpu"), 1.0)
        self.assertLess(stability(FakeSam(), image, box, shrunk, device="cpu"), 1.0)
        self.assertEqual(auto_box(FakeSam(), image, "cpu"), [20.0, 10.0, 60.0, 30.0])

    def test_auto_box_fills_missing_boxes_and_flags_instability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"train/good/a{i}.jpg" for i in range(12)]
            make_images(root, names)
            boxes = {n: [10, 5, 70, 35] for n in names[:11]}
            rows = [dict(path=n, subset="propagate", box_source="manual" if n in boxes else "nenhuma",
                         box_conf="") for n in names]
            table = generate_masks(FakeSam(), root, rows, boxes, root / "out", device="cpu")
            last = table[-1]
            self.assertEqual((last["box_source"], last["status"]), ("sam_ponto", "pendente"))
            self.assertEqual(json.loads(last["box"]), [20.0, 10.0, 60.0, 30.0])
            self.assertTrue(last["suspect"])  # preenche a caixa e é instável
            self.assertIn("instável", table[0]["suspect_reasons"])
            self.assertTrue((root / "out/reference_ranges.json").exists())
            stable = generate_masks(ConstSam(), root, rows, boxes, root / "out2", device="cpu")
            self.assertFalse(any("instável" in e["suspect_reasons"] for e in stable))

    def test_check_report_rates(self):
        def row(status, suspect, audit="False", reasons=""):
            return dict(status=status, suspect=str(suspect), audit=audit, suspect_reasons=reasons)

        rows = [row("corrigida", True, reasons="fragmentada (0.5)"), row("aceita", True, reasons="fragmentada (0.7)"),
                row("refeita", True, reasons="instável sob variação da caixa (0.5)"),
                row("aceita", False, "True"), row("corrigida", False, "True"), row("aceita", False),
                row("pendente", False)]
        report = check_report(rows)
        self.assertEqual((report["revisadas"], report["suspeitas"]), (6, 3))
        self.assertAlmostEqual(report["ruins_entre_suspeitas"], 2 / 3)
        self.assertAlmostEqual(report["ruins_entre_nao_suspeitas"], 1 / 3)
        self.assertAlmostEqual(report["ruins_na_auditoria"], 0.5)
        self.assertEqual(report["por_motivo"]["fragmentada"], dict(n=2, ruins=0.5))


if __name__ == "__main__":
    unittest.main()
