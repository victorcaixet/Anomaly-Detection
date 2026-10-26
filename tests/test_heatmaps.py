"""Verificações numéricas e do fluxo de exportação, sem GPU ou checkpoints reais."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from insulator_anomaly import heatmaps as hm
from insulator_anomaly.config import ProjectConfig


class HeatmapTests(unittest.TestCase):
    def test_diagnostic_figure_renders_with_real_matplotlib(self):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, figure_path = root / "input.png", root / "diagnostic.png"
            Image.new("RGB", (32, 32), color=(120, 130, 140)).save(image_path)
            array = np.zeros((32, 32))
            array[10:20, 12:24] = 0.8
            features, hot, gradient = hm.heatmap_features(array, 0.5)
            hm.plot_heatmap_case(
                image_path,
                array,
                hot,
                gradient,
                np.zeros(1024),
                0.5,
                features,
                "Teste sintético",
                (0, 1),
                0.5,
                figure_path,
            )
            with Image.open(figure_path) as figure:
                self.assertEqual(figure.size, (2240, 1260))
            self.assertGreater(figure_path.stat().st_size, 10000)

    def test_gradient_distinguishes_level_from_transition(self):
        low, high = np.zeros((32, 32)), np.full((32, 32), 10.0)
        a, _, ga = hm.heatmap_features(low, 1)
        b, _, gb = hm.heatmap_features(high, 1)
        self.assertEqual(a["hot_area_fraction"], 0)
        self.assertEqual(b["hot_area_fraction"], 1)
        np.testing.assert_allclose(ga, 0)
        np.testing.assert_allclose(gb, 0, atol=1e-14)
        ramp = np.indices((32, 32))[1] * 2 + np.indices((32, 32))[0] * 3
        np.testing.assert_allclose(hm.heatmap_gradient(ramp, sigma=0), np.sqrt(13))

    def test_regions_and_threshold_sweep(self):
        array = np.zeros((32, 32))
        array[8:12, 8:12] = 4
        array[20:22, 20:22] = 2
        result, hot, _ = hm.heatmap_features(array, 1)
        self.assertEqual(result["components"], 2)
        self.assertAlmostEqual(result["hot_area_fraction"], 20 / 1024)
        self.assertAlmostEqual(result["largest_component_hot_share"], 16 / 20)
        self.assertEqual(result["hot_border_share"], 0)
        self.assertEqual(hm.component_areas(np.eye(3, dtype=bool)), [3])
        areas = [
            hm.heatmap_features(array, t)[0]["hot_area_fraction"] for t in [0, 1, 2, 4]
        ]
        self.assertEqual(areas, sorted(areas, reverse=True))
        self.assertEqual(hm.heatmap_features(array, 4)[0]["components"], 0)

    def test_roi_and_defect_overlap(self):
        array = np.zeros((32, 32))
        array[0, 0] = 99
        array[12:14, 12:14] = 5
        array[12, 12] = 6
        roi = np.zeros_like(array, bool)
        roi[6:26, 6:26] = True
        defect = np.zeros_like(roi)
        defect[12:14, 12:14] = True
        result, hot, _ = hm.heatmap_features(array, 4, roi=roi, defect=defect)
        self.assertEqual(result["map_peak"], 6)
        self.assertFalse(hot[0, 0])
        for metric in [
            "defect_iou",
            "defect_dice",
            "defect_recall",
            "defect_precision",
            "peak_on_defect",
        ]:
            self.assertEqual(result[metric], 1)
        defect[14, 14] = True
        result, _, _ = hm.heatmap_features(array, 4, roi=roi, defect=defect)
        self.assertAlmostEqual(result["defect_iou"], 4 / 5)
        self.assertAlmostEqual(result["defect_dice"], 8 / 9)
        constant, _, _ = hm.heatmap_features(np.ones_like(array), 2, defect=defect)
        self.assertTrue(np.isnan(constant["peak_on_defect"]))
        with self.assertRaises(ValueError):
            hm.heatmap_features(array, 1, roi=np.zeros_like(roi))
        with self.assertRaises(ValueError):
            hm.heatmap_features(array, 1, roi=roi, defect=np.ones_like(roi))
        with self.assertRaises(ValueError):
            hm.validate_heatmap([[float("nan"), 0], [0, 0]])

    def test_selection_is_reproducible_and_keeps_normal_false_positives(self):
        frame = pd.DataFrame(
            [
                dict(
                    image_path=f"{i}.png",
                    y_true=i % 2,
                    pred_label=(i // 2) % 2,
                    pred_score=i / 20,
                )
                for i in range(20)
            ]
        )
        first = hm.select_heatmap_cases(frame, 2, 32)
        second = hm.select_heatmap_cases(frame.sample(frac=1, random_state=10), 2, 32)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            set(first.loc[first.is_case, "group"]), {"FP", "FN", "TP", "TN"}
        )
        self.assertTrue((first.is_reference & (first.group == "FP")).any())
        self.assertFalse((first.is_reference & (first.y_true == 1)).any())
        self.assertTrue((first[first.is_case].group.value_counts() <= 2).all())

    def test_aligned_masks_reject_geometry_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "dataset" / "gray" / "test" / "good" / "a.jpg"
            destination = base / "masks" / "gray" / "test" / "good" / "a.png"
            destination.parent.mkdir(parents=True)
            Image.fromarray(np.ones((16, 16), np.uint8) * 255).save(destination)
            self.assertTrue(
                hm.read_aligned_mask(
                    base / "masks", "gray", path, base / "dataset", (16, 16)
                ).all()
            )
            with self.assertRaises(ValueError):
                hm.read_aligned_mask(
                    base / "masks", "gray", path, base / "dataset", (32, 32)
                )

    def test_inference_exports_and_persistence(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.value

            def reshape(self, *shape):
                return self.value.reshape(*shape)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset = base / "dataset"
            images, predictions = [], {}
            for name, truth, label, value in [
                ("tn", 0, 0, 0.1),
                ("fp", 0, 1, 0.6),
                ("fn", 1, 0, 0.4),
                ("tp", 1, 1, 0.8),
            ]:
                path = (
                    dataset
                    / "gray"
                    / "test"
                    / ("good" if truth == 0 else "anomaly")
                    / f"{name}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32)).save(path)
                array = np.zeros((1, 1, 32, 32), np.float32)
                array[..., 12:16, 12:16] = value
                predictions[str(path)] = types.SimpleNamespace(
                    pred_score=Tensor([value]),
                    pred_label=Tensor([label]),
                    anomaly_map=Tensor(array),
                )
                images.append(
                    dict(
                        image_path=str(path),
                        y_true=truth,
                        pred_label=label,
                        pred_score=value,
                    )
                )
            checkpoint = base / "last.ckpt"
            checkpoint.write_bytes(b"fixture")

            class Model:
                @classmethod
                def load_from_checkpoint(cls, *args, **kwargs):
                    return cls()

            class Engine:
                def __init__(self, **kwargs):
                    pass

                def predict(self, model, data_path, return_predictions):
                    return [predictions[str(data_path)]]

            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False)
            )
            fake_models = types.SimpleNamespace(Patchcore=Model)
            fake_engine = types.SimpleNamespace(Engine=Engine)
            fake_modules = {
                "torch": fake_torch,
                "anomalib.models": fake_models,
                "anomalib.engine": fake_engine,
            }

            def save_figure(*args, **kwargs):
                Path(args[-1]).write_bytes(b"figure-placeholder")

            evaluation = {
                "gray": (
                    pd.DataFrame(images),
                    {"checkpoint_path": str(checkpoint)},
                    None,
                )
            }
            config = ProjectConfig(base, persistent_root=base / "persist")
            dataset.rename(config.processed_root)
            for row in images:
                old = row["image_path"]
                row["image_path"] = old.replace(
                    str(dataset), str(config.processed_root)
                )
                predictions[row["image_path"]] = predictions.pop(old)
            evaluation["gray"] = (pd.DataFrame(images), evaluation["gray"][1], None)
            with (
                patch.dict(sys.modules, fake_modules),
                patch.object(hm, "plot_heatmap_case", side_effect=save_figure),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                table = hm.run_heatmap_analysis(
                    config, evaluation, ["gray"], reference_count=2
                )
            self.assertEqual(len(table), 4)
            self.assertEqual(set(table.group), {"FN", "FP", "TP", "TN"})
            output = config.persistent_results_root / "gray" / "heatmap_analysis"
            for filename in [
                "heatmap_metrics.csv",
                "threshold_sweep.csv",
                "group_summary.csv",
                "analysis_config.json",
            ]:
                self.assertTrue((output / filename).is_file(), filename)
            self.assertEqual(len(list(output.glob("*_map.npy"))), 4)
            self.assertEqual(len(list(output.glob("*_diagnostic.png"))), 4)
            sweep = pd.read_csv(output / "threshold_sweep.csv")
            for _, group in sweep.groupby("image_path"):
                self.assertTrue(
                    (
                        group.sort_values("quantile").hot_area_fraction.diff().dropna()
                        <= 0
                    ).all()
                )
            metadata = json.loads(
                (output / "analysis_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(metadata["reference_images"]), 2)
            self.assertEqual(metadata["reference_source"], "test/good: exploratorio")
            # Um checkpoint que não reproduz a avaliação deve interromper a análise.
            evaluation["gray"][0].loc[0, "pred_score"] = 9
            with (
                patch.dict(sys.modules, fake_modules),
                self.assertRaisesRegex(ValueError, "Predição mudou"),
            ):
                hm.run_heatmap_analysis(config, evaluation, ["gray"], reference_count=2)

    def test_notebook_is_portable_and_compiles(self):
        notebook = json.loads(
            Path("PatchCore_Anomalib.ipynb").read_text(encoding="utf-8")
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn(
            "from insulator_anomaly.heatmaps import run_heatmap_analysis", code
        )
        self.assertNotIn("def run_heatmap_analysis", code)


if __name__ == "__main__":
    unittest.main()
