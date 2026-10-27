"""Regressões do fluxo modular: imagens reais pequenas e inferência simulada."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
from PIL import Image

from insulator_anomaly.config import ProjectConfig
from insulator_anomaly.data import safe_output_name
from insulator_anomaly.evaluation import evaluate_all
from insulator_anomaly.inspection import analyze_false_negatives
from insulator_anomaly.paths import extract_project, resolve_project
from insulator_anomaly.preprocessing import (
    prepare_processed_datasets,
    preprocess_image_and_mask,
    principal_axis_angle,
)
from insulator_anomaly.training import train_one_patchcore


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

    def __len__(self):
        return len(self.value)


def make_images(root):
    for split in ("train/good", "test/good", "test/anomaly"):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(120, 130, 140)).save(folder / "sample.png")


class PipelineTests(unittest.TestCase):
    def test_local_gpu_configuration_and_notebook_guard(self):
        config = ProjectConfig(".", accelerator="gpu")
        self.assertEqual(config.yolo_device, 0)
        self.assertEqual(config.num_workers, 0)
        self.assertEqual(ProjectConfig(".", accelerator="cpu").yolo_device, "cpu")
        self.assertIsNone(ProjectConfig(".").yolo_device)
        for name in ("PatchCore_Anomalib", "Crop_Comparison", "Heatmap_Comparison"):
            notebook = json.loads(Path(name + ".ipynb").read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), name, "exec")
            setup = next("".join(c["source"]) for c in notebook["cells"]
                         if 'ACCELERATOR = "gpu"' in "".join(c["source"]))
            setup = setup[setup.index("import torch"):]
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(RuntimeError, "CUDA indisponivel"):
                    exec(setup, {})

    def test_paths_support_processed_only_and_reject_archive_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "anomalib_glass" / "glass-preprocessed").mkdir(parents=True)
            project, persistent = resolve_project(project_path=root)
            self.assertEqual(project, (root / "anomalib_glass").resolve())
            self.assertEqual(persistent, project)
            self.assertFalse(ProjectConfig(project).dataset_root.exists())
            with self.assertRaises(FileNotFoundError):
                resolve_project(project_path=root / "missing")
            with self.assertRaises(ValueError):
                resolve_project(project_path=root, zip_path=root / "x.zip")
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "bad")
            with self.assertRaises(ValueError):
                extract_project(archive, root / "safe")
            self.assertFalse((root / "outside.txt").exists())
            with zipfile.ZipFile(root / "valid.zip", "w") as bundle:
                bundle.writestr(
                    "anomalib_glass/glass-preprocessed/gray/test/good/a.png", b"fixture"
                )
            with contextlib.redirect_stdout(io.StringIO()):
                extracted, _ = resolve_project(zip_path=root / "valid.zip")
            self.assertEqual(extracted, project)

    def test_alignment_for_both_diagonals_and_three_backgrounds(self):
        config = ProjectConfig(Path.cwd())
        for sign in (-1, 1):
            mask = np.zeros((80, 80), np.uint8)
            cv2.line(
                mask, (12, 15 if sign == 1 else 65), (65, 65 if sign == 1 else 15), 1, 8
            )
            image = np.zeros((80, 80, 3), np.uint8)
            image[mask > 0] = (220, 120, 50)
            outputs, debug = preprocess_image_and_mask(config, image, mask.astype(bool))
            self.assertEqual(set(outputs), {"original", "blur", "gray"})
            self.assertLess(abs(principal_axis_angle(debug["object_mask"] > 0)), 2)
            for output in outputs.values():
                self.assertEqual(output.shape, (256, 256, 3))
                self.assertEqual(output.dtype, np.uint8)

    def test_preprocessing_is_explicit_and_never_replaces_existing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ProjectConfig(Path(temporary))
            make_images(config.dataset_root)
            config.yolo_model_path.write_bytes(b"fixture")
            with self.assertRaises(FileNotFoundError):
                prepare_processed_datasets(config)
            self.assertFalse(config.processed_root.exists())

            class YOLO:
                def __init__(self, *args):
                    pass

                def predict(self, source, **kwargs):
                    mask = np.zeros((1, 64, 64), np.float32)
                    mask[0, 20:45, 10:55] = 1
                    return [
                        types.SimpleNamespace(
                            masks=types.SimpleNamespace(data=Tensor(mask))
                        )
                    ]

            with (
                patch.dict(
                    sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=YOLO)}
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = prepare_processed_datasets(config, generate=True)
            self.assertFalse(summary["errors"])
            for mode in config.background_modes:
                for split in ("train/good", "test/good", "test/anomaly"):
                    self.assertEqual(
                        len(list((config.processed_root / mode / split).glob("*.png"))),
                        1,
                    )
            before = {
                str(p): p.read_bytes() for p in config.processed_root.rglob("*.png")
            }
            self.assertEqual(prepare_processed_datasets(config)["cache"], "existing")
            with self.assertRaises(FileExistsError):
                prepare_processed_datasets(config, generate=True)
            self.assertEqual(
                before,
                {str(p): p.read_bytes() for p in config.processed_root.rglob("*.png")},
            )
            self.assertNotEqual(
                safe_output_name(Path("train/good/a.jpg"), Path(".")),
                safe_output_name(Path("test/good/a.jpg"), Path(".")),
            )

    def test_training_passes_split_settings_reuses_and_preserves_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ProjectConfig(Path(temporary), background_modes=("gray",))
            make_images(config.dataset_root)
            make_images(config.processed_root / "gray")
            config.yolo_model_path.write_bytes(b"fixture")
            calls = []

            class Model:
                def __init__(self, **kwargs):
                    pass

                @staticmethod
                def configure_pre_processor(**kwargs):
                    return None

            class Engine:
                def __init__(self, **kwargs):
                    self.trainer = types.SimpleNamespace(
                        save_checkpoint=lambda path, **kwargs: Path(path).write_bytes(
                            b"new-checkpoint"
                        )
                    )

                def fit(self, **kwargs):
                    calls.append("fit")

            fake_modules = {
                "anomalib.data": types.SimpleNamespace(
                    Folder=lambda **kwargs: calls.append(kwargs)
                ),
                "anomalib.models": types.SimpleNamespace(Patchcore=Model),
                "anomalib.engine": types.SimpleNamespace(Engine=Engine),
                "torch": types.SimpleNamespace(
                    Tensor=Tensor,
                    cuda=types.SimpleNamespace(is_available=lambda: False),
                ),
            }
            with (
                patch.dict(sys.modules, fake_modules),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = train_one_patchcore(config, "gray")
                reused = train_one_patchcore(config, "gray")
            self.assertEqual(calls[0]["test_split_mode"], "from_dir")
            self.assertEqual(calls[0]["val_split_mode"], "none")
            self.assertEqual(calls.count("fit"), 1)
            self.assertTrue(reused["resumed_from_checkpoint"])
            checkpoint = Path(report["checkpoint"])
            self.assertEqual(checkpoint.read_bytes(), b"new-checkpoint")
            self.assertFalse((config.results_root / "gray" / "last.ckpt").exists())
            config.output_size = 128
            with self.assertRaises(FileExistsError):
                train_one_patchcore(config, "gray")
            with (
                patch.dict(sys.modules, fake_modules),
                patch.object(Engine, "fit", side_effect=RuntimeError("failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    train_one_patchcore(config, "gray", force=True)
            self.assertEqual(checkpoint.read_bytes(), b"new-checkpoint")

    def test_evaluation_exact_coverage_metrics_figures_and_persistence(self):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as temporary:
            config = ProjectConfig(
                Path(temporary),
                persistent_root=Path(temporary) / "drive",
                background_modes=("gray",),
            )
            batches = []
            for name, truth, score, label in [
                ("fp", 0, 0.8, 1),
                ("tn", 0, 0.1, 0),
                ("fn", 1, 0.4, 0),
            ]:
                path = (
                    config.processed_root
                    / "gray"
                    / "test"
                    / ("good" if truth == 0 else "anomaly")
                    / f"{name}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32)).save(path)
                batches.append(
                    types.SimpleNamespace(
                        image_path=[str(path)],
                        pred_score=Tensor([score]),
                        pred_label=Tensor([label]),
                        anomaly_map=Tensor(np.full((1, 1, 32, 32), score)),
                    )
                )
            checkpoint = config.checkpoints_root / "gray" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"old-checkpoint")

            class Model:
                @staticmethod
                def load_from_checkpoint(*args, **kwargs):
                    return Model()

            class Engine:
                def __init__(self, **kwargs):
                    pass

                def predict(self, **kwargs):
                    if "data_path" in kwargs:
                        return [
                            batch
                            for batch in batches
                            if batch.image_path == [str(kwargs["data_path"])]
                        ]
                    return list(batches)

            fake_modules = {
                "anomalib.data": types.SimpleNamespace(
                    PredictDataset=lambda **kwargs: types.SimpleNamespace(
                        collate_fn=None
                    )
                ),
                "anomalib.models": types.SimpleNamespace(Patchcore=Model),
                "anomalib.engine": types.SimpleNamespace(Engine=Engine),
                "torch": types.SimpleNamespace(
                    Tensor=Tensor,
                    cuda=types.SimpleNamespace(is_available=lambda: False),
                ),
                "torch.utils.data": types.SimpleNamespace(
                    DataLoader=lambda *args, **kwargs: None
                ),
            }
            with (
                patch.dict(sys.modules, fake_modules),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evaluations, comparison = evaluate_all(config, show=False)
                fn_analysis = analyze_false_negatives(config, evaluations, show_first=0)
            self.assertEqual(len(fn_analysis), 1)
            self.assertTrue(
                (config.persistent_results_root / "gray/fn_four_views").is_dir()
            )
            frame, metrics, _ = evaluations["gray"]
            self.assertEqual(len(frame), 3)
            self.assertEqual(metrics["checkpoint"]["fp"], 1)
            self.assertEqual(metrics["checkpoint"]["fn"], 1)
            self.assertEqual(metrics["recall_exploratorio"]["recall"], 1)
            self.assertAlmostEqual(metrics["average_precision"], 0.5)
            self.assertAlmostEqual(metrics["auroc"], 0.5)
            self.assertEqual(comparison.iloc[0].n_anomaly, 1)
            self.assertTrue(
                (
                    config.persistent_results_root
                    / "gray/precision_recall_test/precision_recall_curve.png"
                ).is_file()
            )
            self.assertEqual(checkpoint.read_bytes(), b"old-checkpoint")
            batches.append(batches[0])
            with (
                patch.dict(sys.modules, fake_modules),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "cobrem exatamente"),
            ):
                evaluate_all(config, show=False)

    def test_colab_bundle_contains_only_importable_code(self):
        with zipfile.ZipFile("insulator_anomaly.zip") as bundle:
            self.assertTrue(bundle.namelist())
            self.assertTrue(
                all(
                    name.startswith("insulator_anomaly/") and name.endswith(".py")
                    for name in bundle.namelist()
                )
            )
            for name in bundle.namelist():
                self.assertEqual(bundle.read(name), Path(name).read_bytes())
            with tempfile.TemporaryDirectory() as temporary:
                bundle.extractall(temporary)
                for path in Path(temporary).rglob("*.py"):
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_notebook_bootstraps_from_code_zip_outside_repository(self):
        notebook = json.loads(
            Path("PatchCore_Anomalib.ipynb").read_text(encoding="utf-8")
        )
        bootstrap = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and 'CODE_ZIP_PATH = ""' in "".join(cell["source"])
        )
        bootstrap = bootstrap.replace(
            'CODE_ZIP_PATH = ""',
            f"CODE_ZIP_PATH = {str(Path('insulator_anomaly.zip').resolve())!r}",
        )
        with tempfile.TemporaryDirectory() as temporary:
            check = (
                bootstrap
                + "\nfrom insulator_anomaly.config import ProjectConfig\nassert ProjectConfig('.').project_root == Path.cwd()\n"
            )
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", check],
                cwd=temporary,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                (
                    Path(temporary) / ".runtime_code/insulator_anomaly/config.py"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
