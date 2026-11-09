"""Avaliação dos checkpoints: cobertura exata do teste e limiar exploratório separado."""

from pathlib import Path
import gc
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from .data import IMAGE_EXTENSIONS as EXTENSIONS


def val_threshold(normal_scores, fpr):
    """Limiar de alarme (score > limiar): quantil 1-fpr dos scores de validação normais."""
    return float(np.quantile(normal_scores, 1 - fpr, method="higher"))


def resolve_evaluation_checkpoint(config, mode):
    candidates = [
        config.results_root / mode / "last.ckpt",
        config.persistent_results_root / mode / "last.ckpt",
        config.checkpoints_root / mode / "last.ckpt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Checkpoint não encontrado: " + ", ".join(map(str, candidates))
    )


def predict_checked_heatmaps(engine, model, rows):
    """Uma inferência para todas as linhas; confere cada predição e geometria."""
    from anomalib.data import PredictDataset
    from torch.utils.data import DataLoader

    rows = list(rows)
    paths = [Path(row.image_path) for row in rows]
    dataset = PredictDataset(path=paths[0])
    dataset.image_filenames = paths
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batches = engine.predict(model=model, dataloaders=loader, return_predictions=True)
    if not batches or len(batches) != len(rows):
        raise ValueError(f"Esperava {len(rows)} predições, recebi {len(batches or [])}.")
    arrays = []
    for path, row, batch in zip(paths, rows, batches, strict=True):
        score = float(batch.pred_score.detach().cpu().reshape(-1)[0])
        label = int(batch.pred_label.detach().cpu().reshape(-1)[0])
        if not np.isclose(score, row.pred_score, rtol=1e-3, atol=1e-4) or label != int(
            row.pred_label
        ):
            raise ValueError(
                f"Predição mudou em {path}; execute novamente a avaliação."
            )
        array = batch.anomaly_map.detach().cpu().float().numpy().squeeze()
        if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
            raise ValueError(f"Mapa inválido: {path}")
        with Image.open(path) as image:
            if image.size != (array.shape[1], array.shape[0]):
                raise ValueError(f"Mapa e entrada com geometrias diferentes: {path}.")
        arrays.append(array)
    return arrays


def evaluate_saved_patchcore(config, mode, target_recall=0.99, show=True):
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        auc,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )

    if not 0 < target_recall <= 1:
        raise ValueError("target_recall deve estar entre 0 e 1.")
    test_root = config.processed_root / mode / "test"
    checkpoint = resolve_evaluation_checkpoint(config, mode)
    output = config.results_root / mode / "precision_recall_test"
    expected = {}
    for folder, label in (("good", 0), ("anomaly", 1)):
        images = sorted(
            (
                p
                for p in (test_root / folder).rglob("*")
                if p.suffix.lower() in EXTENSIONS
            )
        )
        if not images:
            raise ValueError(f"Sem imagens: {test_root / folder}")
        expected.update({str(p.resolve()): label for p in images})
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    model = Patchcore.load_from_checkpoint(
        str(checkpoint), map_location="cpu", weights_only=False
    )
    engine = Engine(
        accelerator=config.accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=output,
    )
    dataset = PredictDataset(path=test_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    rows = []
    try:
        predictions = engine.predict(
            model=model, dataloaders=loader, return_predictions=True
        )
        for batch in predictions:
            paths = batch.image_path
            if isinstance(paths, (str, Path)):
                paths = [paths]
            scores = batch.pred_score.detach().cpu().numpy().reshape(-1)
            labels = batch.pred_label.detach().cpu().numpy().reshape(-1)
            for image_path, score, label in zip(paths, scores, labels, strict=True):
                key = str(Path(image_path).resolve())
                if key not in expected:
                    raise ValueError(f"Imagem fora do conjunto de teste: {key}")
                rows.append(
                    dict(
                        image_path=key,
                        y_true=expected[key],
                        pred_score=float(score),
                        pred_label=int(label),
                    )
                )
        frame = pd.DataFrame(rows)
        if (
            frame.empty
            or frame.image_path.duplicated().any()
            or set(frame.image_path) != set(expected)
        ):
            raise ValueError("As predições não cobrem exatamente as imagens de teste.")
        if not np.isfinite(frame.pred_score).all():
            raise ValueError("Scores inválidos no teste.")
        truth, score = (frame.y_true, frame.pred_score)
        precision, recall, thresholds = precision_recall_curve(truth, score)
        curve = pd.DataFrame(
            {
                "threshold": thresholds,
                "precision": precision[:-1],
                "recall": recall[:-1],
            }
        )
        threshold = float(curve.loc[curve.recall >= target_recall, "threshold"].max())
        frame["pred_label_recall_exploratorio"] = (score >= threshold).astype(int)
        frame.to_csv(output / "predictions.csv", index=False)
        curve.to_csv(output / "precision_recall_curve.csv", index=False)
        metrics = dict(
            mode=mode,
            checkpoint_path=str(checkpoint),
            test_root=str(test_root),
            n_good=int((truth == 0).sum()),
            n_anomaly=int((truth == 1).sum()),
            average_precision=float(average_precision_score(truth, score)),
            pr_auc_trapezoidal=float(auc(recall, precision)),
            auroc=float(roc_auc_score(truth, score)),
            target_recall=target_recall,
            exploratory_threshold=threshold,
        )
        for name, column in (
            ("checkpoint", "pred_label"),
            ("recall_exploratorio", "pred_label_recall_exploratorio"),
        ):
            prediction = frame[column]
            tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
            metrics[name] = dict(
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                recall=float(tp / (tp + fn)),
                precision=float(tp / (tp + fp)) if tp + fp else 0.0,
                false_positive_rate=float(fp / (fp + tn)),
            )
            fps = frame[(truth == 0) & (prediction == 1)].sort_values(
                "pred_score", ascending=False
            )
            fns = frame[(truth == 1) & (prediction == 0)].sort_values("pred_score")
            fps.to_csv(output / f"false_positives_{name}.csv", index=False)
            fns.to_csv(output / f"false_negatives_{name}.csv", index=False)
            print(f"\n{mode} / {name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(
                classification_report(
                    truth,
                    prediction,
                    labels=[0, 1],
                    target_names=["normal", "anomalia"],
                    digits=4,
                    zero_division=0,
                )
            )
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.step(
            recall,
            precision,
            where="post",
            label=f"AP={metrics['average_precision']:.4f}",
        )
        ax.axhline(
            float(truth.mean()),
            color="gray",
            linestyle=":",
            label="Prevalência de anomalias",
        )
        ax.axvline(
            target_recall,
            color="red",
            linestyle="--",
            label=f"Meta recall={target_recall:.0%}",
        )
        ax.set(
            xlabel="Recall de anomalias",
            ylabel="Precisão de anomalias",
            title=f"PatchCore — {mode} — teste",
            xlim=(0, 1.01),
            ylim=(0, 1.01),
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "precision_recall_curve.png", dpi=160)
        if show:
            plt.show()
        plt.close(fig)
        return (frame, metrics, curve)
    finally:
        del model, engine
        if "predictions" in locals():
            del predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_all(config, modes=None, target_recall=0.99, show=True):
    evaluations = {}
    for mode in modes or config.background_modes:
        evaluations[mode] = evaluate_saved_patchcore(config, mode, target_recall, show)
        source = config.results_root / mode / "precision_recall_test"
        destination = config.persistent_results_root / mode / "precision_recall_test"
        if source.resolve() != destination.resolve():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
    comparison = pd.DataFrame(
        [
            {
                "mode": mode,
                "n_good": m["n_good"],
                "n_anomaly": m["n_anomaly"],
                "AP": m["average_precision"],
                "PR_AUC_trapezoidal": m["pr_auc_trapezoidal"],
                "recall_checkpoint": m["checkpoint"]["recall"],
                "FP_checkpoint": m["checkpoint"]["fp"],
                "FN_checkpoint": m["checkpoint"]["fn"],
                "limiar_exploratorio": m["exploratory_threshold"],
                "recall_exploratorio": m["recall_exploratorio"]["recall"],
                "FPR_exploratorio": m["recall_exploratorio"]["false_positive_rate"],
            }
            for mode, (_, m, _) in evaluations.items()
        ]
    )
    config.results_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        config.results_root / "precision_recall_comparison.csv", index=False
    )
    if config.results_root.resolve() != config.persistent_results_root.resolve():
        config.persistent_results_root.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            config.persistent_results_root / "precision_recall_comparison.csv",
            index=False,
        )
    return evaluations, comparison
