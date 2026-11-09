"""Ablation of background processing with paired inputs and frozen masks."""

from pathlib import Path
import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

from .data import build_dataset_records
from .preprocessing import read_image, letterbox, predict_object_mask, predict_object_mask_sam

MODES = ("full", "whole", "gray", "blur", "context")


def export_masks(config, destination):
    """Export once; empty masks record missing detections, never exclude samples.

    Runtime/infrastructure errors stop the run rather than masquerading as
    segmentation failures. Timings include readback, not model loading/warmup.
    """
    from ultralytics import YOLO, SAM

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use um cache novo: {destination}")
    if config.segmenter not in {"yolo", "sam"}:
        raise ValueError("Segmentador deve ser yolo ou sam.")
    records = build_dataset_records(config.dataset_root)
    model = SAM(config.sam_model) if config.segmenter == "sam" else YOLO(str(config.yolo_model_path))
    predictor = predict_object_mask_sam if config.segmenter == "sam" else predict_object_mask
    destination.mkdir(parents=True)
    rows = []
    for record in records:
        relative = record.image_path.relative_to(config.dataset_root).as_posix()
        image = read_image(record.image_path)
        start = perf_counter()
        status = "ok"
        try:
            mask = predictor(config, record.image_path, model, image)
        except RuntimeError as error:
            # Only known no-detection outcomes can use the full-image fallback.
            if not any(message in str(error) for message in (
                "nao encontrou objeto", "Nenhuma instancia das classes", "nao gerou mascara"
            )):
                raise
            mask = np.zeros(image.shape[:2], dtype=bool)
            status = "no_detection"
        elapsed = perf_counter() - start
        path = destination / (hashlib.sha256(relative.encode()).hexdigest() + ".png")
        ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise OSError(f"Falha ao salvar máscara: {path}")
        encoded.tofile(path)
        rows.append(dict(source_id=relative, mask_path=str(path), status=status,
                         source_sha256=hashlib.sha256(record.image_path.read_bytes()).hexdigest(),
                         mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         segmentation_seconds=elapsed))
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "masks.csv", index=False)
    (destination / "config.json").write_text(
        json.dumps(vars(config), default=str, indent=2), encoding="utf-8")
    return frame


def render_variants(image, mask_path, settings):
    """Fixed ROI/scale for all background variants; no alignment or mask cleanup."""
    from .crop_experiment import padded_box

    mask_path = Path(mask_path)
    mask_bytes = mask_path.read_bytes()
    mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError(f"Máscara deve ter a geometria original: {mask_path}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Máscara deve ser binária: {mask_path}")
    mask = mask > 0
    h, w = mask.shape
    fallback = not mask.any()
    if fallback:
        box = (0, 0, w, h)
    else:
        y, x = np.nonzero(mask)
        box = padded_box((x.min(), y.min(), x.max()+1, y.max()+1), w, h, settings.margin)
    x0, y0, x1, y1 = box
    crop, roi = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    outputs, diagnostics = {}, {}
    for mode in MODES:
        start = perf_counter()
        rendered = image if mode == "full" else crop
        if not fallback and mode in {"gray", "blur", "context"}:
            keep = roi
            if mode == "context":
                radius = int(np.ceil(min(roi.shape) * settings.context_ratio))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
                keep = cv2.dilate(roi.astype(np.uint8), kernel).astype(bool)
            background = np.full_like(crop, 127)
            if mode == "blur":
                sigma = max(2., min(roi.shape) * settings.blur_sigma_ratio)
                background = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma)
            rendered = np.where(keep[..., None], crop, background)
        outputs[mode] = letterbox(rendered, settings.output_size, (114, 114, 114))
        a, b, c, d = (0, 0, w, h) if mode == "full" else box
        diagnostics[mode] = dict(
            x0=a, y0=b, x1=c, y1=d, box_source="full_image" if mode == "full" or fallback else "mask",
            mask_path=str(mask_path.resolve()), mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
            segmentation_failed=fallback, preprocessing_seconds=perf_counter()-start,
            original_width=w, original_height=h,
        )
    return outputs, diagnostics


def prepare_study(config, experiment_root, masks_csv, groups, settings=None):
    """Require explicit acquisition groups and a complete, unchanged mask cache."""
    from .crop_experiment import prepare_crop_experiment

    masks = pd.read_csv(masks_csv, dtype={"source_id": str})
    if masks.source_id.duplicated().any():
        raise ValueError("Cache tem origens duplicadas.")
    known = {r.image_path.relative_to(config.dataset_root).as_posix()
             for r in build_dataset_records(config.dataset_root)}
    if groups is None or set(groups) != known or any(not str(v).strip() for v in groups.values()):
        raise ValueError("Forneça um grupo de aquisição para cada imagem, sem omissões.")
    if set(masks.source_id) != known:
        raise ValueError("Cache não cobre exatamente o dataset.")
    for row in masks.itertuples():
        source = config.dataset_root / row.source_id
        if hashlib.sha256(source.read_bytes()).hexdigest() != row.source_sha256:
            raise ValueError(f"Origem alterada após gerar máscaras: {source}")
        if hashlib.sha256(Path(row.mask_path).read_bytes()).hexdigest() != row.mask_sha256:
            raise ValueError(f"Máscara alterada após gerar cache: {row.mask_path}")
    return prepare_crop_experiment(
        config.dataset_root, experiment_root, settings=settings, groups=groups,
        mask_index=dict(zip(masks.source_id, masks.mask_path)),
    )


def paired_effects(scores, baseline="whole", normal_fpr=0.05, n_boot=2000, seed=42):
    """Paired group-bootstrap deltas; intervals conditional on fitted models/calibration.

    Metrics are per source image. Groups are resampled together, including groups
    with mixed labels. One-class bootstrap draws cannot estimate AUROC and are skipped.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from .crop_experiment import compare_source_scores

    if n_boot < 100 or not 0 < normal_fpr < 1:
        raise ValueError("Use n_boot >= 100 e normal_fpr em (0,1).")
    if scores.duplicated(["mode", "split", "source_id"]).any():
        raise ValueError("Scores duplicados por origem.")
    if scores.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Grupo compartilhado entre calibração e teste.")
    if baseline not in set(scores["mode"]):
        raise ValueError("Baseline ausente.")
    reference = scores[scores["mode"] == baseline].set_index(["split", "source_id"])
    for _, frame in scores.groupby("mode"):
        indexed = frame.set_index(["split", "source_id"])
        if set(indexed.index) != set(reference.index):
            raise ValueError("Variantes devem ter as mesmas origens de calibração e teste.")
        if not indexed.loc[reference.index, ["group_id", "y_true"]].equals(reference[["group_id", "y_true"]]):
            raise ValueError("Grupos ou rótulos divergentes entre variantes.")
    _, decisions = compare_source_scores(scores, normal_fpr)
    base = decisions[decisions["mode"] == baseline].set_index("source_id").sort_index()
    y = base.y_true.to_numpy()
    groups = base.group_id.astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Bootstrap exige pelo menos dois grupos de teste.")
    indices = [np.flatnonzero(groups == group) for group in unique]

    def metrics(frame, idx):
        truth = y[idx]
        return np.array([
            roc_auc_score(truth, frame.pred_score.to_numpy()[idx]),
            average_precision_score(truth, frame.pred_score.to_numpy()[idx]),
            frame.pred_label.to_numpy()[idx][truth == 1].mean(),
            frame.pred_label.to_numpy()[idx][truth == 0].mean(),
        ])

    rows = []
    for mode, frame in decisions.groupby("mode"):
        if mode == baseline:
            continue
        other = frame.set_index("source_id").loc[base.index]
        all_indices = np.arange(len(base))
        delta = metrics(other, all_indices) - metrics(base, all_indices)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            idx = np.concatenate([indices[i] for i in rng.integers(0, len(unique), len(unique))])
            if len(np.unique(y[idx])) == 2:
                draws.append(metrics(other, idx) - metrics(base, idx))
        if len(draws) < 100:
            raise ValueError("Poucas réplicas com ambas as classes; aumente grupos/n_boot.")
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        for i, metric in enumerate(("AUROC", "AP", "recall", "FPR")):
            rows.append(dict(mode=mode, baseline=baseline, metric=metric, delta=delta[i],
                             ci95_low=low[i], ci95_high=high[i], valid_bootstraps=len(draws),
                             n_groups=len(unique)))
    return pd.DataFrame(rows)
