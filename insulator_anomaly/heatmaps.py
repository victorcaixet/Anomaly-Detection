"""Diagnóstico descritivo dos mapas PatchCore; não altera o modelo nem os rótulos."""

from pathlib import Path
import hashlib
import json
import math
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .evaluation import predict_checked_heatmap


def validate_heatmap(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("O mapa deve ser 2D, finito e ter ao menos 2 pixels por eixo.")
    return array


def heatmap_gradient(array, sigma=1.0):
    """Derivada espacial após suavização fixa, em unidades do mapa/pixel."""
    array = validate_heatmap(array)
    if sigma < 0:
        raise ValueError("sigma deve ser >= 0.")
    if sigma:
        radius = max(1, math.ceil(3 * sigma))
        array = cv2.GaussianBlur(
            array,
            (2 * radius + 1, 2 * radius + 1),
            sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    dy, dx = np.gradient(array)
    return np.hypot(dx, dy)


def component_areas(binary):
    """Áreas dos componentes com conectividade 8; sem filtro de tamanho."""
    binary = np.asarray(binary, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("Máscara deve ser 2D.")
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)


def read_aligned_mask(root, mode, image_path, dataset_root, shape):
    """Máscaras binárias na geometria exata da entrada processada; sem redimensionar."""
    if root is None:
        return None
    relative = (
        Path(image_path).resolve().relative_to((Path(dataset_root) / mode).resolve())
    )
    path = Path(root) / mode / relative.with_suffix(".png")
    if not path.is_file():
        raise FileNotFoundError(f"Máscara configurada, mas ausente: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or mask.shape != shape or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(
            f"Máscara deve ser binária 0/1 ou 0/255, 2D e ter shape {shape}: {path}"
        )
    return mask > 0


def heatmap_features(
    array, threshold, roi=None, sigma=1.0, border_fraction=0.05, defect=None
):
    """Intensidade, extensão, componentes, borda e gradiente; sem min-max por imagem."""
    array = validate_heatmap(array)
    roi = np.ones(array.shape, bool) if roi is None else np.asarray(roi, dtype=bool)
    if roi.shape != array.shape or not roi.any():
        raise ValueError("ROI vazia ou incompatível com o mapa.")
    if not np.isfinite(threshold) or not 0 < border_fraction < 0.5:
        raise ValueError(
            "Limiar deve ser finito e border_fraction deve estar entre 0 e 0.5."
        )
    values = array[roi]
    top_k = max(1, math.ceil(values.size * 0.01))
    gradient = heatmap_gradient(array, sigma)
    # Exclui da estatística do gradiente a faixa junto ao limite da ROI.
    radius = math.ceil(3 * sigma) + 1
    interior = cv2.erode(
        roi.astype(np.uint8),
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    hot = (
        array > threshold
    ) & roi  # Empates não criam área quente em mapas constantes.
    areas = component_areas(hot)
    height, width = array.shape
    by, bx = (
        max(1, math.ceil(height * border_fraction)),
        max(1, math.ceil(width * border_fraction)),
    )
    border = np.ones(array.shape, bool)
    border[by : height - by, bx : width - bx] = False
    peak_index = np.argmax(np.where(roi, array, -np.inf))
    peak_y, peak_x = np.unravel_index(peak_index, array.shape)
    result = {
        "map_height": height,
        "map_width": width,
        "roi_pixels": int(roi.sum()),
        "map_min": float(values.min()),
        "map_mean": float(values.mean()),
        "map_peak": float(values.max()),
        "map_q95": float(np.quantile(values, 0.95)),
        "map_q99": float(np.quantile(values, 0.99)),
        "top1_mean": float(np.partition(values, values.size - top_k)[-top_k:].mean()),
        "hot_area_fraction": float(hot.sum() / roi.sum()),
        "components": len(areas),
        "largest_component_fraction": float(areas[0] / roi.sum()) if areas else 0.0,
        "largest_component_hot_share": float(areas[0] / hot.sum()) if areas else 0.0,
        "hot_border_share": float((hot & border).sum() / hot.sum())
        if hot.any()
        else np.nan,
        "border_roi_fraction": float((roi & border).sum() / roi.sum()),
        "peak_x_fraction": float(peak_x / (width - 1)),
        "peak_y_fraction": float(peak_y / (height - 1)),
        "gradient_p95": float(np.quantile(gradient[interior], 0.95))
        if interior.any()
        else np.nan,
        "gradient_p99": float(np.quantile(gradient[interior], 0.99))
        if interior.any()
        else np.nan,
        "gradient_max": float(gradient[interior].max()) if interior.any() else np.nan,
        "gradient_mean": float(gradient[interior].mean()) if interior.any() else np.nan,
        "gradient_valid_pixels": int(interior.sum()),
        "map_constant": bool(np.ptp(values) <= np.finfo(float).eps),
    }
    result["hot_border_enrichment"] = (
        result["hot_border_share"] / result["border_roi_fraction"]
        if result["border_roi_fraction"] > 0
        else np.nan
    )
    if defect is not None:
        defect = np.asarray(defect, dtype=bool)
        if defect.shape != array.shape or (defect & ~roi).any():
            raise ValueError(
                "Máscara de defeito incompatível ou com defeito fora da ROI."
            )
        tp, fp, fn = (
            int((hot & defect).sum()),
            int((hot & ~defect).sum()),
            int((~hot & defect).sum()),
        )
        peak_support = (array == values.max()) & roi
        result.update(
            defect_pixels=int(defect.sum()),
            defect_iou=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
            defect_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            defect_recall=tp / (tp + fn) if tp + fn else np.nan,
            defect_precision=tp / (tp + fp) if tp + fp else np.nan,
            peak_on_defect=float(defect[peak_y, peak_x])
            if defect.any() and peak_support.sum() == 1
            else np.nan,
            peak_support_defect_fraction=float(
                (peak_support & defect).sum() / peak_support.sum()
            )
            if defect.any()
            else np.nan,
        )
    return result, hot, gradient


def select_heatmap_cases(
    frame, max_per_group=40, reference_count=32, seed=42, fn_score_range=None
):
    if reference_count < 2 or (max_per_group is not None and max_per_group < 1):
        raise ValueError("Use ao menos 2 referências e max_per_group positivo ou None.")
    frame = frame.copy().sort_values("image_path").reset_index(drop=True)
    if frame.empty or frame.image_path.duplicated().any():
        raise ValueError("Predições vazias ou com caminhos duplicados.")
    if not frame.y_true.isin([0, 1]).all() or not frame.pred_label.isin([0, 1]).all():
        raise ValueError("Rótulos devem ser 0/1.")
    frame["group"] = np.select(
        [
            (frame.y_true == 1) & (frame.pred_label == 0),
            (frame.y_true == 0) & (frame.pred_label == 1),
            (frame.y_true == 1) & (frame.pred_label == 1),
        ],
        ["FN", "FP", "TP"],
        default="TN",
    )
    normal = frame[frame.y_true == 0]
    if len(normal) < 2:
        raise ValueError("São necessárias pelo menos duas imagens boas de referência.")
    reference = normal.sample(n=min(reference_count, len(normal)), random_state=seed)
    samples = []
    for group in ("FN", "FP", "TP", "TN"):
        subset = frame[frame.group == group]
        if group == "FN" and fn_score_range is not None:
            subset = subset[subset.pred_score.between(*fn_score_range)]
        samples.append(
            subset
            if max_per_group is None
            else subset.sample(n=min(max_per_group, len(subset)), random_state=seed)
        )
    cases = pd.concat(samples).drop_duplicates("image_path")
    selection = (
        pd.concat([cases, reference])
        .drop_duplicates("image_path")
        .sort_values("image_path")
        .reset_index(drop=True)
    )
    selection["is_reference"] = selection.image_path.isin(reference.image_path)
    selection["is_case"] = selection.image_path.isin(cases.image_path)
    return selection


def plot_heatmap_case(
    image_path,
    array,
    hot,
    gradient,
    reference_pixels,
    threshold,
    features,
    title,
    map_limits,
    gradient_limit,
    figure_path,
    show=False,
):
    import matplotlib.pyplot as plt

    with Image.open(image_path) as image:
        image = np.asarray(image.convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    try:
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Entrada processada")
        axes[0, 1].imshow(image)
        heat = axes[0, 1].imshow(
            array, cmap="inferno", alpha=0.65, vmin=map_limits[0], vmax=map_limits[1]
        )
        axes[0, 1].set_title("Mapa: escala comum dentro deste modo")
        fig.colorbar(heat, ax=axes[0, 1], fraction=0.046)
        axes[0, 2].imshow(image)
        overlay = np.zeros((*hot.shape, 4))
        overlay[hot] = [1, 0, 0, 0.65]
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title(
            f"Regiões acima da referência\nlimiar pixel={threshold:.5g}"
        )
        grad = axes[1, 0].imshow(gradient, cmap="magma", vmin=0, vmax=gradient_limit)
        axes[1, 0].set_title("Gradiente espacial suavizado (mapa/pixel)")
        fig.colorbar(grad, ax=axes[1, 0], fraction=0.046)
        bins = np.linspace(*map_limits, 51)
        axes[1, 1].hist(
            reference_pixels,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Pixels das boas de referência",
        )
        axes[1, 1].hist(
            array.ravel(),
            bins=bins,
            density=True,
            alpha=0.5,
            label="Pixels desta imagem (quadro todo)",
        )
        axes[1, 1].axvline(
            threshold, color="red", linestyle="--", label="Limiar descritivo"
        )
        axes[1, 1].set(xlabel="Valor do mapa", ylabel="Densidade")
        axes[1, 1].legend(fontsize=8)
        summary = (
            f"Pico: {features['map_peak']:.5g}\nMédia do top 1%: {features['top1_mean']:.5g}\n"
            f"Área acima do limiar: {features['hot_area_fraction']:.2%}\n"
            f"Maior componente / ROI: {features['largest_component_fraction']:.2%}\n"
            f"Componentes: {features['components']}\n"
            f"Fração quente na borda: {features['hot_border_share']:.2%}\n"
            f"Gradiente P95: {features['gradient_p95']:.5g}\n\n"
            "Indicadores descritivos.\nGradiente não é importância do pixel.\nSem máscara do defeito, não mede acerto espacial."
        )
        axes[1, 2].text(0, 1, summary, va="top", fontsize=11)
        for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
            axis.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=140)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def run_heatmap_analysis(
    config,
    evaluations,
    modes=("gray",),
    max_per_group=40,
    reference_count=32,
    fn_score_range=None,
    show_first=8,
    roi_root=None,
    defect_root=None,
):
    """Infere novamente com os checkpoints avaliados; sem fit e sem alterar decisões."""
    import gc
    import torch
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    dataset_root, results_root = config.processed_root, config.results_root
    persistent_results_root = config.persistent_results_root
    seed = config.random_seed
    quantiles, primary_quantile, gradient_sigma = (0.95, 0.99, 0.995, 0.999), 0.995, 1.0
    all_rows = []
    for mode in modes:
        frame, metrics, _ = evaluations[mode]
        selection = select_heatmap_cases(
            frame, max_per_group, reference_count, seed, fn_score_range
        )
        output = Path(results_root) / mode / "heatmap_analysis"
        output.mkdir(parents=True, exist_ok=True)
        records = []
        model = Patchcore.load_from_checkpoint(
            metrics["checkpoint_path"], map_location="cpu", weights_only=False
        )
        engine = None
        try:
            engine = Engine(
                accelerator="auto",
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=output,
            )
            for number, row in enumerate(selection.itertuples(index=False), 1):
                path = Path(row.image_path)
                array = validate_heatmap(predict_checked_heatmap(engine, model, row))
                key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
                map_path = output / f"{key}_map.npy"
                np.save(map_path, array.astype(np.float32))
                records.append(dict(row._asdict(), map_path=str(map_path), key=key))
                if number % 20 == 0 or number == len(selection):
                    print(f"{mode}: {number}/{len(selection)} mapas calculados.")
        finally:
            del model, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reference_parts = []
        shape = None
        for record in records:
            array = np.load(record["map_path"])
            if shape is not None and array.shape != shape:
                raise ValueError(
                    "Use mapas com a mesma resolução para comparar gradientes e áreas."
                )
            shape = array.shape
            if record["is_reference"]:
                roi = read_aligned_mask(
                    roi_root, mode, record["image_path"], dataset_root, shape
                )
                values = array.ravel() if roi is None else array[roi]
                if values.size == 0:
                    raise ValueError("Referência com ROI vazia.")
                reference_parts.append(values)
        reference_pixels = np.concatenate(reference_parts)
        thresholds = {q: float(np.quantile(reference_pixels, q)) for q in quantiles}
        minimum = min(float(np.load(record["map_path"]).min()) for record in records)
        maximum = max(float(np.load(record["map_path"]).max()) for record in records)
        limits = (minimum, maximum if maximum > minimum else minimum + 1e-6)
        gradient_limit = max(
            float(heatmap_gradient(np.load(record["map_path"]), gradient_sigma).max())
            for record in records
        )
        gradient_limit = max(gradient_limit, 1e-12)
        rows, sweeps = [], []
        for record in records:
            array = np.load(record["map_path"])
            roi = read_aligned_mask(
                roi_root, mode, record["image_path"], dataset_root, array.shape
            )
            defect = read_aligned_mask(
                defect_root, mode, record["image_path"], dataset_root, array.shape
            )
            features, hot, gradient = heatmap_features(
                array, thresholds[primary_quantile], roi, gradient_sigma, defect=defect
            )
            row = dict(
                record,
                mode=mode,
                reference_quantile=primary_quantile,
                pixel_threshold=thresholds[primary_quantile],
                roi_source="full_frame" if roi is None else "object_mask",
                has_defect_mask=defect is not None,
                **features,
            )
            rows.append(row)
            for quantile, threshold in thresholds.items():
                areas = component_areas(
                    (array > threshold)
                    & (np.ones(array.shape, bool) if roi is None else roi)
                )
                sweeps.append(
                    dict(
                        image_path=record["image_path"],
                        group=record["group"],
                        quantile=quantile,
                        pixel_threshold=threshold,
                        hot_area_fraction=sum(areas) / features["roi_pixels"],
                        components=len(areas),
                        largest_component_fraction=(areas[0] if areas else 0)
                        / features["roi_pixels"],
                    )
                )
            if record["is_case"]:
                figure = output / f"{record['key']}_diagnostic.png"
                plot_heatmap_case(
                    record["image_path"],
                    array,
                    hot,
                    gradient,
                    reference_pixels,
                    thresholds[primary_quantile],
                    features,
                    f"{mode} | {record['group']} | score={record['pred_score']:.4f} | {Path(record['image_path']).name}",
                    limits,
                    gradient_limit,
                    figure,
                    show=sum(r["is_case"] for r in rows) <= show_first,
                )
                row["figure"] = str(figure)
        table = pd.DataFrame(rows)
        # Percentil empírico entre imagens boas; não é p-valor nem probabilidade de defeito.
        for column in ("map_peak", "top1_mean", "hot_area_fraction", "gradient_p95"):
            reference_values = table.loc[table.is_reference, column].dropna().to_numpy()
            table[f"{column}_normal_percentile"] = table[column].map(
                lambda value: (
                    float(
                        100
                        * (
                            np.count_nonzero(reference_values < value)
                            + 0.5 * np.count_nonzero(reference_values == value)
                        )
                        / len(reference_values)
                    )
                    if len(reference_values) and np.isfinite(value)
                    else np.nan
                )
            )
        table.to_csv(output / "heatmap_metrics.csv", index=False)
        pd.DataFrame(sweeps).to_csv(output / "threshold_sweep.csv", index=False)
        columns = [
            "map_peak",
            "top1_mean",
            "hot_area_fraction",
            "largest_component_fraction",
            "gradient_p95",
            "hot_border_enrichment",
        ]
        summary = (
            table[table.is_case]
            .groupby("group")[columns]
            .agg(["count", "median", "mean"])
        )
        summary.to_csv(output / "group_summary.csv")
        checkpoint = Path(metrics["checkpoint_path"])
        metadata = dict(
            mode=mode,
            checkpoint_path=str(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
            reference_source="test/good: exploratorio",
            reference_images=table.loc[table.is_reference, "image_path"].tolist(),
            seed=seed,
            max_per_group=max_per_group,
            fn_score_range=fn_score_range,
            quantiles=list(quantiles),
            thresholds={str(k): v for k, v in thresholds.items()},
            primary_quantile=primary_quantile,
            gradient_sigma=gradient_sigma,
            border_fraction=0.05,
            map_limits=limits,
            gradient_limit=gradient_limit,
            roi_root=str(roi_root) if roi_root is not None else None,
            defect_root=str(defect_root) if defect_root is not None else None,
            warning="Descritivo no teste; referencias podem incluir FP e casos analisados. Sem recalibracao da decisao.",
        )
        (output / "analysis_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if persistent_results_root is not None:
            destination = Path(persistent_results_root) / mode / "heatmap_analysis"
            if output.resolve() != destination.resolve():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(output, destination, dirs_exist_ok=True)
                print(f"Análise persistida em: {destination}")
        print(
            f"{mode}: tabelas e figuras em {output}; referências normais={len(reference_parts)}."
        )
        print(summary.to_string())
        all_rows.append(table)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
