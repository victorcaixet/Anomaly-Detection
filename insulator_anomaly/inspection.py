"""Inspeção visual de ROIs, erros e falsos negativos em quatro painéis."""

from pathlib import Path
import gc
import hashlib
import math
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from .data import IMAGE_EXTENSIONS as EXTENSIONS, safe_output_name


def analyze_false_negatives(
    config,
    evaluations,
    modes=("gray",),
    score_range=(0.25, 0.50),
    show_first=20,
    confirmed_mask_filenames=None,
):
    import matplotlib.pyplot as plt
    import torch
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from .evaluation import predict_checked_heatmap

    # Vinculo exato pelo nome gerado no pipeline; nomes ambiguos nao sao adivinhados.
    source_index = {}
    if config.dataset_root.is_dir():
        for path in config.dataset_root.rglob("*"):
            if path.suffix.lower() in EXTENSIONS:
                source_index.setdefault(
                    safe_output_name(path, config.dataset_root), set()
                ).add(path)
    manifest_path = (
        config.project_root / "patchcore_insulators" / "preprocessing_manifest.csv"
    )
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        for _, row in manifest.iterrows():
            source_string = str(row["source_image"]).replace("\\", "/")
            original = Path(source_string)
            if not original.is_file() and "/glass-insulator/" in source_string:
                original = (
                    config.dataset_root / source_string.split("/glass-insulator/", 1)[1]
                )
            if original.is_file():
                for mode in modes:
                    name = Path(str(row[f"processed_{mode}"]).replace("\\", "/")).name
                    source_index.setdefault(name, set()).add(original)

    analysis_rows = []
    for mode in modes:
        frame, metrics, _ = evaluations[mode]
        selected = frame[
            (frame.y_true == 1)
            & (frame.pred_label == 0)
            & frame.pred_score.between(*score_range)
        ].sort_values("pred_score")
        if confirmed_mask_filenames:
            selected = selected[
                selected.image_path.map(
                    lambda p: Path(p).name in confirmed_mask_filenames
                )
            ]
        print(f"{mode}: {len(selected)} FN no intervalo {score_range}.")
        if selected.empty:
            continue
        output = config.results_root / mode / "fn_four_views"
        output.mkdir(parents=True, exist_ok=True)
        model = Patchcore.load_from_checkpoint(
            metrics["checkpoint_path"], map_location="cpu", weights_only=False
        )
        engine = Engine(
            accelerator=config.accelerator,
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=output,
        )
        records = []
        try:
            for _, row in selected.iterrows():
                image_path = Path(row.image_path)
                anomaly_map = predict_checked_heatmap(engine, model, row)
                key = hashlib.sha256(str(image_path).encode()).hexdigest()[:12]
                map_path = output / f"{key}_anomaly_map.npy"
                np.save(map_path, anomaly_map)
                records.append((row, key, map_path))

            # Escala compartilhada entre todos os FN deste modo, sem min-max por imagem.
            map_min = min(float(np.load(path).min()) for _, _, path in records)
            map_max = max(float(np.load(path).max()) for _, _, path in records)
            if map_max <= map_min:
                map_max = map_min + 1e-6
            normal = frame.loc[frame.y_true == 0, "pred_score"].to_numpy()
            anomalous = frame.loc[frame.y_true == 1, "pred_score"].to_numpy()
            # Limites observados da decisao do checkpoint, na escala exportada.
            accepted = frame.loc[frame.pred_label == 0, "pred_score"]
            rejected = frame.loc[frame.pred_label == 1, "pred_score"]
            boundary = None
            if len(accepted) and len(rejected) and accepted.max() <= rejected.min():
                boundary = (float(accepted.max()), float(rejected.min()))
            for index, (row, key, map_path) in enumerate(records):
                image_path = Path(row.image_path)
                candidates = source_index.get(image_path.name, set())
                original_path = next(iter(candidates)) if len(candidates) == 1 else None
                with Image.open(image_path) as image:
                    processed = np.asarray(image.convert("RGB"))
                anomaly_map = np.load(map_path)
                fig, axes = plt.subplots(1, 4, figsize=(22, 5))
                if original_path is not None:
                    with Image.open(original_path) as image:
                        axes[0].imshow(image.convert("RGB"))
                else:
                    axes[0].text(
                        0.5,
                        0.5,
                        "Original nao localizado com vinculo unico",
                        ha="center",
                        wrap=True,
                    )
                axes[0].set_title("1. Imagem original")
                axes[1].imshow(processed, interpolation="nearest")
                axes[1].set_title(
                    f"2. Entrada processada ({mode})\n{processed.shape[1]} x {processed.shape[0]} pixels"
                )
                height, width = processed.shape[:2]
                axes[2].imshow(processed, extent=(0, width, height, 0))
                heat = axes[2].imshow(
                    anomaly_map,
                    cmap="inferno",
                    alpha=0.60,
                    vmin=map_min,
                    vmax=map_max,
                    extent=(0, width, height, 0),
                )
                axes[2].set_title("3. Mapa de anomalia\nEscala comum aos FN deste modo")
                fig.colorbar(heat, ax=axes[2], fraction=0.046)
                for axis in axes[:3]:
                    axis.axis("off")
                axis = axes[3]
                axis.hist(normal, bins=25, density=True, alpha=0.45, label="Boas")
                axis.hist(
                    anomalous, bins=25, density=True, alpha=0.45, label="Defeituosas"
                )
                axis.axvline(
                    row.pred_score,
                    color="black",
                    linewidth=2,
                    label=f"Imagem: {row.pred_score:.4f}",
                )
                if boundary:
                    axis.axvspan(
                        *boundary,
                        color="red",
                        alpha=0.25,
                        label="Intervalo observado da decisao",
                    )
                axis.axvline(
                    metrics["exploratory_threshold"],
                    color="green",
                    linestyle="--",
                    label=f"Limiar exploratorio: {metrics['exploratory_threshold']:.4f}",
                )
                axis.set(
                    title="4. Score e decisao: FN",
                    xlabel="Score (nao e probabilidade)",
                    ylabel="Densidade",
                )
                axis.legend(fontsize=7)
                fig.suptitle(image_path.name, fontsize=10)
                fig.tight_layout()
                figure_path = output / f"{key}_four_views.png"
                fig.savefig(figure_path, dpi=150)
                if index < show_first:
                    plt.show()
                plt.close(fig)
                analysis_rows.append(
                    dict(
                        mode=mode,
                        image_path=str(image_path),
                        original_path=str(original_path) if original_path else "",
                        pred_score=float(row.pred_score),
                        pred_label=int(row.pred_label),
                        exploratory_threshold=metrics["exploratory_threshold"],
                        decision_lower=boundary[0] if boundary else None,
                        decision_upper=boundary[1] if boundary else None,
                        map_min=map_min,
                        map_max=map_max,
                        anomaly_map=str(map_path),
                        figure=str(figure_path),
                        mask_quality="confirmada pelo usuario"
                        if confirmed_mask_filenames
                        else "nao avaliada",
                        defect_type="",
                        defect_visible_at_input="",
                        heatmap_on_defect="",
                        notes="",
                    )
                )
        finally:
            del model, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        destination = config.persistent_results_root / mode / "fn_four_views"
        if output.resolve() != destination.resolve():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output, destination, dirs_exist_ok=True)

    analysis_table = pd.DataFrame(analysis_rows)
    analysis_table.to_csv(config.results_root / "fn_four_views_index.csv", index=False)
    if config.results_root.resolve() != config.persistent_results_root.resolve():
        config.persistent_results_root.mkdir(parents=True, exist_ok=True)
        analysis_table.to_csv(
            config.persistent_results_root / "fn_four_views_index.csv", index=False
        )
    return analysis_table


def show_rois(config, limit=15):
    import matplotlib.pyplot as plt

    paths = sorted((config.output_root / "debug").rglob("*__roi.png"))[:limit]
    if not paths:
        print("Nenhuma ROI de diagnóstico disponível.")
        return
    fig, axes = plt.subplots(
        math.ceil(len(paths) / 3),
        3,
        figsize=(15, 4.5 * math.ceil(len(paths) / 3)),
        squeeze=False,
    )
    for axis, path in zip(axes.ravel(), paths):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{path.parent.name}/{path.name}", fontsize=8)
    for axis in axes.ravel():
        axis.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_error_cases(frame, limit=40):
    import matplotlib.pyplot as plt

    for title, truth, prediction in (
        ("Falsos positivos", 0, 1),
        ("Falsos negativos", 1, 0),
    ):
        cases = (
            frame[(frame.y_true == truth) & (frame.pred_label == prediction)]
            .sort_values("pred_score", ascending=truth == 1)
            .head(limit)
        )
        if cases.empty:
            print(f"{title}: nenhum caso.")
            continue
        rows = math.ceil(len(cases) / 3)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
        for axis, row in zip(axes.ravel(), cases.itertuples(index=False)):
            with Image.open(row.image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{title} | score={row.pred_score:.4f}", fontsize=9)
        for axis in axes.ravel():
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
