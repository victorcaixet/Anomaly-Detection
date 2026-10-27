"""Figuras do experimento SAM + PatchCore + camada de heatmaps, para o README.

Entrada: pasta do experimento de `run_crop_experiment` (source_scores.csv, map_manifest.csv,
map_evaluation/map_metrics.csv) e o projeto com as imagens geradas pelo SAM.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import build_dataset_records, safe_output_name
from .preprocessing import read_image
from .report import BAD, GOOD

METHODS = ("absolute", "mebin", "contrast")


def val_threshold(source_scores, fpr=0.05):
    val = source_scores[source_scores.split == "val"].pred_score
    return float(np.quantile(val, 1 - fpr, method="higher"))


def pick_cases(test, threshold):
    """Um caso por quadrante, nos extremos de score: TP mais forte, FN mais fraco, FP mais forte, TN mais fraco."""
    pos, neg = test[test.y_true == 1], test[test.y_true == 0]
    return [("TP", pos.loc[pos.pred_score.idxmax()]), ("FN", pos.loc[pos.pred_score.idxmin()]),
            ("FP", neg.loc[neg.pred_score.idxmax()]), ("TN", neg.loc[neg.pred_score.idxmin()])]


def fig_scores(source_scores, threshold):
    test = source_scores[source_scores.split == "test"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(test.pred_score.min(), test.pred_score.max(), 30)
    for label, name, color in ((0, "normal", GOOD), (1, "ferrugem", BAD)):
        ax.hist(test.pred_score[test.y_true == label], bins=bins, alpha=0.6, color=color,
                label=f"{name} (n={int((test.y_true == label).sum())})")
    ax.axvline(threshold, color="black", ls="--", label="limiar (5% FPR na validação normal)")
    alarm = test.pred_score > threshold
    tp, fp = int(alarm[test.y_true == 1].sum()), int(alarm[test.y_true == 0].sum())
    ax.set(xlabel="score do PatchCore (máx. do mapa)", ylabel="imagens",
           title=f"Teste: {tp}/{int(test.y_true.sum())} ferrugens e {fp}/{int((test.y_true == 0).sum())} falsos alarmes")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_layer(source_scores, map_metrics, n_boot=2000, seed=0):
    """AUROC no teste (IC95 bootstrap) do PatchCore e de cada score do mapa."""
    test = source_scores[source_scores.split == "test"][["source_id", "y_true", "pred_score"]]
    scores = {"PatchCore": test.set_index("source_id").pred_score}
    for method in METHODS:
        part = map_metrics[(map_metrics.method == method) & (map_metrics.split == "test")]
        scores[method] = part.groupby("source_id").score.max()
    frame = pd.DataFrame(scores).join(test.set_index("source_id").y_true).dropna()
    y, rng = frame.y_true.to_numpy(), np.random.default_rng(seed)
    idx = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if y[i].min() != y[i].max()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for x, name in enumerate(scores):
        s = frame[name].to_numpy()
        auc = roc_auc_score(y, s)
        low, high = np.percentile([roc_auc_score(y[i], s[i]) for i in idx], [2.5, 97.5])
        ax.bar(x, auc, color=GOOD if name == "PatchCore" else "#7570b3", yerr=[[auc - low], [high - auc]], capsize=5)
        ax.text(x, 0.03, f"{auc:.2f}", ha="center", color="white", fontsize=14)
    ax.axhline(0.5, color="gray", ls=":", label="acaso")
    ax.set(xticks=range(len(scores)), xticklabels=list(scores), ylim=(0, 1), ylabel="AUROC",
           title="A camada de heatmaps melhora a detecção? (IC 95%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_pipeline(cases, source_dataset, boxes, debug_root, map_manifest):
    """Linhas = TP/FN/FP/TN; colunas = original + caixa, máscara SAM, entrada do modelo, mapa."""
    names = {safe_output_name(r.image_path, source_dataset): r.image_path for r in build_dataset_records(source_dataset)}
    fig, axes = plt.subplots(len(cases), 4, figsize=(11, 2.8 * len(cases)), squeeze=False)
    heads = ["original + caixa", "máscara SAM", "entrada (fundo cinza)", "mapa de anomalia"]
    vmax = max(float(np.load(map_manifest.set_index("image_path").map_path[row.image_path]).max()) for _, row in cases)
    for r, (kind, row) in enumerate(cases):
        original = names[Path(row.image_path).name]
        relative = original.relative_to(source_dataset).as_posix()
        group = "_".join(Path(row.image_path).name.split("__")[:2])
        mask = Path(debug_root) / group / (Path(row.image_path).stem + "__mask.png")
        x0, y0, x1, y1 = boxes[relative]
        panels = [read_image(original)[..., ::-1], read_image(mask), read_image(Path(row.image_path))[..., ::-1]]
        for c, image in enumerate(panels):
            axes[r, c].imshow(image)
        axes[r, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
        axes[r, 3].imshow(panels[2])
        heat = np.load(map_manifest.set_index("image_path").map_path[row.image_path])
        axes[r, 3].imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        axes[r, 0].set_ylabel(f"{kind}\n{'ferrugem' if row.y_true else 'normal'}\nscore {row.pred_score:.0f}",
                              rotation=0, ha="right", va="center", fontsize=11)
    for c, head in enumerate(heads):
        axes[0, c].set_title(head)
    for ax in axes.ravel():
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    fig.suptitle("Do original ao mapa (escala de cor única)", fontsize=14)
    fig.tight_layout()
    return fig


def make_showcase(experiment_root, project_root, out_dir, fpr=0.05):
    exp, proj, out = Path(experiment_root), Path(project_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(exp / "source_scores.csv")
    maps = pd.read_csv(exp / "map_evaluation" / "map_metrics.csv")
    manifest = pd.read_csv(exp / "map_manifest.csv")
    threshold = val_threshold(scores, fpr)
    cases = pick_cases(scores[scores.split == "test"].merge(
        manifest[manifest.split == "test"][["source_id", "image_path"]], on="source_id"), threshold)
    boxes = json.loads((proj / "boxes_all.json").read_text(encoding="utf-8"))
    figures = {"scores": fig_scores(scores, threshold), "camada_heatmaps": fig_layer(scores, maps),
               "pipeline": fig_pipeline(cases, proj / "glass-insulator", boxes,
                                        proj / "patchcore_insulators" / "debug", manifest)}
    paths = []
    for name, fig in figures.items():
        paths.append(out / f"{name}.png")
        fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
        plt.close(fig)
    return paths
