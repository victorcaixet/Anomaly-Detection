"""Figuras dos resultados: relatório de apresentação (PNGs e PDF 16:9) e figuras do README.

O relatório lê as avaliações salvas por `evaluate_all`. As figuras do README leem a pasta do
experimento de `run_crop_experiment` (source_scores.csv, map_manifest.csv, map_evaluation/) e o
projeto com as imagens geradas pelo SAM.
"""

from pathlib import Path
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from .data import build_dataset_records, safe_output_name
from .evaluation import val_threshold
from .preprocessing import read_image

GOOD, BAD = "#2c7fb8", "#e6550d"
SLIDE = (13.33, 7.5)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def load_results(results_root, modes=("original", "blur", "gray")):
    """Lê os arquivos salvos por evaluate_all: {modo: (frame, metrics, curva)}."""
    out = {}
    for mode in modes:
        base = Path(results_root) / mode / "precision_recall_test"
        out[mode] = (pd.read_csv(base / "predictions.csv"),
                     json.loads((base / "metrics.json").read_text(encoding="utf-8")),
                     pd.read_csv(base / "precision_recall_curve.csv"))
    return out


def fig_metrics(results):
    rows = [dict(modo=m, AUROC=met["auroc"], AP=met["average_precision"],
                 **{"Recall (checkpoint)": met["checkpoint"]["recall"],
                    "FPR (checkpoint)": met["checkpoint"]["false_positive_rate"]})
            for m, (_, met, _) in results.items()]
    table = pd.DataFrame(rows).set_index("modo")
    ax = table.plot.bar(figsize=SLIDE, rot=0, width=0.8, color=["#1b9e77", "#7570b3", GOOD, BAD])
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=11)
    ax.set(title="Métricas por modo de fundo", ylim=(0, 1.05), xlabel="")
    ax.legend(ncol=4, loc="upper center", frameon=False)
    return ax.figure


def fig_scores(results):
    fig, axes = plt.subplots(1, len(results), figsize=SLIDE, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (mode, (frame, _, _)) in zip(np.atleast_1d(axes), results.items()):
        for x, (label, color) in enumerate(((0, GOOD), (1, BAD))):
            s = frame.pred_score[frame.y_true == label]
            ax.boxplot(s, positions=[x], widths=0.5, showfliers=False)
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(s)), s, s=8, color=color, alpha=0.4)
        ax.set(title=mode, xticks=[0, 1], xticklabels=["normal", "anomalia"])
    np.atleast_1d(axes)[0].set_ylabel("score de anomalia")
    fig.suptitle("Distribuição dos scores no teste", fontsize=20)
    return fig


def fig_pr(results):
    fig, ax = plt.subplots(figsize=SLIDE)
    for (mode, (frame, met, curve)) in results.items():
        ax.step(curve.recall, curve.precision, where="post", lw=2.5, label=f"{mode} (AP={met['average_precision']:.2f})")
    prevalence = float(next(iter(results.values()))[0].y_true.mean())
    ax.axhline(prevalence, color="gray", ls=":", label=f"acaso ({prevalence:.2f})")
    ax.set(title="Precisão × recall", xlabel="recall", ylabel="precisão", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False)
    return fig


def fig_confusion(results):
    fig, axes = plt.subplots(1, len(results), figsize=SLIDE)
    for ax, (mode, (_, met, _)) in zip(np.atleast_1d(axes), results.items()):
        c = met["checkpoint"]
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(grid / grid.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, v, ha="center", va="center", fontsize=22,
                    color="white" if grid[i, j] / grid[i].sum() > 0.5 else "black")
        ax.set(title=mode, xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "anomalia"],
               yticklabels=["normal", "anomalia"], xlabel="previsto", ylabel="real")
        ax.grid(False)
    fig.suptitle("Matriz de confusão (limiar do checkpoint)", fontsize=20)
    return fig


def fig_dates(frame, mode):
    date = frame.image_path.str.extract(r"Fotos_(\d\d-\d\d-\d{4})")[0]
    table = frame.assign(date=date).groupby(["date", "y_true"]).pred_score.agg(["mean", "size"]).unstack()
    table = table.sort_index(key=lambda s: pd.to_datetime(s, format="%d-%m-%Y"))
    fig, ax = plt.subplots(figsize=SLIDE)
    x = np.arange(len(table))
    ax.bar(x - 0.2, table[("mean", 0)], 0.4, color=GOOD, label="normal")
    ax.bar(x + 0.2, table[("mean", 1)].fillna(0), 0.4, color=BAD, label="anomalia")
    labels = [f"{d[:5]}\n{int(n0)}/{int(n1)}" for d, n0, n1 in
              zip(table.index, table[("size", 0)].fillna(0), table[("size", 1)].fillna(0))]
    ax.set(xticks=x, xticklabels=labels, ylabel="score médio", title=f"Score por data de captura ({mode})",
           xlabel="data (dd-mm) e nº de imagens normal/anomalia")
    ax.legend(frameon=False)
    return fig


def select_cases(frame, per_group=6, label_column="pred_label"):
    """TP/FN/FP/TN: casos espaçados pela ordem de score, representativos de cada grupo."""
    rows = []
    for name, (truth, pred) in dict(TP=(1, 1), FN=(1, 0), FP=(0, 1), TN=(0, 0)).items():
        part = frame[(frame.y_true == truth) & (frame[label_column] == pred)].sort_values("pred_score")
        if len(part):
            rows.append(part.iloc[np.linspace(0, len(part) - 1, min(per_group, len(part))).round().astype(int)]
                        .assign(group=name))
    return pd.concat(rows, ignore_index=True)


def fig_gallery(cases, maps, mode):
    """Linhas = TP, FN, FP, TN; colunas = casos; imagem com o mapa de anomalia sobreposto."""
    names = [g for g in ("TP", "FN", "FP", "TN") if g in set(cases.group)]
    columns = int(cases.groupby("group").size().max())
    vmax = max(float(m.max()) for m in maps.values())
    fig, axes = plt.subplots(len(names), columns, figsize=SLIDE, squeeze=False)
    titles = dict(TP="acerto: anomalia", FN="erro: anomalia perdida", FP="erro: falso alarme", TN="acerto: normal")
    for ax in axes.ravel():
        ax.axis("off")
    for r, name in enumerate(names):
        for c, row in enumerate(cases[cases.group == name].itertuples()):
            ax = axes[r, c]
            ax.imshow(Image.open(row.image_path).convert("RGB"))
            ax.imshow(maps[row.image_path], cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
            ax.set_title(f"{row.pred_score:.2f}", fontsize=12)
        axes[r, 0].text(-0.08, 0.5, f"{name}\n{titles[name]}", transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=12, color=BAD if name in ("FN", "FP") else GOOD)
    fig.suptitle(f"Casos e mapas de anomalia ({mode}) — escala de cor única", fontsize=20)
    fig.tight_layout(rect=(0.06, 0, 1, 0.94))
    return fig


def fig_sessions(table):
    """Global × por grupo (saída de session_experiment.evaluate com a coluna 'sistema')."""
    part = table[table.score == "max"]
    pivot = part.pivot(index="group", columns="sistema", values=["AUROC", "AP"])
    fig, axes = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    for ax, metric in zip(axes, ("AUROC", "AP")):
        pivot[metric].plot.bar(ax=ax, rot=0, color=["#7570b3", "#1b9e77"], width=0.8)
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.2f", fontsize=11)
        ax.set(title=metric, xlabel="grupo de sessões", ylim=(0, 1.25))
    fig.suptitle("Modelo global × um modelo por grupo (mesmo teste)", fontsize=20)
    return fig


def make_report(results, out_dir, mode="gray", cases=None, maps=None, sessions=None):
    """Salva PNGs e `resultados.pdf`. `cases`/`maps` e `sessions` são opcionais."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = {"1_metricas": fig_metrics(results), "2_scores": fig_scores(results),
               "3_precisao_recall": fig_pr(results), "4_confusao": fig_confusion(results),
               "5_por_data": fig_dates(results[mode][0], mode)}
    if cases is not None and maps is not None:
        figures["6_casos"] = fig_gallery(cases, maps, mode)
    if sessions is not None:
        figures["7_global_vs_grupo"] = fig_sessions(sessions)
    with PdfPages(out / "resultados.pdf") as pdf:
        for name, fig in figures.items():
            fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return sorted(out.iterdir())


METHODS = ("absolute", "mebin", "contrast")


def pick_cases(test, threshold):
    """Um caso por quadrante, nos extremos de score: TP mais forte, FN mais fraco, FP mais forte, TN mais fraco."""
    pos, neg = test[test.y_true == 1], test[test.y_true == 0]
    return [("TP", pos.loc[pos.pred_score.idxmax()]), ("FN", pos.loc[pos.pred_score.idxmin()]),
            ("FP", neg.loc[neg.pred_score.idxmax()]), ("TN", neg.loc[neg.pred_score.idxmin()])]


def fig_score_hist(source_scores, threshold, fpr=0.4):
    test = source_scores[source_scores.split == "test"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(test.pred_score.min(), test.pred_score.max(), 30)
    for label, name, color in ((0, "normal", GOOD), (1, "ferrugem", BAD)):
        ax.hist(test.pred_score[test.y_true == label], bins=bins, alpha=0.6, color=color,
                label=f"{name} (n={int((test.y_true == label).sum())})")
    ax.axvline(threshold, color="black", ls="--", label=f"limiar ({fpr:.0%} FPR na validação normal)")
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


def make_showcase(experiment_root, project_root, out_dir, fpr=0.4):
    exp, proj, out = Path(experiment_root), Path(project_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(exp / "source_scores.csv")
    maps = pd.read_csv(exp / "map_evaluation" / "map_metrics.csv")
    manifest = pd.read_csv(exp / "map_manifest.csv")
    threshold = val_threshold(scores[scores.split == "val"].pred_score, fpr)
    cases = pick_cases(scores[scores.split == "test"].merge(
        manifest[manifest.split == "test"][["source_id", "image_path"]], on="source_id"), threshold)
    boxes = json.loads((proj / "boxes_all.json").read_text(encoding="utf-8"))
    figures = {"scores": fig_score_hist(scores, threshold, fpr), "camada_heatmaps": fig_layer(scores, maps),
               "pipeline": fig_pipeline(cases, proj / "glass-insulator", boxes,
                                        proj / "patchcore_insulators" / "debug", manifest)}
    paths = []
    for name, fig in figures.items():
        paths.append(out / f"{name}.png")
        fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
        plt.close(fig)
    return paths
