"""Figuras comparativas dos resultados para apresentação (PNGs e um PDF em slides 16:9)."""

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
