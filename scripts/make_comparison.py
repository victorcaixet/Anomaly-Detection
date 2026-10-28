"""Figuras de apresentação: datasets, YOLO x SAM (vidro) e análise dos falsos negativos (FPR alvo 40%).

python scripts/make_comparison.py [--fpr 0.4] [--out docs/img]
Espera o layout de C:/dados (ver constantes); caminhos curtos evitam o limite de 260 caracteres do Windows.
"""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.evaluation import val_threshold
from insulator_anomaly.data import build_dataset_records, safe_output_name
from insulator_anomaly.preprocessing import read_image
from insulator_anomaly.figures import BAD, GOOD

G = Path("C:/dados/gcmp")  # vidro: máscaras YOLO recalculadas sobre os recortes (só para a figura dos datasets)
GLASS_SRC = Path("C:/dados/anomalib_glass/glass-preprocessed/original")
SH = Path("C:/dados/p2")  # haste (SAM): glass-insulator/, glass-preprocessed/, boxes_all.json
SH_EXP = Path("C:/dados/e2")  # source_scores.csv, map_manifest.csv
SESS = Path("C:/dados/sessions_global")  # vidro (YOLO): scores.csv, patchcore_checkpoints/global
SAM_COLOR = "#7570b3"


def metrics(df, col, fpr):
    thr = val_threshold(df[df.split == "val"][col], fpr)
    test = df[df.split == "test"]
    alarm = test[col] > thr
    y = test.y_true == 1
    return dict(thr=thr, AUROC=roc_auc_score(y, test[col]), recall=float(alarm[y].mean()),
                FPR=float(alarm[~y].mean()), TP=int(alarm[y].sum()), FN=int((~alarm[y]).sum()),
                FP=int(alarm[~y].sum()), n_anomaly=int(y.sum()), n_good=int((~y).sum()))


def auroc_ci(df, col, n_boot=2000, seed=0):
    test = df[df.split == "test"]
    y, x = test.y_true.to_numpy(), test[col].to_numpy()
    rng, d = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], x[i]))
    return tuple(np.percentile(d, [2.5, 97.5]))


def glass_heatmaps(run, paths):
    """Mapas de anomalia do checkpoint de `run` para as imagens `paths` (só inferência)."""
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from torch.utils.data import DataLoader

    paths = [Path(p) for p in paths]
    model = Patchcore.load_from_checkpoint(run / "patchcore_checkpoints/global/last.ckpt", map_location="cpu",
                                           weights_only=False)
    engine = Engine(default_root_dir=run / "inference/fn", logger=False, accelerator="gpu", devices=1,
                    enable_progress_bar=False, enable_model_summary=False)
    ds = PredictDataset(path=paths[0])
    ds.image_filenames = paths
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
    batches = engine.predict(model=model, dataloaders=loader, return_predictions=True)
    return [b.anomaly_map.detach().cpu().float().numpy().squeeze() for b in batches]


def locate(base, name):
    """O split do experimento difere da pasta de origem (normais são reparticionados): busca pelo nome."""
    return next(p for sub in ("train/good", "test/good", "test/anomaly") if (p := Path(base) / sub / name).exists())


def style(axs):
    for ax in np.ravel(axs):
        ax.set(xticks=[], yticks=[])
        ax.grid(False)


def debug_mask(name):
    return SH / "patchcore_insulators/debug" / "_".join(name.split("__")[:2]) / (Path(name).stem + "__mask.png")


def fig_datasets(seed=3):
    """Linhas: vidro (YOLO) e haste (SAM) × normal/defeito. Colunas: original, máscara, entrada."""
    rng = np.random.default_rng(seed)
    gl = pd.read_csv(SESS / "scores.csv")
    gl = gl[(gl.split == "test") & gl["name"].map(lambda n: any((G / "yolo/mask" / sub / n).exists() for sub in ("train/good", "test/good", "test/anomaly")))]
    cases = []
    for y, label in ((0, "normal"), (1, "defeito")):
        part = gl[gl.y_true == y]
        name = part.iloc[rng.integers(0, len(part))]["name"]
        cases.append(("Vidro\n(máscara YOLO)", label, read_image(locate(GLASS_SRC, name)),
                      read_image(locate(G / "yolo/mask", name)), read_image(locate(G / "yolo/gray", name)), None))
    scores = pd.read_csv(SH_EXP / "source_scores.csv")
    names = {safe_output_name(r.image_path, SH / "glass-insulator"): r.image_path
             for r in build_dataset_records(SH / "glass-insulator")}
    boxes = json.loads((SH / "boxes_all.json").read_text(encoding="utf-8"))
    for y, label in ((0, "normal"), (1, "defeito")):
        part = scores[(scores.split == "test") & (scores.y_true == y)]
        part = part[part.source_id.map(lambda n: debug_mask(n).exists())]  # máscaras de diagnóstico: só parte das imagens
        name = part.iloc[rng.integers(0, len(part))].source_id
        original = names[name]
        gray = SH / "glass-preprocessed/gray" / "/".join(name.split("__")[:2]) / name
        cases.append(("Haste\n(máscara SAM)", label, read_image(original), read_image(debug_mask(name)),
                      read_image(gray), boxes[original.relative_to(SH / "glass-insulator").as_posix()]))
    fig, axes = plt.subplots(4, 3, figsize=(8.5, 11), squeeze=False)
    for r, (dataset, label, original, mask, gray, box) in enumerate(cases):
        axes[r, 0].imshow(original[..., ::-1])
        if box:
            axes[r, 0].add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec="lime", lw=2))
        axes[r, 1].imshow(mask)
        axes[r, 2].imshow(gray[..., ::-1])
        axes[r, 0].set_ylabel(f"{dataset}\n{label}", rotation=0, ha="right", va="center", fontsize=12,
                              color=BAD if label == "defeito" else GOOD)
    for c, head in enumerate(["original", "máscara", "entrada do modelo"]):
        axes[0, c].set_title(head, fontsize=13)
    style(axes)
    fig.suptitle("Os dois datasets", fontsize=15)
    fig.tight_layout()
    return fig


def fig_yolo_vs_sam(fpr):
    """PatchCore com entrada mascarada pelo YOLO (vidro) x pelo SAM (haste), no mesmo FPR alvo."""
    glass = pd.read_csv(SESS / "scores.csv")
    haste = pd.read_csv(SH_EXP / "source_scores.csv").rename(columns={"pred_score": "max"})
    systems = [("Vidro (máscara YOLO)", glass, GOOD), ("Haste (máscara SAM)", haste, SAM_COLOR)]
    stats = {name: metrics(df, "max", fpr) | dict(AUROC_IC=auroc_ci(df, "max")) for name, df, _ in systems}
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.4)
    for c, (name, df, color) in enumerate(systems):
        m, test = stats[name], df[df.split == "test"]
        ax = fig.add_subplot(grid[0, c])
        bins = np.linspace(test["max"].min(), test["max"].max(), 30)
        ax.hist(test["max"][test.y_true == 0], bins=bins, alpha=0.6, color=GOOD, label=f"normal (n={m['n_good']})")
        ax.hist(test["max"][test.y_true == 1], bins=bins, alpha=0.6, color=BAD, label=f"defeito (n={m['n_anomaly']})")
        ax.axvline(m["thr"], color="black", ls="--", label=f"limiar (FPR alvo {fpr:.0%})")
        ax.set(title=f"{name}: acha {m['TP']}/{m['n_anomaly']}, {m['FP']}/{m['n_good']} falsos alarmes",
               xlabel="score do PatchCore", ylabel="imagens")
        ax.title.set_fontsize(12)
        ax.legend(frameon=False, fontsize=9)
    bars = fig.add_subplot(grid[1, :])
    keys = [("AUROC", "AUROC (IC 95%)"), ("recall", f"recall (FPR alvo {fpr:.0%})"), ("FPR", "falsos alarmes")]
    for x, (k, label) in enumerate(keys):
        for j, (name, _, color) in enumerate(systems):
            m, pos = stats[name], x + (j - 0.5) * 0.38
            err = [[m[k] - m["AUROC_IC"][0]], [m["AUROC_IC"][1] - m[k]]] if k == "AUROC" else None
            bars.bar(pos, m[k], 0.36, color=color, yerr=err, capsize=4, label=name if x == 0 else None)
            bars.text(pos, m[k] + (0.05 if k == "AUROC" else 0.015), f"{m[k]:.2f}", ha="center", fontsize=11)
    bars.set(xticks=range(3), xticklabels=[k[1] for k in keys], ylim=(0, 1.2))
    bars.legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle("PatchCore: entrada mascarada pelo YOLO (vidro) × pelo SAM (haste)", fontsize=15)
    return fig, stats


def fig_false_negatives(fpr, n_show=4):
    """Por implementação: scores dos defeitos vs limiar e os falsos negativos mais fracos, com o mapa."""
    gy = pd.read_csv(SESS / "scores.csv")
    my = metrics(gy, "max", fpr)
    gt = gy[(gy.split == "test") & (gy.y_true == 1)].sort_values("max")
    fn = gt[gt["max"] <= my["thr"]].head(n_show)
    heats = glass_heatmaps(SESS, fn.path)
    glass_items = [(read_image(Path(p)), h, s) for p, h, s in zip(fn.path, heats, fn["max"])]
    sc = pd.read_csv(SH_EXP / "source_scores.csv").rename(columns={"pred_score": "max"})
    man = pd.read_csv(SH_EXP / "map_manifest.csv").drop_duplicates("source_id").set_index("source_id")
    ms = metrics(sc, "max", fpr)
    st = sc[(sc.split == "test") & (sc.y_true == 1)].sort_values("max")
    sfn = st[st["max"] <= ms["thr"]].head(n_show)
    sh_items = [(read_image(Path(man.image_path[r.source_id])), np.load(man.map_path[r.source_id]), r["max"])
                for _, r in sfn.iterrows()]
    rows = [("Vidro (YOLO)", gt["max"].to_numpy(), my, glass_items), ("Haste (SAM)", st["max"].to_numpy(), ms, sh_items)]
    fig, axes = plt.subplots(2, 1 + n_show, figsize=(3 * (n_show + 1.6), 6.8),
                             gridspec_kw=dict(width_ratios=[1.9] + [1] * n_show), squeeze=False)
    for r, (name, all_scores, m, items) in enumerate(rows):
        ax = axes[r, 0]
        values = np.sort(all_scores)
        ax.bar(range(len(values)), values, color=[BAD if v <= m["thr"] else GOOD for v in values], width=0.85)
        ax.axhline(m["thr"], color="black", ls="--", lw=1.3)
        ax.set(title=f"{name}: {m['FN']}/{m['n_anomaly']} defeitos perdidos", xlabel="defeitos, do menor ao maior score",
               ylabel="score", xticks=[], ylim=(max(0, values.min() - 3), values.max() + 2))
        ax.title.set_fontsize(12)
        ax.text(0.98, 0.04, f"limiar {m['thr']:.1f}\nFP {m['FP']}/{m['n_good']}", transform=ax.transAxes, ha="right", fontsize=10)
        vmax = max(float(h.max()) for _, h, _ in items)
        for cell, (img, heat, score) in zip(axes[r, 1:], items):
            cell.imshow(img[..., ::-1])
            cell.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
            cell.set_title(f"score {score:.0f}", fontsize=10)
        style(axes[r, 1:])
        for cell in axes[r, 1 + len(items):]:
            cell.axis("off")
    fig.suptitle(f"Onde o modelo falha: falsos negativos com FPR alvo de {fpr:.0%} (vermelho = perdido)", fontsize=14)
    fig.tight_layout()
    return fig, dict(glass=my, haste=ms)


GL_LAYER = Path("C:/dados/gl_layer")  # saída de scripts/glass_heatmap_layer.py (mapas e map_evaluation do vidro)


def test_scores():
    """Score do PatchCore por imagem de teste, nas duas implementações: {nome: (df com y_true, max, split)}."""
    glass = pd.read_csv(SESS / "scores.csv")
    haste = pd.read_csv(SH_EXP / "source_scores.csv").rename(columns={"pred_score": "max"})
    return {"Vidro (YOLO)": glass, "Haste (SAM)": haste}


def sweep_table(df, step=1.0):
    test = df[df.split == "test"]
    y, x = test.y_true.to_numpy(), test["max"].to_numpy()
    rows = []
    for thr in np.arange(np.floor(x.min()), np.ceil(x.max()) + step, step):
        a = x > thr
        tp, fp = int(a[y == 1].sum()), int(a[y == 0].sum())
        rows.append(dict(limiar=thr, TP=tp, FN=int((y == 1).sum()) - tp, FP=fp, TN=int((y == 0).sum()) - fp,
                         recall=tp / (y == 1).sum(), FPR=fp / (y == 0).sum(), precisao=tp / max(1, tp + fp)))
    return pd.DataFrame(rows)


def fig_threshold_sweep(fpr):
    """Varredura do limiar: quantas anomalias reais são achadas (TP/FN) e quantos normais viram alarme (FP)."""
    systems = test_scores()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")
    tables = {}
    for c, (name, df) in enumerate(systems.items()):
        table, m = sweep_table(df), metrics(df, "max", fpr)
        tables[name] = table
        ax = axes[0, c]
        ax.plot(table.limiar, table.recall, color=BAD, lw=2, label="recall (defeitos achados)")
        ax.plot(table.limiar, table.FPR, color=GOOD, lw=2, label="falsos alarmes (normais marcados)")
        ax.plot(table.limiar, table.precisao, color="gray", lw=1.5, ls=":", label="precisão")
        ax.axvline(m["thr"], color="black", ls="--")
        ax.set(title=f"{name}: limiar {m['thr']:.1f} (FPR alvo {fpr:.0%}) → recall {m['recall']:.0%}, FP {m['FPR']:.0%}",
               ylabel="taxa", ylim=(0, 1.02))
        ax.title.set_fontsize(11)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax = axes[1, c]
        ax.plot(table.limiar, table.TP, color=BAD, lw=2, label=f"anomalias reais detectadas (TP, de {m['n_anomaly']})")
        ax.plot(table.limiar, table.FN, color=BAD, lw=2, ls="--", label="anomalias perdidas (FN)")
        ax.plot(table.limiar, table.FP, color=GOOD, lw=2, label=f"normais com alarme (FP, de {m['n_good']})")
        ax.axvline(m["thr"], color="black", ls="--")
        ax.set(xlabel="limiar do score do PatchCore (alarme se score > limiar)", ylabel="imagens de teste")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Varredura do limiar: o que o modelo marca como anomalia × o que é anomalia real", fontsize=14)
    fig.tight_layout()
    return fig, tables


def fig_heatmap_why(fpr):
    """Por que a camada de heatmaps não ajudou: o score do mapa repete o do PatchCore e normais também têm vermelho."""
    from scipy.stats import spearmanr

    sources = {"Vidro (YOLO)": (SESS / "scores.csv", GL_LAYER, True), "Haste (SAM)": (SH_EXP / "source_scores.csv", SH_EXP, False)}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))
    stats = {}
    for r, (name, (score_file, exp, glass)) in enumerate(sources.items()):
        sc = pd.read_csv(score_file)
        sc = (sc.rename(columns={"max": "pc", "name": "source_id"}) if glass else sc.rename(columns={"pred_score": "pc"}))
        sc = sc[sc.split == "test"][["source_id", "y_true", "pc"]]
        maps = pd.read_csv(exp / "map_evaluation/map_metrics.csv")
        absolute = maps[(maps.method == "absolute") & (maps.split == "test")].groupby("source_id").score.max().rename("heat")
        contrast = maps[(maps.method == "contrast") & (maps.split == "test")].groupby("source_id").score.max().rename("contrast")
        frame = sc.merge(absolute, left_on="source_id", right_index=True).merge(contrast, left_on="source_id", right_index=True)
        rho = spearmanr(frame.pc, frame.heat)[0]
        stats[name] = dict(spearman_pc_vs_heat=float(rho), auroc_pc=float(roc_auc_score(frame.y_true, frame.pc)),
                           auroc_heat=float(roc_auc_score(frame.y_true, frame.heat)),
                           auroc_contrast=float(roc_auc_score(frame.y_true, frame.contrast)))
        ax = axes[r, 0]
        for y, color, label in ((0, GOOD, "normal"), (1, BAD, "defeito")):
            part = frame[frame.y_true == y]
            ax.scatter(part.pc, part.heat, s=14, alpha=0.6, color=color, label=label)
        ax.set(xlabel="score do PatchCore", ylabel="score do heatmap",
               title=f"{name}: correlação {rho:.2f}")
        ax.title.set_fontsize(11)
        ax.legend(frameon=False, fontsize=9)
        man = pd.read_csv(exp / "map_manifest.csv").drop_duplicates("source_id").set_index("source_id")
        neg, pos = frame[frame.y_true == 0], frame[frame.y_true == 1]
        cases = [("normal com maior score", neg.loc[neg.pc.idxmax()]), ("defeito achado (maior score)", pos.loc[pos.pc.idxmax()]),
                 ("defeito perdido (menor score)", pos.loc[pos.pc.idxmin()])]
        heats = [np.load(man.map_path[row.source_id]) for _, row in cases]
        vmax = max(float(h.max()) for h in heats)
        for c, ((title, row), heat) in enumerate(zip(cases, heats), start=1):
            cell = axes[r, c]
            cell.imshow(read_image(Path(man.image_path[row.source_id]))[..., ::-1])
            cell.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
            cell.set_title(f"{title}\nscore {row.pc:.0f}", fontsize=10)
            style(cell)
    fig.suptitle("Por que os heatmaps não ajudam: o score do mapa repete o do PatchCore e normais também ficam vermelhos", fontsize=13)
    fig.tight_layout()
    return fig, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpr", type=float, default=0.4)
    parser.add_argument("--out", default="docs/img")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, make in (("datasets", lambda: (fig_datasets(), None)), ("yolo_vs_sam", lambda: fig_yolo_vs_sam(args.fpr)),
                       ("falsos_negativos", lambda: fig_false_negatives(args.fpr)),
                       ("varredura_limiar", lambda: fig_threshold_sweep(args.fpr)),
                       ("heatmaps_porque", lambda: fig_heatmap_why(args.fpr))):
        fig, stat = make()
        fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        if name == "varredura_limiar":
            pd.concat({k: v for k, v in stat.items()}, names=["sistema"]).reset_index(level=0).to_csv(
                out.parent / "logs" / "varredura_limiar.csv", index=False)
            stat = None
        stats[name] = stat
        print(out / f"{name}.png")
    (out.parent / "logs" / "comparacao_metricas.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
