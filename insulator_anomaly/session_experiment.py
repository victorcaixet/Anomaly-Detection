"""Divisão por rajada e grupos de sessões de captura (um PatchCore por grupo; driver: scripts/run_sessions.py).

Rajada = fotos consecutivas (DJI) da mesma data; nunca cruza treino/val/teste.
O limiar de cada modelo é um quantil dos scores da validação normal.
"""

from pathlib import Path
import hashlib
import math
import os
import re
import shutil

import cv2
import numpy as np
import pandas as pd

from .evaluation import val_threshold

NAME = re.compile(r"Fotos_(\d\d-\d\d-\d{4})_DJI_(\d+)")
FEATURES = ["r", "g", "b", "sat", "val", "std", "sharp"]


def build_manifest(source_root, mode="gray", burst_gap=5, val=0.15, test=0.20, seed=42):
    """Todas as normais (train+test originais) e anomalias; split por rajada.

    O sorteio depende só de (seed, data, 1ª foto da rajada): idêntico para
    qualquer agrupamento, então modelos com grupos diferentes compartilham o teste.
    """
    root, rows = Path(source_root) / mode, []
    for folder, label in (("train/good", 0), ("test/good", 0), ("test/anomaly", 1)):
        for path in sorted((root / folder).glob("*.png")):
            date, dji = NAME.search(path.name).groups()
            rows.append(dict(path=str(path), date=date, dji=int(dji), y_true=label))
    frame = pd.DataFrame(rows).sort_values(["date", "dji"], ignore_index=True)
    new = frame.groupby("date").dji.diff().fillna(math.inf) > burst_gap
    frame["burst"] = new.cumsum()
    first = frame.groupby("burst").dji.transform("min")

    def draw(date, start):
        return int(hashlib.sha256(f"{seed}:{date}:{start}".encode()).hexdigest()[:8], 16) / 2**32

    u = pd.Series([draw(d, s) for d, s in zip(frame.date, first)])
    split = np.where(u < test, "test", np.where(u < test + val, "val", "train"))
    frame["split"] = np.where(frame.y_true == 1, "test", split)
    return frame


def cluster_dates(manifest, k=3, min_good=150, seed=42):
    """Agrupa datas por estatísticas de cor/nitidez; funde grupos pequenos demais."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    rows = []
    for path, date in zip(manifest.path, manifest.date):
        p = Path(path)  # mesma imagem no modo "original": o fundo revela as condições da cena
        image = cv2.imread(str(p.parents[2].with_name("original") / p.parents[1].name / p.parent.name / p.name))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows.append(dict(date=date, r=image[..., 2].mean(), g=image[..., 1].mean(), b=image[..., 0].mean(),
                         sat=hsv[..., 1].mean(), val=hsv[..., 2].mean(), std=gray.std(),
                         sharp=cv2.Laplacian(gray, cv2.CV_64F).var()))
    per_date = pd.DataFrame(rows).groupby("date")[FEATURES].median()
    good = manifest[manifest.y_true == 0].groupby("date").size().reindex(per_date.index, fill_value=0)
    x = StandardScaler().fit_transform(per_date)
    label = KMeans(min(k, len(per_date)), n_init=20, random_state=seed).fit_predict(x)
    # ponytail: funde o menor grupo no vizinho mais próximo até todos terem min_good normais
    while len(set(label)) > 1:
        size = pd.Series(good.to_numpy()).groupby(label).sum()
        small = size.idxmin()
        if size[small] >= min_good:
            break
        centers = {c: x[label == c].mean(axis=0) for c in set(label)}
        nearest = min((c for c in centers if c != small), key=lambda c: np.linalg.norm(centers[c] - centers[small]))
        label[label == small] = nearest
    order = pd.Series(good.to_numpy()).groupby(label).sum().sort_values(ascending=False).index
    names = {c: f"g{i}" for i, c in enumerate(order)}
    return {d: names[c] for d, c in zip(per_date.index, label)}


def prepare_groups(manifest, groups, root, max_train=350):
    """Hard link (ou cópia) das imagens em <root>/glass-preprocessed/<grupo>/<split>/<classe>.

    O treino é limitado a `max_train` fotos espaçadas por data/número: o banco de
    patches do PatchCore (~1k patches x 1536 dim por imagem) precisa caber na GPU.
    """
    root = Path(root)
    for name in sorted(set(groups.values())):
        part = manifest[manifest.date.map(groups) == name]
        for split, label in (("train", 0), ("val", 0), ("test", 0), ("test", 1)):
            if not ((part.split == split) & (part.y_true == label)).any():
                raise ValueError(f"Grupo {name} sem imagens em {split}/{'anomaly' if label else 'good'}.")
        target_root = root / "glass-preprocessed" / name
        if target_root.exists():
            raise FileExistsError(f"Use um destino novo: {target_root}")
        keep = part[(part.split != "train") | (part.y_true == 1)]
        train = part[(part.split == "train") & (part.y_true == 0)].sort_values(["date", "dji"])
        if len(train) > max_train:
            train = train.iloc[np.linspace(0, len(train) - 1, max_train).round().astype(int)]
        for row in pd.concat([keep, train]).itertuples():
            target = target_root / row.split / ("anomaly" if row.y_true else "good") / Path(row.path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(row.path, target)
            except OSError:
                shutil.copy2(row.path, target)


def evaluate(scores, reference_groups, normal_fpr=0.05):
    """Limiar por modelo = quantil (1-fpr) da validação normal; métricas por grupo de referência.

    AUROC/AP usam score/limiar, comparável entre modelos de escalas diferentes.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for kind in ("max", "top"):
        val = scores[scores.split == "val"]
        threshold = val.groupby("model_group")[kind].agg(lambda v: val_threshold(v, normal_fpr))
        test = scores[scores.split == "test"].copy()
        test["ratio"] = test[kind] / test.model_group.map(threshold)
        test["alarm"] = test.ratio > 1
        test["ref_group"] = test.date.map(reference_groups)
        for ref, part in [("ALL", test)] + list(test.groupby("ref_group")):
            if part.y_true.nunique() < 2:
                continue
            positive = part.y_true == 1
            rows.append(dict(score=kind, group=ref, n_good=int((~positive).sum()), n_anomaly=int(positive.sum()),
                             AUROC=roc_auc_score(part.y_true, part.ratio),
                             AP=average_precision_score(part.y_true, part.ratio),
                             recall=float(part.alarm[positive].mean()), FPR=float(part.alarm[~positive].mean()),
                             precision=float(part.alarm[positive].sum() / max(1, part.alarm.sum()))))
    return pd.DataFrame(rows)
