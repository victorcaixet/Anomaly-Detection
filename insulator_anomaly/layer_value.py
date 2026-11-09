"""A camada de heatmaps compensa? Compara o score do PatchCore com o do mapa no mesmo teste.

Limiares fixados só em normais de validação, mesma taxa de falso alarme (fpr).
IC por bootstrap pareado sobre as imagens de teste.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .evaluation import val_threshold


def _rates(y, alarm):
    return dict(recall=float(alarm[y == 1].mean()), FPR=float(alarm[y == 0].mean()),
                TP=int(alarm[y == 1].sum()), FP=int(alarm[y == 0].sum()))


def layer_value(source_scores, map_scores, method="absolute", fpr=0.05, n_boot=2000, seed=0):
    """source_scores: source_id, split, y_true, pred_score (val/test).
    map_scores: source_id, split, method, score (reference/calibration/test).
    Retorna dict com AUROC, delta pareado (mapa - PatchCore), IC95 e alarmes
    do PatchCore sozinho, OU e E com o mapa.
    """
    maps = map_scores[map_scores.method == method]
    maps = maps.groupby(["source_id", "split"], as_index=False).score.max()
    val = source_scores[source_scores.split == "val"]
    test = source_scores[source_scores.split == "test"].merge(
        maps[maps.split == "test"][["source_id", "score"]], on="source_id", validate="one_to_one")
    map_val = maps[maps.split.isin(["reference", "calibration"])].score
    if val.empty or map_val.empty or test.y_true.nunique() < 2:
        raise ValueError("Exige validação normal e teste com as duas classes.")
    y, raw, heat = test.y_true.to_numpy(), test.pred_score.to_numpy(), test.score.to_numpy()
    fused = rankdata(raw) + rankdata(heat)
    alarm_raw = raw > val_threshold(val.pred_score, fpr)
    alarm_map = heat > val_threshold(map_val, fpr)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            deltas.append(roc_auc_score(y[i], heat[i]) - roc_auc_score(y[i], raw[i]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return dict(
        method=method, n_good=int((y == 0).sum()), n_anomaly=int((y == 1).sum()),
        AUROC_patchcore=roc_auc_score(y, raw), AUROC_map=roc_auc_score(y, heat),
        AUROC_fusion=roc_auc_score(y, fused),
        delta_AUROC=roc_auc_score(y, heat) - roc_auc_score(y, raw), delta_IC95=(float(low), float(high)),
        patchcore=_rates(y, alarm_raw), map=_rates(y, alarm_map),
        OR=_rates(y, alarm_raw | alarm_map), AND=_rates(y, alarm_raw & alarm_map),
    )


def demo():
    rng = np.random.default_rng(1)
    n = 200
    y = np.r_[np.zeros(n), np.ones(60)]
    ids = [f"t{i}" for i in range(len(y))]
    raw = rng.normal(y, 1.0)
    heat = raw + rng.normal(0, 0.3, len(y))
    val_ids = [f"v{i}" for i in range(100)]
    source = pd.concat([
        pd.DataFrame(dict(source_id=ids, split="test", y_true=y.astype(int), pred_score=raw)),
        pd.DataFrame(dict(source_id=val_ids, split="val", y_true=0, pred_score=rng.normal(0, 1, 100)))])
    maps = pd.concat([
        pd.DataFrame(dict(source_id=ids, split="test", method="absolute", score=heat)),
        pd.DataFrame(dict(source_id=val_ids, split="reference", method="absolute", score=rng.normal(0, 1, 100)))])
    r = layer_value(source, maps, n_boot=200)
    assert r["n_anomaly"] == 60 and abs(r["delta_AUROC"]) < 0.05
    assert r["delta_IC95"][0] <= r["delta_AUROC"] <= r["delta_IC95"][1]
    assert r["OR"]["recall"] >= r["patchcore"]["recall"] >= r["AND"]["recall"]
    assert r["OR"]["FPR"] >= r["patchcore"]["FPR"] >= r["AND"]["FPR"]


if __name__ == "__main__":
    demo()
    print("ok")
