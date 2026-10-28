"""Camada de heatmaps no vidro (YOLO): mapas de val+test do modelo global e layer_value com FPR alvo 0.4."""
import sys, hashlib
sys.path.insert(0, r"C:\Users\Decktop3\Documents\GitHub\Anomaly-Detection"); sys.path.insert(0, r"C:\Users\Decktop3\Documents\GitHub\Anomaly-Detection\scripts")
import numpy as np, pandas as pd
from pathlib import Path
from make_comparison import glass_heatmaps, SESS
from insulator_anomaly.map_postprocessing import MapSettings, evaluate_map_methods
from insulator_anomaly.layer_value import layer_value
FPR = 0.4
out = Path("C:/dados/gl_layer"); (out/"maps").mkdir(parents=True, exist_ok=True)
from insulator_anomaly.session_experiment import build_manifest
bm = build_manifest("C:/dados/anomalib_glass/glass-preprocessed", "gray")
burst = dict(zip(bm.path.map(lambda p: Path(p).name), bm.burst))  # grupo = rajada: nunca cruza splits
sc = pd.read_csv(SESS/"scores.csv"); sc = sc[sc.split.isin(["val", "test"])].reset_index(drop=True)
rows = [(r.path, r.name, r.split, r.y_true, (f"b{burst[r.name]}" if r.y_true == 0 else "a_" + r.name), r.max) for r in sc.itertuples()]
heats = glass_heatmaps(SESS, [r[0] for r in rows])
man = []
for (path, name, split, y, date, mx), h in zip(rows, heats):
    mp = out/"maps"/(hashlib.sha256(path.encode()).hexdigest()[:20] + ".npy"); np.save(mp, h)
    assert abs(float(h.max()) - mx) < 1e-2 * max(1, mx), (name, h.max(), mx)
    man.append(dict(mode="gray", split=split, source_id=name, group_id=date, y_true=int(y), image_path=path, map_path=str(mp)))
man = pd.DataFrame(man); man.to_csv(out/"map_manifest.csv", index=False)
comp, table = evaluate_map_methods(man, out/"map_evaluation", settings=MapSettings(alpha=FPR), show_first=0)
pd.set_option("display.width", 250)
print("COMP"); print(comp[["method","unit","n","AP","AUROC","TN","FP","FN","TP","recall","FPR"]].to_string())
src = sc.rename(columns={"max": "pred_score", "name": "source_id"})[["source_id", "split", "y_true", "pred_score"]]
maps = pd.read_csv(out/"map_evaluation/map_metrics.csv")
for m in ("absolute", "mebin", "contrast"):
    r = layer_value(src, maps, method=m, fpr=FPR)
    print(m, f"AUROC pc={r['AUROC_patchcore']:.3f} map={r['AUROC_map']:.3f} fus={r['AUROC_fusion']:.3f} d={r['delta_AUROC']:+.3f} IC={tuple(round(x,3) for x in r['delta_IC95'])}")
    for k in ("patchcore", "map", "OR", "AND"): print("   ", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in r[k].items()})
