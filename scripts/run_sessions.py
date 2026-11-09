"""Um PatchCore por grupo de sessões de captura vs um global; salva scores e a tabela comparativa.

Uso: python scripts/run_sessions.py <pasta com original/ e gray/> <destino novo> [--k 3] [--min-good 60]
Saída: <destino>/sessions_comparison.csv (coluna `sistema`: global | por_grupo), entrada de make_report.py --sessions.
"""

from pathlib import Path
import argparse
import hashlib
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.config import ProjectConfig
from insulator_anomaly.session_experiment import build_manifest, cluster_dates, evaluate, prepare_groups


def predict_scores(engine, model, folder, top=0.01):
    """Score por imagem: máximo do mapa e média do top `top` dos pixels."""
    from anomalib.data import PredictDataset
    from torch.utils.data import DataLoader

    dataset = PredictDataset(path=folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    rows = []
    for batch in engine.predict(model=model, dataloaders=loader, return_predictions=True):
        paths = [batch.image_path] if isinstance(batch.image_path, (str, Path)) else batch.image_path
        maps = batch.anomaly_map.detach().cpu().float().numpy().reshape(len(paths), -1)
        k = max(1, math.ceil(maps.shape[1] * top))
        for path, flat in zip(paths, maps, strict=True):
            if not np.isfinite(flat).all():
                raise ValueError(f"Mapa inválido: {path}")
            rows.append(dict(path=str(Path(path).resolve()), max=float(flat.max()),
                             top=float(np.partition(flat, -k)[-k:].mean())))
    return pd.DataFrame(rows)


def run_session_experiment(manifest, groups, root, accelerator="gpu", top=0.01, seed=42):
    """Treina um PatchCore por grupo e devolve scores de val/teste por imagem."""
    import gc
    import torch
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything
    from .training import train_one_patchcore

    root = Path(root)
    names = sorted(set(groups.values()))
    config = ProjectConfig(root, background_modes=tuple(names), random_seed=seed, accelerator=accelerator)
    out = []
    for name in names:
        files = sorted(p.name for p in (root / "glass-preprocessed" / name / "train" / "good").iterdir())
        signature = hashlib.sha256("".join(files).encode()).hexdigest()
        seed_everything(seed, workers=True)
        report = train_one_patchcore(config, name, post_processor=False, preprocessing_id=signature)
        model = Patchcore.load_from_checkpoint(report["checkpoint"], map_location="cpu", weights_only=False)
        engine = Engine(default_root_dir=root / "inference" / name, logger=False, accelerator=accelerator,
                        devices=1, enable_progress_bar=False, enable_model_summary=False)
        try:
            base = root / "glass-preprocessed" / name
            for split, label in (("val", "good"), ("test", "good"), ("test", "anomaly")):
                scores = predict_scores(engine, model, base / split / label, top)
                scores["split"], scores["model_group"] = split, name
                out.append(scores)
        finally:
            del model, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    scores = pd.concat(out, ignore_index=True)
    # as imagens do experimento são links/cópias: o nome do arquivo é a chave única
    scores["name"] = scores.path.map(lambda p: Path(p).name)
    info = manifest.assign(name=manifest.path.map(lambda p: Path(p).name))[["name", "date", "y_true"]]
    return scores.merge(info, on="name", validate="one_to_one")


def main(source, out, k, min_good):
    manifest = build_manifest(source, "gray")
    by_group = cluster_dates(manifest, k=k, min_good=min_good)
    tables = []
    for system, groups in (("global", {d: "global" for d in manifest.date.unique()}), ("por_grupo", by_group)):
        root = Path(out) / system
        prepare_groups(manifest, groups, root)
        scores = run_session_experiment(manifest, groups, root)
        scores.to_csv(root / "scores.csv", index=False)
        table = evaluate(scores, by_group)
        table.insert(0, "sistema", system)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(Path(out) / "sessions_comparison.csv", index=False)
    print(result.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-good", type=int, default=60)
    args = parser.parse_args()
    main(args.source, args.out, args.k, args.min_good)
