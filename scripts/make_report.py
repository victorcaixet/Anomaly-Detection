"""Gera as figuras de apresentação a partir dos resultados salvos de evaluate_all.

python scripts/make_report.py PROJETO [--persistent DRIVE] [--sessions comparacao.csv]
"""

from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.config import ProjectConfig
from insulator_anomaly.evaluation import predict_checked_heatmaps, resolve_evaluation_checkpoint
from insulator_anomaly.figures import load_results, make_report, select_cases

parser = argparse.ArgumentParser()
parser.add_argument("project")
parser.add_argument("--persistent")
parser.add_argument("--mode", default="gray")
parser.add_argument("--out")
parser.add_argument("--sessions", help="CSV de session_experiment.evaluate com a coluna 'sistema'")
parser.add_argument("--no-gallery", action="store_true", help="não carrega o modelo")
args = parser.parse_args()

config = ProjectConfig(args.project, args.persistent, accelerator="gpu")
results = load_results(config.results_root)
cases = maps = None
if not args.no_gallery:
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    frame = results[args.mode][0]
    cases = select_cases(frame)
    model = Patchcore.load_from_checkpoint(str(resolve_evaluation_checkpoint(config, args.mode)),
                                           map_location="cpu", weights_only=False)
    engine = Engine(accelerator=config.accelerator, devices=1, logger=False, enable_progress_bar=False,
                    enable_model_summary=False, default_root_dir=config.results_root / args.mode)
    arrays = predict_checked_heatmaps(engine, model, list(cases.itertuples(index=False)))
    maps = dict(zip(cases.image_path, arrays))
sessions = pd.read_csv(args.sessions) if args.sessions else None
out = Path(args.out) if args.out else config.results_root / "apresentacao"
for path in make_report(results, out, args.mode, cases, maps, sessions):
    print(path)
