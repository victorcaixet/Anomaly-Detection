"""Figuras do README (docs/img) a partir de um experimento com SAM + camada de heatmaps.

python scripts/make_showcase.py EXPERIMENTO PROJETO [--out docs/img]
EXPERIMENTO: pasta de run_crop_experiment com map_evaluation/; PROJETO: pasta com boxes_all.json.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.showcase import make_showcase

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("project")
parser.add_argument("--out", default="docs/img")
args = parser.parse_args()
for path in make_showcase(args.experiment, args.project, args.out):
    print(path)
