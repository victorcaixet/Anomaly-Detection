"""Roda o SAM nas caixas e lista as máscaras suspeitas (heurística em preprocessing.suspect_mask).

Uso: python scripts/flag_masks.py <pasta com train/ e test/> <boxes.json> <suspeitas.json> [sam2.1_b.pt]
Saída: {caminho relativo: [motivos]}; serve de entrada para annotate_points.py.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.preprocessing import read_image, sam_mask, suspect_mask


def main(root, boxes, out, model="sam2.1_b.pt"):
    import ultralytics

    root, sam = Path(root).resolve(), ultralytics.SAM(model)
    boxes = json.loads(Path(boxes).read_text(encoding="utf-8"))
    flagged = {}
    for number, (relative, box) in enumerate(boxes.items(), 1):
        try:
            reasons = suspect_mask(sam_mask(sam, read_image(root / relative), box=box), box)
        except RuntimeError as error:
            reasons = [str(error)]
        if reasons:
            flagged[relative] = reasons
            print(f"{number}/{len(boxes)} {relative}: {'; '.join(reasons)}", flush=True)
    Path(out).write_text(json.dumps(flagged, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(flagged)}/{len(boxes)} suspeitas em {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
