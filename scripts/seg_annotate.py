"""Anotação assistida do pool de segmentação.

  masks <dataset> <pool.csv> <boxes.json> <saida> [--points points.json] [--auto-point 'x,y;x,y']
      SAM gera uma máscara por imagem. Caixas: manuais (boxes.json) e, nas demais, uma caixa
      que o próprio SAM obtém de pontos positivos (--auto-point off desliga). Cada máscara é verificada
      (forma, faixa das manuais, estabilidade sob variação da caixa). Escreve review.csv,
      suspeitas.json, boxes.json, masks/ e overlays/{suspeita,auditoria,restante}/.
  report <review.csv>
      Após revisar (status aceita/corrigida/refeita): taxa de máscaras ruins entre suspeitas, entre
      não suspeitas e na auditoria, por motivo. Calibra as verificações.
  detector <dataset> <pool.csv> <boxes.json> <saida>
      Alternativa: treina um detector nas caixas manuais e propõe caixas (box_source.csv);
      passe-o a `masks` com --box-source.
Destinos precisam ser novos. Refine as suspeitas com scripts/annotate_points.py e repita `masks`.
Depois da revisão, marque status (aceita/corrigida/refeita) no review.csv e rode seg_train.py finalize.
"""

from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.seg_annotate import (
    STABILITY_MIN, box_iou, check_report, generate_masks, propose_boxes, write_detection_dataset,
)


def read_pool(path):
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_detector(args):
    from ultralytics import YOLO

    pool = read_pool(args.pool)
    manual = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    sam50 = [row["path"] for row in pool if row["subset"] == "sam50"]
    missing = [path for path in sam50 if path not in manual]
    if missing:
        raise SystemExit(f"{len(missing)} imagens sem caixa manual, ex.: {missing[:3]}")
    output = Path(args.output)
    output.mkdir(parents=True)
    boxes = {path: manual[path] for path in sam50}
    yaml, val = write_detection_dataset(args.dataset, boxes, output / "data", seed=args.seed)
    model = YOLO(args.weights)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=8, workers=0,
                device=args.device, project=str((output / "detector").resolve()), name="run",
                seed=args.seed, patience=30, verbose=False)
    best = YOLO(str(output / "detector" / "run" / "weights" / "best.pt"))
    proposals = propose_boxes(best, args.dataset, [row["path"] for row in pool], args.conf, args.device)
    ious = [box_iou(proposals[p][0], boxes[p]) for p in val if proposals[p]]
    print(f"IoU médio das caixas na validação: {sum(ious) / max(len(ious), 1):.3f} "
          f"({len(ious)}/{len(val)} detectadas)")
    all_boxes, rows = dict(boxes), []
    for row in pool:
        path = row["path"]
        if path in boxes:
            rows.append(dict(path=path, subset=row["subset"], box_source="manual", box_conf=""))
        elif proposals[path]:
            all_boxes[path] = proposals[path][0]
            rows.append(dict(path=path, subset=row["subset"], box_source="detector",
                             box_conf=f"{proposals[path][1]:.4f}"))
        else:
            rows.append(dict(path=path, subset=row["subset"], box_source="nenhuma", box_conf=""))
    (output / "boxes_all.json").write_text(json.dumps(all_boxes, indent=1), encoding="utf-8")
    (output / "sem_caixa.json").write_text(
        json.dumps([r["path"] for r in rows if r["box_source"] == "nenhuma"], indent=1),
        encoding="utf-8")
    with open(output / "box_source.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["path", "subset", "box_source", "box_conf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Caixas: {len(all_boxes)}/{len(pool)}; sem caixa: {len(pool) - len(all_boxes)}.")


def run_masks(args):
    from ultralytics import SAM

    boxes = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    points = json.loads(Path(args.points).read_text(encoding="utf-8")) if args.points else None
    if args.box_source:
        rows = read_pool(args.box_source)
    else:
        rows = [dict(path=r["path"], subset=r["subset"], box_conf="",
                     box_source="manual" if r["path"] in boxes else "nenhuma")
                for r in read_pool(args.pool)]
    auto = None if args.auto_point == "off" else tuple(
        tuple(float(v) for v in pair.split(",")) for pair in args.auto_point.split(";"))
    table = generate_masks(SAM(args.sam), args.dataset, rows, boxes, args.output, points,
                           args.audit, args.device, args.seed, auto, args.stability_min)
    done = [e for e in table if e["status"] == "pendente"]
    print(f"Máscaras: {len(done)}/{len(table)}; suspeitas: {sum(e['suspect'] for e in done)}; "
          f"auditoria: {sum(e['audit'] for e in done)}; caixa do SAM: "
          f"{sum(e['box_source'] == 'sam_ponto' for e in done)}.")


def run_report(args):
    print(json.dumps(check_report(read_pool(args.review)), indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    masks = sub.add_parser("masks")
    for name in ("dataset", "pool", "boxes", "output"):
        masks.add_argument(name)
    masks.add_argument("--box-source")
    masks.add_argument("--points")
    masks.add_argument("--auto-point", default="0.5,0.2;0.5,0.5;0.5,0.8")
    masks.add_argument("--stability-min", type=float, default=STABILITY_MIN)
    masks.add_argument("--sam", default="sam2.1_b.pt")
    masks.add_argument("--audit", type=float, default=0.15)
    masks.add_argument("--device", default=0)
    masks.add_argument("--seed", type=int, default=42)
    masks.set_defaults(run=run_masks)
    report = sub.add_parser("report")
    report.add_argument("review")
    report.set_defaults(run=run_report)
    detector = sub.add_parser("detector")
    for name in ("dataset", "pool", "boxes", "output"):
        detector.add_argument(name)
    detector.add_argument("--weights", default="yolo11n.pt")
    detector.add_argument("--epochs", type=int, default=100)
    detector.add_argument("--imgsz", type=int, default=640)
    detector.add_argument("--conf", type=float, default=0.1)
    detector.add_argument("--device", default=0)
    detector.add_argument("--seed", type=int, default=42)
    detector.set_defaults(run=run_detector)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
