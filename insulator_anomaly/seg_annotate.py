"""Anotação assistida: caixas -> SAM -> máscaras verificadas, para revisão."""

from pathlib import Path
import csv
import hashlib
import json
import random
import shutil

import cv2
import numpy as np

from .preprocessing import read_image, sam_mask, suspect_mask

BOX_CONF_MIN = 0.5  # abaixo disso a caixa proposta entra como suspeita
STABILITY_MIN = 0.9  # IoU médio sob pequenas variações da caixa; limiar chutado
JITTER = 0.04
AUTO_POINTS = ((0.5, 0.2), (0.5, 0.5), (0.5, 0.8))  # cadeias verticais e centradas; 1 ponto cai numa peça só
RANGE_FEATURES = ("area_fraction", "holes_fraction", "bbox_fill", "log_aspect")


def xyxy_to_yolo(box, width, height):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def write_detection_dataset(root, boxes, destination, val_fraction=0.2, seed=42):
    """Dataset YOLO de uma classe; retorna (data.yaml, caminhos de validação)."""
    root, destination = Path(root), Path(destination)
    paths = sorted(boxes)
    if len({Path(p).name for p in paths}) != len(paths):
        raise ValueError("Nomes de arquivo repetidos: o dataset YOLO usa só o nome.")
    random.Random(seed).shuffle(paths)
    val = set(paths[: max(1, round(len(paths) * val_fraction))])
    for path in paths:
        split = "val" if path in val else "train"
        image_dir, label_dir = (destination / kind / split for kind in ("images", "labels"))
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / path, image_dir / Path(path).name)
        height, width = read_image(root / path).shape[:2]
        x0, y0, x1, y1 = boxes[path]  # a seleção do mouse pode passar da borda
        box = (max(x0, 0), max(y0, 0), min(x1, width), min(y1, height))
        values = " ".join(f"{v:.6f}" for v in xyxy_to_yolo(box, width, height))
        (label_dir / (Path(path).stem + ".txt")).write_text(f"0 {values}\n", encoding="utf-8")
    yaml = destination / "data.yaml"
    yaml.write_text(
        f"path: {destination.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: object\n", encoding="utf-8")
    return yaml, sorted(val)


def propose_boxes(model, root, paths, conf=0.1, device=0):
    """Maior confiança por imagem: {caminho: (caixa, conf) | None}."""
    result = {}
    for path in paths:
        prediction = model.predict(source=read_image(Path(root) / path), conf=conf,
                                   verbose=False, device=device)[0]
        if prediction.boxes is None or len(prediction.boxes) == 0:
            result[path] = None
            continue
        best = int(prediction.boxes.conf.argmax())
        result[path] = ([round(float(v), 1) for v in prediction.boxes.xyxy[best].tolist()],
                        float(prediction.boxes.conf[best]))
    return result


def pick_audit(paths, fraction, seed=42):
    """Amostra das máscaras não suspeitas, para estimar o erro das 'aceitas'."""
    paths = sorted(paths)
    return set(random.Random(seed).sample(paths, round(len(paths) * fraction)))


def overlay(image, mask, box, size=720):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = image.copy()
    cv2.drawContours(canvas, contours, -1, (0, 0, 255), 3)
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 128, 0), 2)
    scale = size / max(canvas.shape[:2])
    return cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, array)
    if not ok:
        raise OSError(f"Falha ao salvar: {path}")
    encoded.tofile(path)


def read_mask(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE) > 0


def mask_features(mask):
    """Medidas de forma da máscara; None se vazia."""
    binary = mask.astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ys, xs = np.nonzero(binary)
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    _, labels, hole_stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=4)
    outer = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = sum(hole_stats[i, cv2.CC_STAT_AREA] for i in range(1, len(hole_stats)) if i not in outer)
    return dict(area_fraction=area / binary.size, n_components=count - 1,
                main_fraction=stats[1:, cv2.CC_STAT_AREA].max() / area, holes_fraction=holes / area,
                bbox_fill=area / (width * height), log_aspect=float(np.log(width / height)))


def reference_ranges(features, margin=0.5):
    """Faixa p5-p95 das máscaras manuais, alargada em `margin` da largura da faixa."""
    ranges = {}
    for name in RANGE_FEATURES:
        low, high = np.percentile([f[name] for f in features], [5, 95])
        pad = (high - low) * margin
        ranges[name] = (float(low - pad), float(high + pad))
    return ranges


def range_reasons(features, ranges):
    return [f"{name} fora do padrão manual ({features[name]:.3f})"
            for name, (low, high) in ranges.items() if not low <= features[name] <= high]


def mask_iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def jittered_boxes(box, width, height, amount=JITTER):
    """Caixa maior, menor e deslocada nas duas diagonais, dentro da imagem."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * amount, (y1 - y0) * amount
    variants = [(x0 - dx, y0 - dy, x1 + dx, y1 + dy), (x0 + dx, y0 + dy, x1 - dx, y1 - dy),
                (x0 + dx, y0 + dy, x1 + dx, y1 + dy), (x0 - dx, y0 - dy, x1 - dx, y1 - dy)]
    return [[max(a, 0), max(b, 0), min(c, width), min(d, height)] for a, b, c, d in variants]


def stability(sam, image, box, mask, points=None, device=0):
    """IoU médio entre a máscara e as obtidas com a caixa perturbada (instável = suspeita)."""
    height, width = image.shape[:2]
    ious = []
    for variant in jittered_boxes(box, width, height):
        try:
            ious.append(mask_iou(mask, sam_mask(sam, image, device, variant, points)))
        except RuntimeError:
            ious.append(0.0)
    return float(np.mean(ious))


def auto_box(sam, image, device=0, points=AUTO_POINTS):
    """Caixa da máscara obtida com pontos positivos (frações x, y); None se o SAM não achar nada."""
    height, width = image.shape[:2]
    try:
        mask = sam_mask(sam, image, device, points=[[fx * width, fy * height, 1] for fx, fy in points])
    except RuntimeError:
        return None
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def generate_masks(sam, root, rows, boxes, output, points=None, audit=0.15, device=0, seed=42,
                   auto_point=AUTO_POINTS, stability_min=STABILITY_MIN):
    """rows: [{path, subset, box_source, box_conf}]. Destino novo; escreve review.csv.

    Sem caixa e com auto_point (lista de pontos x, y em fração), o SAM propõe a caixa (box_source 'sam_ponto').
    Verificações: heurísticas de suspect_mask, faixa das máscaras manuais e estabilidade.
    Sem caixa possível: sem máscara, status 'sem_caixa' (marcar com annotate_boxes e repetir).
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True)
    points = points or {}
    boxes = dict(boxes)
    table = []
    for row in rows:
        path = row["path"]
        entry = dict(row, mask_path="", suspect_reasons=[], suspect=False, audit=False,
                     status="sem_caixa", image_sha256="", mask_sha256="", box="", stability="")
        image = read_image(root / path)
        if path not in boxes and auto_point is not None:
            found = auto_box(sam, image, device, auto_point)
            if found:
                boxes[path] = found
                entry["box_source"] = "sam_ponto"
        if path in boxes:
            mask = sam_mask(sam, image, device, boxes[path], points.get(path))
            reasons = suspect_mask(mask, boxes[path])
            if row.get("box_conf") not in ("", None) and float(row["box_conf"]) < BOX_CONF_MIN:
                reasons.append(f"caixa incerta ({float(row['box_conf']):.2f})")
            stable = stability(sam, image, boxes[path], mask, points.get(path), device)
            if stable < stability_min:
                reasons.append(f"instável sob variação da caixa ({stable:.2f})")
            mask_path = output / "masks" / Path(path).with_suffix(".png")
            save_png(mask_path, mask.astype(np.uint8) * 255)
            entry.update(mask_path=str(mask_path), suspect_reasons=reasons, status="pendente",
                         box=json.dumps(boxes[path]), stability=f"{stable:.3f}",
                         _features=mask_features(mask) or {},
                         image_sha256=hashlib.sha256((root / path).read_bytes()).hexdigest(),
                         mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
        table.append(entry)
    manual = [e["_features"] for e in table if e["box_source"] == "manual" and e.get("_features")]
    ranges = reference_ranges(manual) if len(manual) >= 10 else None
    for entry in table:
        features = entry.pop("_features", None)
        if features is None:
            continue
        reasons = entry["suspect_reasons"] + (range_reasons(features, ranges) if ranges and features else [])
        entry.update({name: f"{value:.4f}" for name, value in features.items()},
                     suspect_reasons=";".join(reasons), suspect=bool(reasons))
    for entry in table:
        entry["suspect_reasons"] = ";".join(entry["suspect_reasons"]) if isinstance(
            entry["suspect_reasons"], list) else entry["suspect_reasons"]
    audited = pick_audit([e["path"] for e in table if e["status"] == "pendente" and not e["suspect"]],
                         audit, seed)
    for entry in table:
        entry["audit"] = entry["path"] in audited
        if entry["status"] == "pendente":
            group = "suspeita" if entry["suspect"] else "auditoria" if entry["audit"] else "restante"
            save_png(output / "overlays" / group / (Path(entry["path"]).stem + ".jpg"),
                     overlay(read_image(root / entry["path"]), read_mask(entry["mask_path"]),
                             boxes[entry["path"]]))
    fields = ["path", "subset", "box_source", "box_conf", "box", "mask_path", "suspect", "suspect_reasons",
              "audit", "stability", "area_fraction", "n_components", "main_fraction", "holes_fraction",
              "bbox_fill", "log_aspect", "status", "image_sha256", "mask_sha256"]
    with open(output / "review.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    (output / "suspeitas.json").write_text(  # formato de annotate_points.py: {caminho: [motivos]}
        json.dumps({e["path"]: e["suspect_reasons"].split(";") for e in table if e["suspect"]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    (output / "boxes.json").write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    (output / "reference_ranges.json").write_text(json.dumps(ranges, indent=1), encoding="utf-8")
    return table


def check_report(rows):
    """Após a revisão: quanto as verificações acertaram. Ruim = status corrigida/refeita."""
    done = [r for r in rows if r["status"] in ("aceita", "corrigida", "refeita")]

    def bad_rate(group):
        return sum(r["status"] != "aceita" for r in group) / len(group) if group else float("nan")

    flagged = [r for r in done if r["suspect"] == "True"]
    quiet = [r for r in done if r["suspect"] != "True"]
    report = dict(revisadas=len(done), suspeitas=len(flagged), ruins_entre_suspeitas=bad_rate(flagged),
                  ruins_entre_nao_suspeitas=bad_rate(quiet),
                  ruins_na_auditoria=bad_rate([r for r in quiet if r["audit"] == "True"]))
    reasons = {}
    for row in flagged:
        for reason in row["suspect_reasons"].split(";"):
            reasons.setdefault(reason.split(" (")[0], []).append(row)
    report["por_motivo"] = {name: dict(n=len(group), ruins=bad_rate(group)) for name, group in reasons.items()}
    return report
