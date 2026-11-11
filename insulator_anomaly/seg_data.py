"""Dataset de segmentação a partir das máscaras revisadas, dividido por rajada."""

from pathlib import Path
import csv
import hashlib
import random
import shutil

import cv2
import numpy as np

from .seg_annotate import read_mask, save_png

ACCEPTED = {"aceita", "corrigida", "refeita"}


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def mask_to_yolo_polygons(mask, epsilon=0.002, min_area=0.0005):
    """Linhas 'classe x1 y1 ...' (coordenadas normalizadas) dos contornos externos."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area * mask.size:
            continue
        points = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(points) >= 3:
            coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in points)
            lines.append(f"0 {coords}")
    return lines


def finalize(review_csv, pool_csv, dataset, output, val_fraction=0.2, seed=42):
    """Copia imagens e máscaras aceitas para output/{train,val}/{images,masks,labels}.

    Só entram status aceita/corrigida/refeita; qualquer outro interrompe. A divisão é por
    rajada (burst_id do pool.csv), para quadros quase idênticos não cruzarem train e val.
    O hash da imagem de origem precisa bater com o do review.csv.
    """
    rows = read_rows(review_csv)
    pending = [r["path"] for r in rows if r["status"] not in ACCEPTED]
    if pending:
        raise ValueError(f"{len(pending)} máscaras sem revisão concluída, ex.: {pending[:3]}")
    names = [Path(r["path"]).name for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Nomes de arquivo repetidos.")
    output = Path(output)
    output.mkdir(parents=True)
    burst = {r["path"]: r["burst_id"] for r in read_rows(pool_csv)}
    bursts = sorted({burst[r["path"]] for r in rows})
    random.Random(seed).shuffle(bursts)
    val_bursts = set(bursts[: max(1, round(len(bursts) * val_fraction))])
    split_rows = []
    for row in rows:
        split = "val" if burst[row["path"]] in val_bursts else "train"
        name = Path(row["path"]).name
        source = Path(dataset) / row["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Imagem alterada desde a geração da máscara: {row['path']}")
        target = output / split / "images" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mask = read_mask(row["mask_path"])
        save_png(output / split / "masks" / (Path(name).stem + ".png"), mask.astype(np.uint8) * 255)
        label = output / split / "labels" / (Path(name).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(mask_to_yolo_polygons(mask)), encoding="utf-8")
        split_rows.append(dict(path=row["path"], name=name, burst_id=burst[row["path"]], split=split,
                               status=row["status"], image_sha256=row["image_sha256"],
                               mask_sha256=row["mask_sha256"]))
    with open(output / "split.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: object\n",
        encoding="utf-8")
    return split_rows
