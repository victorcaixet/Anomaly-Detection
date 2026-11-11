"""Treino e avaliação de YOLO-seg, U-Net e SegFormer sobre o mesmo dataset (seg_data.finalize)."""

from pathlib import Path
from time import perf_counter
import csv
import gc
import json
import random
import shutil
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import read_image
from .seg_annotate import read_mask

SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
KINDS = ("yolo", "unet", "segformer")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pairs(split_dir):
    """[(imagem, máscara)] de split_dir/images e split_dir/masks, pelo nome."""
    split_dir = Path(split_dir)
    images = sorted((split_dir / "images").iterdir())
    pairs = [(image, split_dir / "masks" / (image.stem + ".png")) for image in images]
    missing = [str(m) for _, m in pairs if not m.exists()]
    if missing:
        raise FileNotFoundError(f"Máscaras ausentes, ex.: {missing[:3]}")
    return pairs


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, augment, size=SIZE, seed=42):
        self.pairs = list_pairs(split_dir)
        self.augment, self.size = augment, size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def augmented(self, image, mask):
        """Espelho horizontal, recorte com escala 0,6-1 e brilho/contraste."""
        if self.rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        height, width = mask.shape
        scale = self.rng.uniform(0.6, 1.0)
        crop_h, crop_w = int(height * scale), int(width * scale)
        top = self.rng.integers(0, height - crop_h + 1)
        left = self.rng.integers(0, width - crop_w + 1)
        image = image[top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]
        alpha, beta = self.rng.uniform(0.8, 1.2), self.rng.uniform(-20, 20)
        return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8), mask

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = read_image(image_path)[..., ::-1]
        mask = read_mask(mask_path).astype(np.uint8)
        if self.augment:
            image, mask = self.augmented(image, mask)
        image = cv2.resize(np.ascontiguousarray(image), (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
        return x, torch.from_numpy(mask.astype(np.float32))[None]


class SegformerLogits(torch.nn.Module):
    """Saída do SegFormer (1/4 da resolução) ampliada ao tamanho da entrada."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(kind, encoder="resnet34", segformer="nvidia/mit-b1", pretrained=True):
    """segformer=None usa a configuração padrão (b0) sem baixar nada, só para testes."""
    if kind == "unet":
        import segmentation_models_pytorch as smp

        return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, classes=1)
    if kind == "segformer":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            return SegformerLogits(SegformerForSemanticSegmentation.from_pretrained(
                segformer, num_labels=1, ignore_mismatched_sizes=True))
        config = SegformerConfig.from_pretrained(segformer, num_labels=1) if segformer else SegformerConfig(num_labels=1)
        return SegformerLogits(SegformerForSemanticSegmentation(config))
    raise ValueError(f"Modelo desconhecido: {kind}")


def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2, 3))
    denom = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (denom + 1)).mean()


def batch_iou(logits, target):
    pred, truth = logits > 0, target > 0.5
    inter = (pred & truth).sum((1, 2, 3)).float()
    union = (pred | truth).sum((1, 2, 3)).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def train_torch(kind, data_dir, output, epochs=60, batch=8, lr=3e-4, size=SIZE, patience=15,
                device="cuda", seed=42, pretrained=True, encoder="resnet34", segformer="nvidia/mit-b1"):
    """Treina em train/, escolhe o checkpoint pelo IoU em val/ e grava output/<kind>/best.pt."""
    seed_everything(seed)
    data_dir, target = Path(data_dir), Path(output) / kind
    target.mkdir(parents=True)
    train_set = PairDataset(data_dir / "train", True, size, seed)
    val_set = PairDataset(data_dir / "val", False, size, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch, shuffle=True, generator=generator, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch, shuffle=False, num_workers=0)
    model = build_model(kind, encoder, segformer, pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    config = dict(kind=kind, size=size, encoder=encoder, segformer=segformer, epochs=epochs, batch=batch,
                  lr=lr, seed=seed, pretrained=pretrained,
                  parameters=sum(p.numel() for p in model.parameters()))
    best, stale, log = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = dice_bce(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            ious = torch.cat([batch_iou(model(x.to(device)), y.to(device)) for x, y in val_loader])
        val_iou = float(ious.mean())
        log.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_iou=val_iou))
        if val_iou > best:
            best, stale = val_iou, 0
            torch.save(model.state_dict(), target / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    config.update(best_val_iou=best, epochs_run=len(log))
    (target / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    with open(target / "log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["epoch", "train_loss", "val_iou"])
        writer.writeheader()
        writer.writerows(log)
    return target / "best.pt"


def YOLO_MODEL(weights):
    from ultralytics import YOLO

    return YOLO(str(weights))


def train_yolo(data_dir, output, epochs=60, imgsz=640, weights="yolo26s-seg.pt", device=0, seed=42, batch=8):
    from ultralytics import YOLO

    YOLO(weights).train(data=str(Path(data_dir).resolve() / "data.yaml"), epochs=epochs, imgsz=imgsz,
                        batch=batch, workers=0, device=device, project=str(Path(output).resolve()),
                        name="yolo", seed=seed, patience=15, verbose=False)
    return Path(output) / "yolo" / "weights" / "best.pt"


def load_predictor(kind, weights, device="cuda", confidence=0.25):
    """Retorna fn(imagem BGR) -> máscara bool na resolução original."""
    weights = Path(weights)
    if kind == "yolo":
        from ultralytics import YOLO

        model = YOLO(str(weights))
        yolo_device = 0 if device == "cuda" else device

        def predict_yolo(image):
            result = model.predict(source=image, conf=confidence, verbose=False, device=yolo_device)[0]
            mask = np.zeros(image.shape[:2], np.uint8)
            if result.masks is not None:
                for polygon in result.masks.xy:
                    if len(polygon) >= 3:
                        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            return mask.astype(bool)

        return predict_yolo
    config = json.loads((weights.parent / "config.json").read_text(encoding="utf-8"))
    model = build_model(kind, config["encoder"], config["segformer"], pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    size = config["size"]

    def predict_torch(image):
        height, width = image.shape[:2]
        rgb = cv2.resize(np.ascontiguousarray(image[..., ::-1]), (size, size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())[None]
        with torch.no_grad():
            prob = torch.sigmoid(model(x.to(device)))[0, 0].cpu().numpy()
        return cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5

    return predict_torch


def evaluate(predictor, split_dir, output_csv=None):
    """IoU, Dice e latência por imagem na resolução original; retorna (linhas, resumo)."""
    rows = []
    for image_path, mask_path in list_pairs(split_dir):
        image, truth = read_image(image_path), read_mask(mask_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        pred = predictor(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        inter, union = int((pred & truth).sum()), int((pred | truth).sum())
        total = int(pred.sum() + truth.sum())
        rows.append(dict(name=image_path.name, iou=inter / union if union else 1.0,
                         dice=2 * inter / total if total else 1.0, pred_empty=not pred.any(),
                         truth_area=float(truth.mean()), milliseconds=elapsed * 1000))
    summary = dict(n=len(rows), iou=float(np.mean([r["iou"] for r in rows])),
                   dice=float(np.mean([r["dice"] for r in rows])),
                   fail_rate=float(np.mean([r["iou"] < 0.5 for r in rows])),
                   empty_rate=float(np.mean([r["pred_empty"] for r in rows])),
                   milliseconds=float(np.median([r["milliseconds"] for r in rows])))
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows, summary


def train_models(kinds, data_dir, output, epochs=60, seed=42, yolo_weights="yolo26s-seg.pt",
                 encoder="resnet34", segformer="nvidia/mit-b1", retry=False):
    """Treina em sequência: cada modelo só começa quando o anterior termina.

    Retoma pulando os que têm summary.json. Uma falha é registrada em <modelo>_erro.txt e não
    interrompe os demais. Pasta de modelo incompleta exige retry=True (ela é apagada).
    Escreve comparison.csv com os concluídos; retorna (resultados, erros).
    """
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, errors = [], {}
    for kind in kinds:
        target = output / kind
        if (target / "summary.json").exists():
            results.append(json.loads((target / "summary.json").read_text(encoding="utf-8")))
            continue
        if target.exists():
            if not retry:
                raise FileExistsError(f"{target} está incompleto: apague-o ou use retry.")
            shutil.rmtree(target)
        predictor = None
        try:
            if kind == "yolo":
                weights = train_yolo(data_dir, output, epochs, weights=yolo_weights,
                                     device=0 if device == "cuda" else "cpu", seed=seed)
                parameters = sum(p.numel() for p in YOLO_MODEL(weights).model.parameters())
            else:
                weights = train_torch(kind, data_dir, output, epochs, device=device, seed=seed,
                                      encoder=encoder, segformer=segformer)
                parameters = json.loads((weights.parent / "config.json").read_text(encoding="utf-8"))["parameters"]
            predictor = load_predictor(kind, weights, device)
            _, summary = evaluate(predictor, data_dir / "val", target / "val_metrics.csv")
            row = dict(model=kind, weights=str(weights), parameters=parameters, **summary)
            (target / "summary.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
            results.append(row)
        except Exception:
            errors[kind] = traceback.format_exc()
            (output / f"{kind}_erro.txt").write_text(errors[kind], encoding="utf-8")
        finally:
            predictor = None
            gc.collect()
            torch.cuda.empty_cache()
    if results:
        with open(output / "comparison.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return results, errors
