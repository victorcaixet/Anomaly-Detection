"""YOLO, ROI e três tratamentos de fundo; geração sempre explícita."""

from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import math
import cv2
import numpy as np

from .data import (
    IMAGE_EXTENSIONS,
    build_dataset_records,
    list_files,
    safe_output_name,
    validate_processed_dataset,
)


def odd_at_least(value: int, minimum: int = 3) -> int:
    """Retorna um inteiro impar adequado a kernels Gaussian/morfologicos."""
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {path}")
    return image


def predict_object_mask(
    config, image_path: Path, yolo_model: Any, image: np.ndarray
) -> np.ndarray:
    """Gera mascara do objeto para uma imagem sem mascara salva."""
    result = yolo_model.predict(
        source=image, conf=config.yolo_confidence, verbose=False, device=config.yolo_device
    )[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError(f"YOLO-seg nao encontrou objeto em {image_path}")
    masks = result.masks.data.detach().cpu().numpy()
    if config.yolo_class_ids is not None:
        if result.boxes is None or result.boxes.cls is None:
            raise RuntimeError(f"YOLO nao retornou classes para {image_path}")
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        selected = np.isin(classes, np.asarray(config.yolo_class_ids))
        masks = masks[selected]
        if len(masks) == 0:
            raise RuntimeError(
                f"Nenhuma instancia das classes {config.yolo_class_ids} em {image_path}"
            )
    union_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
    union_mask = cv2.resize(
        union_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return union_mask.astype(bool)


def clean_mask(config, mask: np.ndarray) -> np.ndarray:
    """Remove ruido pequeno e aplica fechamento leve, preservando anomalias."""
    binary = mask.astype(np.uint8)
    min_area = max(8, int(mask.size * config.min_component_area_ratio))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == component_id] = 1
    kernel_size = odd_at_least(config.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    if cleaned.sum() == 0:
        raise ValueError("A mascara ficou vazia apos a limpeza.")
    return cleaned.astype(bool)


def principal_axis_angle(mask: np.ndarray) -> float:
    """Calcula o angulo, em graus, do maior eixo da mascara em relacao a x."""
    y, x = np.nonzero(mask)
    if len(x) < 10:
        return 0.0
    points = np.column_stack((x, y)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def rotate_image_and_mask(
    image: np.ndarray, mask: np.ndarray, angle_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotaciona em canvas expandido para nao cortar as extremidades."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(math.ceil(height * sin_value + width * cos_value))
    new_height = int(math.ceil(height * cos_value + width * sin_value))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return (rotated_image, rotated_mask)


def crop_with_padding(
    image: np.ndarray, mask: np.ndarray, padding_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai ROI completa com margem proporcional ao objeto."""
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("Mascara vazia ao calcular a ROI.")
    x_min, x_max = (int(x.min()), int(x.max()) + 1)
    y_min, y_max = (int(y.min()), int(y.max()) + 1)
    box_width = x_max - x_min
    box_height = y_max - y_min
    pad_x = max(2, int(round(box_width * padding_ratio)))
    pad_y = max(2, int(round(box_height * padding_ratio)))
    x_min = max(0, x_min - pad_x)
    x_max = min(image.shape[1], x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(image.shape[0], y_max + pad_y)
    return (image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max])


def soft_alpha(config, mask: np.ndarray) -> np.ndarray:
    """Cria transicao suave para nao introduzir uma borda artificial."""
    sigma = max(1.0, min(mask.shape) * config.mask_feather_sigma_ratio)
    kernel_size = odd_at_least(int(math.ceil(6 * sigma + 1)))
    alpha = cv2.GaussianBlur(
        mask.astype(np.float32), (kernel_size, kernel_size), sigmaX=sigma
    )
    return np.clip(alpha, 0.0, 1.0)[..., None]


def apply_background_mode(
    config, image: np.ndarray, mask: np.ndarray, mode: str
) -> np.ndarray:
    """Mantem o objeto original e trata somente o exterior da mascara."""
    if mode == "original":
        return image.copy()
    alpha = soft_alpha(config, mask)
    if mode == "blur":
        sigma = max(2.0, min(image.shape[:2]) * config.background_blur_sigma_ratio)
        kernel_size = odd_at_least(int(math.ceil(6 * sigma + 1)))
        background = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    elif mode == "gray":
        background = np.full_like(image, config.gray_background_bgr, dtype=np.uint8)
    else:
        raise ValueError(f"Modo de fundo desconhecido: {mode}")
    blended = alpha * image.astype(np.float32) + (1.0 - alpha) * background.astype(
        np.float32
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def letterbox(
    image: np.ndarray,
    output_size: int,
    color: tuple[int, int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Redimensiona preservando proporcao e centraliza em canvas quadrado."""
    height, width = image.shape[:2]
    scale = min(output_size / width, output_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = (
        cv2.INTER_NEAREST
        if is_mask
        else cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    if is_mask:
        canvas = np.zeros((output_size, output_size), dtype=resized.dtype)
    else:
        canvas = np.full((output_size, output_size, 3), color, dtype=resized.dtype)
    x_offset = (output_size - new_width) // 2
    y_offset = (output_size - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def preprocess_image_and_mask(
    config, image: np.ndarray, object_mask: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Retorna as tres imagens processadas e dados intermediarios de debug."""
    mask = clean_mask(config, object_mask)
    if config.align_horizontal:
        angle = principal_axis_angle(mask)
        image, mask = rotate_image_and_mask(image, mask, angle)
    roi_image, roi_mask = crop_with_padding(image, mask, config.roi_padding_ratio)
    outputs: dict[str, np.ndarray] = {}
    for mode in config.background_modes:
        treated = apply_background_mode(config, roi_image, roi_mask, mode)
        outputs[mode] = letterbox(
            treated, config.output_size, config.letterbox_color_bgr
        )
    debug = {
        "aligned_roi": letterbox(
            roi_image, config.output_size, config.letterbox_color_bgr
        ),
        "object_mask": letterbox(
            roi_mask.astype(np.uint8) * 255, config.output_size, (0, 0, 0), is_mask=True
        ),
    }
    return (outputs, debug)


def preprocess_all(config) -> dict[str, Any]:
    """Gera original/blur/gray mantendo exatamente o split fornecido pelo dataset."""
    if any((config.processed_root / mode).exists() for mode in config.background_modes):
        raise FileExistsError(
            "Pré-processamento existente: escolha um destino novo para gerar sem sobrescrever dados."
        )
    manifest_rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {"groups": {}, "errors": []}
    records = build_dataset_records(config.dataset_root)
    group_counts: dict[str, dict[str, int]] = {}
    if not config.yolo_model_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLO-seg nao encontrado: {config.yolo_model_path}"
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Instale o Ultralytics: pip install ultralytics") from error
    yolo_model = YOLO(str(config.yolo_model_path))
    debug_counts: dict[str, int] = {}
    for record in records:
        counts = group_counts.setdefault(record.name, {"found": 0, "processed": 0})
        counts["found"] += 1
        try:
            image = read_image(record.image_path)
            object_mask = predict_object_mask(
                config, record.image_path, yolo_model, image
            )
            mode_outputs, debug_outputs = preprocess_image_and_mask(
                config, image, object_mask
            )
            output_name = safe_output_name(record.image_path, config.dataset_root)
            row: dict[str, str] = {
                "group": record.name,
                "source_image": str(record.image_path),
                "source_object_mask": "YOLO-seg (gerada em memoria)",
            }
            for mode, output_image in mode_outputs.items():
                destination = (
                    config.processed_root
                    / mode
                    / record.output_relative_dir
                    / output_name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), output_image):
                    raise OSError(f"Falha ao salvar: {destination}")
                row[f"processed_{mode}"] = str(destination)
            debug_index = debug_counts.get(record.name, 0)
            if debug_index < config.debug_samples_per_group:
                debug_dir = config.output_root / "debug" / record.name
                debug_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(output_name).stem
                cv2.imwrite(
                    str(debug_dir / f"{stem}__roi.png"), debug_outputs["aligned_roi"]
                )
                cv2.imwrite(
                    str(debug_dir / f"{stem}__mask.png"), debug_outputs["object_mask"]
                )
                debug_counts[record.name] = debug_index + 1
            manifest_rows.append(row)
            counts["processed"] += 1
        except Exception as error:
            summary["errors"].append(
                {
                    "image": str(record.image_path),
                    "group": record.name,
                    "error": str(error),
                }
            )
    summary["groups"] = group_counts
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_root / "preprocessing_manifest.csv"
    fieldnames = [
        "group",
        "source_image",
        "source_object_mask",
        *(f"processed_{mode}" for mode in config.background_modes),
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary_path = config.output_root / "preprocessing_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Manifesto: {manifest_path}")
    print(f"Diagnosticos: {config.output_root / 'debug'}")
    if summary["errors"]:
        print(
            f"ATENCAO: {len(summary['errors'])} arquivo(s) falharam. Consulte {summary_path}."
        )
    return summary


def source_dataset_signature(config) -> str:
    """Assina nomes, tamanhos e datas das imagens do dataset e o modelo YOLO."""
    rows: list[tuple[Any, ...]] = []
    for path in list_files(config.dataset_root, IMAGE_EXTENSIONS):
        stat = path.stat()
        rows.append(
            (
                "image",
                path.relative_to(config.dataset_root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    if config.yolo_model_path.is_file():
        stat = config.yolo_model_path.stat()
        rows.append(
            ("yolo", config.yolo_model_path.name, stat.st_size, stat.st_mtime_ns)
        )
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def expected_preprocessing_state(config) -> dict[str, Any]:
    """Configuracao que precisa coincidir para um cache ser reutilizado."""
    return {
        "pipeline_version": "patchcore-dataset-folder-v3-axis-sign",
        "source_signature": source_dataset_signature(config),
        "output_size": config.output_size,
        "roi_padding_ratio": config.roi_padding_ratio,
        "align_horizontal": config.align_horizontal,
        "min_component_area_ratio": config.min_component_area_ratio,
        "morph_kernel_size": config.morph_kernel_size,
        "background_modes": list(config.background_modes),
        "background_blur_sigma_ratio": config.background_blur_sigma_ratio,
        "mask_feather_sigma_ratio": config.mask_feather_sigma_ratio,
        "gray_background_bgr": list(config.gray_background_bgr),
        "letterbox_color_bgr": list(config.letterbox_color_bgr),
        "yolo_confidence": config.yolo_confidence,
        "yolo_class_ids": config.yolo_class_ids,
    }


def preprocessing_signature(config) -> str:
    return hashlib.sha256(
        json.dumps(expected_preprocessing_state(config), sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_processed_datasets(config, generate=False):
    """Valida dados prontos; generate=True gera apenas num destino ainda vazio."""
    if generate:
        return preprocess_all(config)
    for mode in config.background_modes:
        validate_processed_dataset(config.processed_root / mode)
    return {"cache": "existing", "dataset_root": str(config.processed_root)}
