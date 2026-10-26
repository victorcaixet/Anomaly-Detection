"""Configuração explícita; construir este objeto não exige dataset ou GPU."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    project_root: Path
    persistent_root: Path | None = None
    yolo_confidence: float = 0.25
    yolo_class_ids: tuple | None = None
    output_size: int = 256
    roi_padding_ratio: float = 0.2
    align_horizontal: bool = True
    min_component_area_ratio: float = 0.0002
    morph_kernel_size: int = 3
    background_modes: tuple = ("original", "blur", "gray")
    background_blur_sigma_ratio: float = 0.035
    mask_feather_sigma_ratio: float = 0.01
    gray_background_bgr: tuple = (127, 127, 127)
    letterbox_color_bgr: tuple = (114, 114, 114)
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 2
    coreset_sampling_ratio: float = 0.03
    num_neighbors: int = 5
    patchcore_backbone: str = "wide_resnet50_2"
    patchcore_layers: tuple = ("layer2", "layer3")
    patchcore_precision: str = "float16"
    debug_samples_per_group: int = 12
    random_seed: int = 42
    test_split_mode: str = "from_dir"
    val_split_mode: str = "none"
    val_split_ratio: float | None = None

    def __post_init__(self):
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.persistent_root = (
            Path(self.persistent_root or self.project_root).expanduser().resolve()
        )

    @property
    def dataset_root(self):
        return self.project_root / "glass-insulator"

    @property
    def processed_root(self):
        return self.project_root / "glass-preprocessed"

    @property
    def yolo_model_path(self):
        return self.project_root / "best.pt"

    @property
    def output_root(self):
        return self.project_root / "patchcore_insulators"

    @property
    def results_root(self):
        return self.output_root / "results"

    @property
    def persistent_output_root(self):
        return self.persistent_root / "patchcore_insulators"

    @property
    def persistent_results_root(self):
        return self.persistent_output_root / "results"

    @property
    def checkpoints_root(self):
        return self.persistent_root / "patchcore_checkpoints"
