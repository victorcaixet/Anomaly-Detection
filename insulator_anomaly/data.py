"""Leitura dos splits e nomes estáveis de imagens."""

from dataclasses import dataclass
from pathlib import Path
import re

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    name: str
    image_path: Path
    output_relative_dir: Path


def list_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.suffix.lower() in extensions)
    )


def build_dataset_records(dataset_root: Path) -> list[SampleRecord]:
    """
    Usa a estrutura de pastas como fonte dos rotulos.

    Esperado:
        dataset_root/
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            └── anomaly/

    O PatchCore sera treinado somente com train/good.
    test/good e test/anomaly ficam exclusivamente para avaliacao.
    """
    required_dirs = {
        "train_good": dataset_root / "train" / "good",
        "test_good": dataset_root / "test" / "good",
        "test_anomaly": dataset_root / "test" / "anomaly",
    }
    missing = [path for path in required_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Estrutura do dataset incompleta. Diretorios ausentes: "
            + ", ".join(map(str, missing))
        )
    records: list[SampleRecord] = []
    groups = (
        ("train_good", required_dirs["train_good"], Path("train/good")),
        ("test_good", required_dirs["test_good"], Path("test/good")),
        ("test_anomaly", required_dirs["test_anomaly"], Path("test/anomaly")),
    )
    for name, source_dir, output_relative_dir in groups:
        images = list_files(source_dir, IMAGE_EXTENSIONS)
        if not images:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {source_dir}")
        records.extend(
            (
                SampleRecord(
                    name=name,
                    image_path=image_path,
                    output_relative_dir=output_relative_dir,
                )
                for image_path in images
            )
        )
    print(
        f"Dataset encontrado: {sum((record.name == 'train_good' for record in records))} train/good, {sum((record.name == 'test_good' for record in records))} test/good, {sum((record.name == 'test_anomaly' for record in records))} test/anomaly."
    )
    return records


def safe_output_name(image_path: Path, dataset_root: Path) -> str:
    """Preserva unicidade quando a origem possui subpastas de defeitos."""
    relative = image_path.relative_to(dataset_root).with_suffix("")
    safe = "__".join(relative.parts)
    safe = re.sub("[^A-Za-z0-9_.-]+", "_", safe)
    return safe + ".png"


def validate_processed_dataset(dataset_root: Path) -> None:
    required = (
        dataset_root / "train/good",
        dataset_root / "test/good",
        dataset_root / "test/anomaly",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset processado incompleto. Use prepare_processed_datasets(config, generate=True) em um destino novo. Ausentes: "
            + ", ".join(map(str, missing))
        )
    empty = [path for path in required if not list_files(path, IMAGE_EXTENSIONS)]
    if empty:
        raise RuntimeError(
            "Diretorios processados vazios: "
            + ", ".join(map(str, empty))
            + ". O dataset deve conter imagens nas pastas train/good, test/good e test/anomaly."
        )
