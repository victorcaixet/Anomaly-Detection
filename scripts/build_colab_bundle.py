"""Empacota apenas código Python, sem datasets, resultados ou caches."""

from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root / "insulator_anomaly.zip"
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted((root / "insulator_anomaly").glob("*.py")):
        bundle.write(path, path.relative_to(root).as_posix())
print(f"Código para o Colab: {destination}")
print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")
