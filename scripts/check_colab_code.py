"""Cole como célula após disponibilizar o código e antes de configurar o projeto."""

from pathlib import Path
import hashlib
import importlib.util
import sys
import zipfile

DRIVE_ZIP = Path("/content/drive/MyDrive/Colab_Notebooks/insulator_anomaly.zip")
# Opcional: hash impresso pelo empacotador local, para confirmar a versão enviada.
EXPECTED_ZIP_SHA256 = ""

if not DRIVE_ZIP.is_file():
    raise FileNotFoundError(f"Monte o Drive e confira o caminho: {DRIVE_ZIP}")

zip_hash = hashlib.sha256(DRIVE_ZIP.read_bytes()).hexdigest()
print("ZIP do Drive:", DRIVE_ZIP)
print("SHA256:", zip_hash)
if EXPECTED_ZIP_SHA256 and zip_hash != EXPECTED_ZIP_SHA256.strip().lower():
    raise RuntimeError("O ZIP do Drive não corresponde à versão local esperada. Atualize o arquivo.")

spec = importlib.util.find_spec("insulator_anomaly")
if spec is None or spec.origin is None:
    raise ModuleNotFoundError("Execute primeiro a célula que disponibiliza o código.")
package_dir = Path(spec.origin).parent
print("Pacote encontrado pelo kernel:", package_dir)

with zipfile.ZipFile(DRIVE_ZIP) as bundle:
    expected = {
        name.removeprefix("insulator_anomaly/"): bundle.read(name)
        for name in bundle.namelist()
        if name.startswith("insulator_anomaly/") and name.endswith(".py")
    }
    if not expected:
        raise ValueError("O ZIP não contém código insulator_anomaly/*.py.")
    missing = [name for name in expected if not (package_dir / name).is_file()]
    changed = [name for name, data in expected.items()
               if name not in missing and (package_dir / name).read_bytes() != data]
    extra = sorted({p.relative_to(package_dir).as_posix() for p in package_dir.rglob("*.py")} - set(expected))

if missing or changed or extra:
    print("Ausentes:", missing)
    print("Diferentes:", changed)
    print("Extras:", extra)
    raise RuntimeError(
        "Pacote diferente do ZIP do Drive. Reinicie a sessão, force CODE_ZIP_PATH "
        "para esse ZIP e execute novamente o carregador."
    )

print("OK: os arquivos do pacote correspondem ao ZIP disponível no Drive.")
if any(name.startswith("insulator_anomaly.") for name in sys.modules):
    print("Há módulos já importados. Se atualizou arquivos nesta sessão, reinicie: "
          "a comparação confere o disco, não o código antigo que pode estar na memória.")
