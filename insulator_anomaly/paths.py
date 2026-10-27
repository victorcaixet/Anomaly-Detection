"""Localiza o projeto no Windows ou no sistema de arquivos do Colab."""

from pathlib import Path
import shutil
import zipfile


def locate_project(path):
    path = Path(path).expanduser()
    for candidate in (path, path / "anomalib_glass"):
        if (candidate / "glass-insulator").is_dir() or (
            candidate / "glass-preprocessed"
        ).is_dir():
            return candidate.resolve()
    return None


def extract_project(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if (
                not (destination / member.filename)
                .resolve()
                .is_relative_to(destination)
            ):
                raise ValueError(f"Caminho inválido no ZIP: {member.filename}")
        print(f"Extraindo {archive} em {destination}. Aguarde...")
        bundle.extractall(destination)
    project = locate_project(destination)
    if project is None:
        raise FileNotFoundError(
            f"Projeto não encontrado em {destination}; confira a estrutura do ZIP."
        )
    return project


def _copy_to_local(project):
    """Imagens e best.pt do Drive para /content (leitura rápida); checkpoints ficam no Drive."""
    local = Path("/content/anomalib_project") / project.name
    local.mkdir(parents=True, exist_ok=True)
    for name in ("glass-insulator", "glass-preprocessed", "best.pt"):
        source, target = project / name, local / name
        if target.exists() or not source.exists():
            continue
        print(f"Copiando {source} para {target}. Aguarde...")
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return local


def resolve_project(project_path="", zip_path="", colab=False):
    if project_path and zip_path:
        raise ValueError("Preencha apenas project_path ou zip_path.")
    roots = [Path.cwd()]
    if colab:
        roots += [
            Path("/content/anomalib_project"),
            Path("/content/drive/MyDrive"),
            Path("/content/drive/MyDrive/Colab_Notebooks"),
        ]
    project, archive = None, None
    if project_path:
        project = locate_project(project_path)
    elif zip_path:
        archive = Path(zip_path).expanduser()
    else:
        project = next(
            (found for root in roots if (found := locate_project(root)) is not None),
            None,
        )
        if project is None:
            archive = next(
                (
                    root / "anomalib_glass.zip"
                    for root in roots
                    if (root / "anomalib_glass.zip").is_file()
                ),
                None,
            )
    if archive is not None:
        if not archive.is_file():
            raise FileNotFoundError(f"ZIP não encontrado no kernel: {archive}")
        project = extract_project(
            archive, Path("/content/anomalib_project") if colab else archive.parent
        )
    if project is None:
        raise FileNotFoundError(
            f"Projeto não encontrado no kernel (cwd={Path.cwd()}). "
            "Informe PROJECT_PATH ou ZIP_PATH. No Colab, disponibilize a pasta ou o ZIP no Drive; "
            "caminhos C:/... exigem um kernel Windows local."
        )
    persistent = project
    if colab and project.is_relative_to(Path("/content/drive")):
        project = _copy_to_local(project)
    elif colab:
        persistent = (
            archive.parent / "anomalib_glass"
            if archive is not None
            and archive.resolve().is_relative_to(Path("/content/drive"))
            else Path("/content/drive/MyDrive/anomalib_glass")
        )
    return project, persistent
