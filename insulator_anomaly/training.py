"""Treino e checkpoints. Métricas e predições pertencem à etapa de avaliação."""

import gc
import hashlib
import json
import shutil

from .data import validate_processed_dataset
from .preprocessing import preprocessing_signature


def experiment_signature(config, mode):
    state = {
        "preprocessing_signature": preprocessing_signature(config),
        "mode": mode,
        "output_size": config.output_size,
        "backbone": config.patchcore_backbone,
        "layers": list(config.patchcore_layers),
        "precision": config.patchcore_precision,
        "coreset_sampling_ratio": config.coreset_sampling_ratio,
        "num_neighbors": config.num_neighbors,
        "random_seed": config.random_seed,
        "test_split_mode": config.test_split_mode,
        "val_split_mode": config.val_split_mode,
        "val_split_ratio": config.val_split_ratio,
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def find_checkpoint(config, mode, signature):
    root = config.checkpoints_root / mode
    state_path = root / "training_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("experiment_signature") != signature:
        return None
    last = root / "last.ckpt"
    if last.is_file():
        return last
    return next(
        iter(
            sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        ),
        None,
    )


def train_one_patchcore(config, mode, force=False):
    """Reutiliza o treino compatível; preserva checkpoints de outros protocolos."""
    if mode not in config.background_modes:
        raise ValueError(f"Modo inválido: {mode}")
    validate_processed_dataset(config.processed_root / mode)
    signature = experiment_signature(config, mode)
    checkpoint_root = config.checkpoints_root / mode
    checkpoint = None if force else find_checkpoint(config, mode, signature)
    report_path = config.persistent_results_root / mode / "experiment_report.json"
    if checkpoint is not None:
        print(f"{mode}: treino compatível já salvo: {checkpoint}")
        return dict(
            background_mode=mode,
            checkpoint=str(checkpoint),
            experiment_signature=signature,
            resumed_from_checkpoint=True,
        )
    # Dados portados ou um protocolo novo não autorizam sobrescrever um treino anterior.
    previous = [
        *checkpoint_root.rglob("*.ckpt"),
        *(config.results_root / mode).rglob("*.ckpt"),
        *(config.persistent_results_root / mode).rglob("*.ckpt"),
    ]
    if previous and not force:
        raise FileExistsError(
            f"{mode}: há checkpoints antigos ou incompatíveis. Avalie-os normalmente; "
            "para outro treino, use um persistent_root novo ou force=True explicitamente."
        )

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    import torch

    model = engine = None
    try:
        datamodule = Folder(
            name=f"insulators_{mode}",
            root=config.processed_root / mode,
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/anomaly",
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            test_split_mode=config.test_split_mode,
            val_split_mode=config.val_split_mode,
            val_split_ratio=config.val_split_ratio,
            seed=config.random_seed,
        )
        model = Patchcore(
            backbone=config.patchcore_backbone,
            layers=list(config.patchcore_layers),
            pre_trained=True,
            coreset_sampling_ratio=config.coreset_sampling_ratio,
            num_neighbors=config.num_neighbors,
            precision=config.patchcore_precision,
            visualizer=False,
            pre_processor=Patchcore.configure_pre_processor(
                image_size=(config.output_size, config.output_size),
                center_crop_size=None,
            ),
        )
        run_root = config.results_root / mode
        run_root.mkdir(parents=True, exist_ok=True)
        engine = Engine(
            default_root_dir=run_root,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        engine.fit(model=model, datamodule=datamodule)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_root / "last.ckpt.tmp"
        engine.trainer.save_checkpoint(str(temporary), weights_only=False)
        temporary.replace(checkpoint_root / "last.ckpt")
        state = dict(mode=mode, experiment_signature=signature, completed_fit=True)
        (checkpoint_root / "training_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        report = dict(
            background_mode=mode,
            dataset_root=str(config.processed_root / mode),
            checkpoint=str(checkpoint_root / "last.ckpt"),
            experiment_signature=signature,
            resumed_from_checkpoint=False,
            configuration=dict(
                output_size=config.output_size,
                backbone=config.patchcore_backbone,
                layers=list(config.patchcore_layers),
                precision=config.patchcore_precision,
                coreset_sampling_ratio=config.coreset_sampling_ratio,
                num_neighbors=config.num_neighbors,
                test_split_mode=config.test_split_mode,
                val_split_mode=config.val_split_mode,
                val_split_ratio=config.val_split_ratio,
            ),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report_path.resolve() != (run_root / report_path.name).resolve():
            shutil.copy2(report_path, run_root / report_path.name)
        print(f"{mode}: checkpoint salvo: {report['checkpoint']}")
        return report
    finally:
        del model, engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_all(config, force_modes=()):
    reports = [
        train_one_patchcore(config, mode, force=mode in force_modes)
        for mode in config.background_modes
    ]
    config.persistent_results_root.mkdir(parents=True, exist_ok=True)
    (config.persistent_results_root / "comparison.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    return reports
