"""Experimento pareado: ROI com fundo versus segmentos, treino somente normal.

Caixas opcionais usam coordenadas na imagem original, nunca máscaras de pixels.
Sem caixa, a imagem inteira é preservada. Não usa YOLO implicitamente.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import build_dataset_records
from .preprocessing import letterbox, read_image


@dataclass(frozen=True)
class CropSettings:
    output_size: int = 256
    margin: float = 0.20
    segment_aspect: float = 1.5
    overlap: float = 0.25
    validation_fraction: float = 0.20
    normal_fpr: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if not (0 <= self.margin and self.segment_aspect > 0 and self.output_size >= 16):
            raise ValueError("Margem >= 0, aspecto > 0 e resolução >= 16 exigidos.")
        if not (0 <= self.overlap < 1 and 0 < self.validation_fraction < 1):
            raise ValueError("Overlap em [0,1) e fração de validação em (0,1).")
        if not 0 < self.normal_fpr < 1:
            raise ValueError("normal_fpr deve estar em (0,1).")


def segment_boxes(width, height, aspect=1.5, overlap=0.25):
    """Janelas sobrepostas cobrem todo o recorte, sem descartar áreas sem objeto.

    O eixo maior é dividido; cadeias diagonais podem exigir caixas melhores.
    """
    if width < 1 or height < 1 or aspect <= 0 or not 0 <= overlap < 1:
        raise ValueError("Geometria ou overlap inválidos.")
    horizontal = width >= height
    length, short = (width, height) if horizontal else (height, width)
    window = min(length, max(1, round(short * aspect)))
    step = max(1, round(window * (1 - overlap)))
    starts = list(range(0, length - window + 1, step))
    if starts[-1] != length - window:
        starts.append(length - window)
    return [(s, 0, s + window, height) if horizontal else (0, s, width, s + window)
            for s in starts]


def padded_box(box, width, height, margin):
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Caixa fora da imagem: {box}, imagem={width}x{height}")
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
            min(width, math.ceil(x1+dx)), min(height, math.ceil(y1+dy)))


def locate_crop_boxes(source_root, model_path, confidence=0.25, class_ids=None, device=None):
    """Usa apenas as caixas YOLO, sem aplicar máscaras; fallback é quadro inteiro.

    União das instâncias selecionadas. Uma detecção parcial ainda pode excluir
    parte da cadeia: inspecione os recortes e compare com caixas manuais.
    """
    from ultralytics import YOLO

    root = Path(source_root).resolve()
    model = YOLO(str(model_path))
    boxes, diagnostics = {}, []
    records = build_dataset_records(root)
    for number, record in enumerate(records, 1):
        image = read_image(record.image_path)
        h, w = image.shape[:2]
        result = model.predict(source=image, conf=confidence, verbose=False, device=device)[0]
        xyxy = np.empty((0,4))
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            if class_ids is not None:
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                xyxy = xyxy[np.isin(classes, class_ids)]
        relative = record.image_path.relative_to(root).as_posix()
        if len(xyxy):
            box = [max(0.,float(xyxy[:,0].min())), max(0.,float(xyxy[:,1].min())),
                   min(float(w),float(xyxy[:,2].max())), min(float(h),float(xyxy[:,3].max()))]
        else:
            box = [0,0,w,h]
        boxes[relative] = box
        diagnostics.append(dict(source_id=relative, n_detections=len(xyxy),
                                fallback_full_image=not len(xyxy), x0=box[0],y0=box[1],x1=box[2],y1=box[3]))
        print(f"Localização: {number}/{len(records)} — {relative}, instâncias={len(xyxy)}")
    return boxes, pd.DataFrame(diagnostics)


def prepare_crop_experiment(source_root, experiment_root, settings=None,
                            boxes=None, groups=None, duplicate_policy="error"):
    """Cria destino novo, manifesto e duas entradas para as mesmas origens.

    boxes: {caminho relativo POSIX: [x0,y0,x1,y1]} na resolução original.
    groups: {caminho relativo POSIX: cadeia/voo/sequência}; obrigatório fornecer
    quando existem imagens relacionadas. Na ausência, cada arquivo é um grupo; duplicatas exatas unem grupos.
    O holdout normal é feito por grupo ANTES de produzir os segmentos.
    duplicate_policy="exclude_train" preserva o teste e exclui do experimento
    grupos de treino ligados ao teste por conteúdo ou groups. Não altera a
    origem. Rótulos conflitantes continuam sendo erro em qualquer política.
    """
    if duplicate_policy not in {"error", "exclude_train"}:
        raise ValueError("duplicate_policy deve ser 'error' ou 'exclude_train'.")
    settings = settings or CropSettings()
    source_root, root = Path(source_root).resolve(), Path(experiment_root).resolve()
    if root.exists():
        raise FileExistsError(f"Use um destino novo para preservar o experimento: {root}")
    records = build_dataset_records(source_root)
    boxes, groups = boxes or {}, groups or {}
    sources = []
    for record in records:
        relative = record.image_path.relative_to(source_root).as_posix()
        sources.append(dict(record=record, relative=relative,
                            group=str(groups.get(relative, relative))))
    known = {s['relative'] for s in sources}
    if (set(boxes) | set(groups)) - known:
        raise ValueError("Há caminhos de boxes/groups que não existem no dataset.")
    # Une grupos ligados por conteúdo antes de sortear o holdout, inclusive
    # conexões transitivas entre grupos fornecidos pelo usuário.
    parents = {s['group']: s['group'] for s in sources}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    hashes = {}
    for source in sources:
        digest = hashlib.sha256(source['record'].image_path.read_bytes()).hexdigest()
        source['sha256'] = digest
        if digest in hashes:
            previous = hashes[digest]
            conflicting_labels = ((previous['record'].name == 'test_anomaly') !=
                                  (source['record'].name == 'test_anomaly'))
            if conflicting_labels or (previous['record'].name != source['record'].name
                                      and duplicate_policy == 'error'):
                raise ValueError(
                    "Imagem duplicada entre partições ou rótulos do dataset (SHA-256): "
                    f"{previous['relative']} <-> {source['relative']}. "
                    "Corrija as cópias na origem antes de executar; "
                    "treino e teste devem ser independentes. "
                    "Para cópias normais, use duplicate_policy='exclude_train' "
                    "para preservar o teste e excluir o grupo de treino do experimento."
                )
            left, right = find(previous['group']), find(source['group'])
            parents[max(left, right)] = min(left, right)
        else:
            hashes[digest] = source
    for source in sources:
        source['group'] = find(source['group'])
    train_groups = {s['group'] for s in sources if s['record'].name == 'train_good'}
    test_groups = {s['group'] for s in sources if s['record'].name != 'train_good'}
    overlap_groups = train_groups & test_groups
    if overlap_groups and duplicate_policy == 'error':
        raise ValueError("Grupo compartilhado entre treino e teste; corrija o split.")
    excluded = [dict(source_id=s['relative'], group_id=s['group'],
                     source_sha256=s['sha256'], reason='train_group_overlaps_test')
                for s in sources
                if s['record'].name == 'train_good' and s['group'] in overlap_groups]
    sources = [s for s in sources
               if not (s['record'].name == 'train_good' and s['group'] in overlap_groups)]
    train_groups -= overlap_groups
    if len(train_groups) < 2:
        raise ValueError("São necessários ao menos dois grupos normais independentes de treino após unir duplicatas.")
    ordered = sorted(train_groups)
    np.random.default_rng(settings.seed).shuffle(ordered)
    n_val = min(len(ordered)-1, max(1, math.ceil(len(ordered)*settings.validation_fraction)))
    validation = set(ordered[:n_val])
    for source in sources:
        source['split'] = (('val' if source['group'] in validation else 'train')
                           if source['record'].name == 'train_good' else 'test')
    root.mkdir(parents=True)
    pd.DataFrame(excluded, columns=['source_id', 'group_id', 'source_sha256', 'reason']).to_csv(
        root/'excluded_sources.csv', index=False)
    if excluded:
        print(f"Excluídas {len(excluded)} imagens de treino de grupos ligados ao teste. "
              f"Originais preservados. Relatório: {root/'excluded_sources.csv'}")
    rows = []
    for number, source in enumerate(sources, 1):
        record, relative = source['record'], source['relative']
        image = read_image(record.image_path)
        height, width = image.shape[:2]
        box = padded_box(boxes.get(relative, (0, 0, width, height)), width, height, settings.margin)
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1]
        label = int(record.name == 'test_anomaly')
        category = 'anomaly' if label else 'good'
        key = hashlib.sha256(relative.encode()).hexdigest()[:20]
        variants = {'whole': [(0, 0, crop.shape[1], crop.shape[0])],
                    'segments': segment_boxes(crop.shape[1], crop.shape[0], settings.segment_aspect, settings.overlap)}
        for mode, windows in variants.items():
            for index, (a,b,c,d) in enumerate(windows):
                path = root/'glass-preprocessed'/mode/source['split']/category/f'{key}_{index:04d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                rendered = letterbox(crop[b:d, a:c], settings.output_size, (114,114,114))
                if not cv2.imwrite(str(path), rendered):
                    raise OSError(f"Falha ao salvar {path}")
                rows.append(dict(mode=mode, split=source['split'], source_id=relative,
                                 group_id=source['group'], source_sha256=source['sha256'],
                                 y_true=label, image_path=str(path.resolve()),
                                 box_source='provided' if relative in boxes else 'full_image',
                                 x0=x0+a, y0=y0+b, x1=x0+c, y1=y0+d))
        print(f"Recortes: {number}/{len(sources)} — {relative}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root/'manifest.csv', index=False)
    (root/'protocol.json').write_text(json.dumps(dict(
        settings=asdict(settings), source_root=str(source_root),
        duplicate_policy=duplicate_policy, n_excluded_train=len(excluded),
        aggregation='max score per original image', training='normal only',
        calibration='held-out normal groups only; strict score > threshold',
        group_warning='User groups' if groups else 'File groups only: visual/sequence leakage not excluded',
        box_warning='Provided boxes must describe object extent, never use defect labels to choose crops',
    ), indent=2), encoding='utf-8')
    return manifest


def aggregate_sources(manifest, predictions):
    """Não atribui rótulos de defeito a segmentos individuais."""
    if predictions.image_path.duplicated().any() or set(predictions.image_path) != set(manifest.image_path):
        raise ValueError("Predições não cobrem exatamente o manifesto selecionado.")
    merged = manifest.merge(predictions, on='image_path', validate='one_to_one')
    if not np.isfinite(merged.pred_score).all():
        raise ValueError("Scores não finitos.")
    return merged.groupby(['mode','split','source_id','group_id','y_true'], as_index=False).agg(
        pred_score=('pred_score','max'), n_crops=('image_path','size'))


def compare_source_scores(source_scores, normal_fpr=0.05):
    from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
    rows, decisions = [], []
    paired = None
    for mode, frame in source_scores.groupby('mode'):
        val = frame[frame.split == 'val']
        test = frame[frame.split == 'test'].copy()
        if val.empty or val.y_true.any() or set(test.y_true) != {0,1}:
            raise ValueError("Exige validação apenas normal e teste com ambas as classes.")
        identity = set(zip(test.source_id, test.y_true))
        if paired is not None and identity != paired:
            raise ValueError("Modos não possuem o mesmo teste de origem.")
        paired = identity
        threshold = float(np.quantile(val.pred_score, 1-normal_fpr, method='higher'))
        test['pred_label'] = (test.pred_score > threshold).astype(int)
        test['threshold'] = threshold
        tn,fp,fn,tp = confusion_matrix(test.y_true,test.pred_label,labels=[0,1]).ravel()
        rows.append(dict(mode=mode, AP=average_precision_score(test.y_true,test.pred_score),
                         AUROC=roc_auc_score(test.y_true,test.pred_score), threshold=threshold,
                         n_validation=len(val), target_normal_fpr=normal_fpr,
                         validation_fpr=float((val.pred_score > threshold).mean()),
                         TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                         recall=tp/(tp+fn), precision=tp/(tp+fp) if tp+fp else 0., FPR=fp/(fp+tn)))
        decisions.append(test)
    return pd.DataFrame(rows), pd.concat(decisions, ignore_index=True)


def run_crop_experiment(experiment_root, save_heatmaps=True, accelerator="auto"):
    """Treina dois modelos normais e compara por origem, sem calibrar no teste.

    Requer o ambiente Anomalib do notebook. Mantém os logs existentes.
    """
    import gc
    import torch
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything
    from torch.utils.data import DataLoader
    from .training import train_one_patchcore

    root = Path(experiment_root).resolve()
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    settings = CropSettings(**protocol['settings'])
    manifest = pd.read_csv(root/'manifest.csv')
    config = ProjectConfig(root, background_modes=('whole','segments'), output_size=settings.output_size,
                           random_seed=settings.seed, accelerator=accelerator)
    fingerprint = hashlib.sha256((root/'manifest.csv').read_bytes() + (root/'protocol.json').read_bytes())
    for path in manifest.image_path:
        fingerprint.update(Path(path).read_bytes())
    preprocessing_id = fingerprint.hexdigest()
    sources = []
    map_manifests = []
    for mode in config.background_modes:
        seed_everything(settings.seed, workers=True)
        report = train_one_patchcore(config, mode, post_processor=False, preprocessing_id=preprocessing_id)
        model = Patchcore.load_from_checkpoint(report['checkpoint'], map_location='cpu', weights_only=False)
        engine = Engine(default_root_dir=root/'inference'/mode, logger=False,
                        accelerator=config.accelerator, devices=1,
                        enable_progress_bar=False, enable_model_summary=False)
        wanted = manifest[(manifest['mode'] == mode) & manifest.split.isin(['val','test'])]
        rows = []
        map_rows = []
        try:
            for split in ('val','test'):
                dataset = PredictDataset(path=root/'glass-preprocessed'/mode/split)
                loader = DataLoader(dataset,batch_size=1,num_workers=0,collate_fn=dataset.collate_fn)
                batches = engine.predict(model=model,dataloaders=loader,return_predictions=True)
                for batch in batches:
                    paths = batch.image_path
                    if isinstance(paths,(str,Path)):
                        paths = [paths]
                    scores = batch.pred_score.detach().cpu().numpy().reshape(-1)
                    rows.extend(dict(image_path=str(Path(p).resolve()), pred_score=float(s))
                                for p,s in zip(paths,scores,strict=True))
                    if save_heatmaps:
                        arrays = batch.anomaly_map.detach().cpu().float().numpy()
                        for p,array in zip(paths,arrays,strict=True):
                            array = np.asarray(array).squeeze()
                            if array.ndim != 2 or not np.isfinite(array).all():
                                raise ValueError(f'Mapa inválido: {p}')
                            key = hashlib.sha256(str(Path(p).resolve()).encode()).hexdigest()[:20]
                            map_path = root/'maps'/mode/f'{key}.npy'
                            map_path.parent.mkdir(parents=True,exist_ok=True)
                            np.save(map_path,array)
                            map_rows.append(dict(image_path=str(Path(p).resolve()),map_path=str(map_path)))
            predictions = pd.DataFrame(rows)
            predictions.to_csv(root/f'{mode}_crop_scores.csv', index=False)
            sources.append(aggregate_sources(wanted,predictions))
            if save_heatmaps:
                maps = pd.DataFrame(map_rows)
                if maps.image_path.duplicated().any() or set(maps.image_path) != set(wanted.image_path):
                    raise ValueError('Mapas não cobrem o manifesto de validação/teste.')
                map_manifests.append(wanted.merge(maps,on='image_path',validate='one_to_one'))
        finally:
            del model, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    scores = pd.concat(sources,ignore_index=True)
    comparison, decisions = compare_source_scores(scores, settings.normal_fpr)
    scores.to_csv(root/'source_scores.csv',index=False)
    comparison.to_csv(root/'comparison.csv',index=False)
    decisions.to_csv(root/'test_decisions.csv',index=False)
    if map_manifests:
        pd.concat(map_manifests,ignore_index=True).to_csv(root/'map_manifest.csv',index=False)
    return comparison, decisions
