"""Comparação reprodutível de mapas; não altera o checkpoint.

MEBin independente em NumPy/OpenCV, com faixa min/max POR IMAGEM. Essa
adaptação evita usar o lote de teste para normalizar, e não é reprodução exata
do MEBin original (faixa definida pelo conjunto). Referência:
https://github.com/HUST-SLOW/MEBin — Huang et al., CVPR 2025.

Mapas antigos: diagnóstico exploratório. Decisão: referência e calibração
normais separadas, nunca imagens de teste para ajustar escalas ou limiares.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .heatmaps import validate_heatmap


@dataclass(frozen=True)
class MapSettings:
    windows: tuple = (9, 17)
    contrast_threshold: float = 3.0
    scale_floor_ratio: float = 0.05
    pixel_quantile: float = 0.995
    mebin_step: int = 4
    stable_steps: int = 4
    erosion_size: int = 1  # Sem erosão por padrão: não apagar defeitos pequenos.
    top_fraction: float = 0.01
    alpha: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if not self.windows or any(w < 3 or w % 2 != 1 for w in self.windows):
            raise ValueError("Janelas devem ser ímpares e >= 3.")
        if not (0 < self.pixel_quantile < 1 and 0 < self.alpha < 1 and 0 < self.top_fraction <= 1):
            raise ValueError("Quantil, alpha e fração do top inválidos.")
        if not (1 <= self.mebin_step <= 255 and self.stable_steps >= 2 and self.erosion_size >= 1):
            raise ValueError("Parâmetros MEBin inválidos.")
        if self.scale_floor_ratio <= 0 or self.contrast_threshold <= 0:
            raise ValueError("Piso e limiar de contraste devem ser positivos.")


def top_mean(array, fraction=0.01):
    values = np.asarray(array).ravel()
    if not len(values):
        return 0.0
    k = max(1, math.ceil(len(values)*fraction))
    return float(np.partition(values, len(values)-k)[-k:].mean())


def mebin_mask(array, step=4, stable_steps=4, erosion_size=1):
    """Estabilidade da contagem de componentes; empates favorecem limiar alto.

    Faixa por imagem é uma variante explícita. Erosão afeta a contagem, não a
    máscara final. Intervalos com zero componentes não são candidatos.
    """
    array = validate_heatmap(array)
    if not 1 <= step <= 255 or stable_steps < 2 or erosion_size < 1:
        raise ValueError("Parâmetros MEBin inválidos.")
    low, high = float(array.min()), float(array.max())
    if high == low:
        return np.zeros(array.shape, bool), high
    thresholds = low + np.arange(255, -1, -step)/255*(high-low)
    counts = []
    for threshold in thresholds:
        binary = (array > threshold).astype(np.uint8)
        if erosion_size > 1:
            binary = cv2.erode(binary, np.ones((erosion_size,erosion_size),np.uint8),
                               borderType=cv2.BORDER_CONSTANT,borderValue=0)
        counts.append(cv2.connectedComponents(binary, connectivity=8)[0]-1)
    best_length, chosen, index = 0, high, 0
    while index < len(counts):
        end = index+1
        while end < len(counts) and counts[end] == counts[index]:
            end += 1
        length = end-index
        if counts[index] > 0 and length >= stable_steps and length > best_length:
            chosen, best_length = float(thresholds[end-1]), length
        index = end
    return array > chosen, chosen


def local_contrast(array, windows=(9,17), scale_floor=1e-3):
    """Contraste positivo multiescala com escala robusta dos resíduos locais.

    Usa mediana de |A-mediana_local(A)|; aproximação robusta, não MAD exata
    calculada independentemente para cada janela. Não é probabilidade/z gaussiano.
    """
    array = validate_heatmap(array)
    if not np.isfinite(scale_floor) or scale_floor <= 0 or not windows or any(w < 3 or w % 2 != 1 for w in windows):
        raise ValueError("Piso positivo e janelas ímpares >=3 exigidos.")
    result = np.zeros_like(array)
    for window in windows:
        residual = array-median_filter(array,size=window,mode='reflect')
        scale = 1.4826*median_filter(np.abs(residual),size=window,mode='reflect')
        result = np.maximum(result, np.maximum(residual,0)/np.maximum(scale,scale_floor))
    return result


def compare_map(array, pixel_threshold, scale_floor, settings=None):
    settings = settings or MapSettings()
    array = validate_heatmap(array)
    if not np.isfinite(pixel_threshold):
        raise ValueError('Limiar pixel não finito.')
    mebin, threshold = mebin_mask(array,settings.mebin_step,settings.stable_steps,settings.erosion_size)
    contrast = local_contrast(array,settings.windows,scale_floor)
    masks = dict(absolute=array > pixel_threshold, mebin=mebin,
                 contrast=contrast > settings.contrast_threshold)
    # MEBin score combines candidate selection and original intensity.
    # A mask is a candidate, not an automatic image-level alarm.
    scores = dict(absolute=top_mean(array,settings.top_fraction),
                  mebin=top_mean(np.where(mebin,array,0),settings.top_fraction),
                  contrast=top_mean(contrast,settings.top_fraction))
    return masks, scores, contrast, threshold


def conformal_pvalues(reference_scores, scores):
    """Scores altos = anomalia; empates conservadores; ajuste finito +1."""
    reference = np.sort(np.asarray(reference_scores,dtype=float))
    values = np.asarray(scores,dtype=float)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Scores de calibração/teste devem ser finitos e não vazios.")
    return (1+len(reference)-np.searchsorted(reference,values,side='left'))/(len(reference)+1)


def _plot(row, array, masks, contrast, threshold, path):
    import matplotlib.pyplot as plt
    from PIL import Image

    fig,axes = plt.subplots(2,3,figsize=(14,8))
    try:
        with Image.open(row.image_path) as source:
            image = np.asarray(source.convert('RGB'))
        axes[0,0].imshow(image)
        axes[0,0].set_title('Entrada')
        heat = axes[0,1].imshow(array,cmap='inferno')
        fig.colorbar(heat,ax=axes[0,1])
        axes[0,1].set_title('Mapa: escala por figura, apenas visualização')
        im = axes[0,2].imshow(contrast,cmap='magma')
        fig.colorbar(im,ax=axes[0,2])
        axes[0,2].set_title('Contraste local (não é probabilidade)')
        for ax,(name,mask) in zip(axes[1],masks.items(),strict=True):
            ax.imshow(image,extent=(0,array.shape[1],array.shape[0],0))
            overlay = np.zeros((*array.shape,4))
            overlay[mask]=[1,0,0,.65]
            ax.imshow(overlay)
            ax.set_title(f'{name}: área={mask.mean():.2%}')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle(f'{row.source_id} | candidatos espaciais; limiar MEBin={threshold:.5g}')
        fig.tight_layout()
        fig.savefig(path,dpi=120)
    finally:
        plt.close(fig)


def _process(frame, output, pixel_threshold, scale_floor, settings, show_first):
    rows = []
    for index,row in enumerate(frame.itertuples(index=False)):
        array = validate_heatmap(np.load(row.map_path,allow_pickle=False))
        masks,scores,contrast,threshold = compare_map(array,pixel_threshold,scale_floor,settings)
        key = hashlib.sha256(f'{row.mode}:{row.image_path}'.encode()).hexdigest()[:20]
        target = output/str(row.mode)
        target.mkdir(parents=True,exist_ok=True)
        defect_path = getattr(row,'defect_path',None)
        defect = None
        if isinstance(defect_path,str) and defect_path:
            defect = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
            if defect is None or defect.shape != array.shape or not np.isin(defect,[0,1,255]).all():
                raise ValueError(f'Máscara de defeito inválida: {defect_path}')
            defect = defect > 0
        for method,mask in masks.items():
            mask_path = target/f'{key}_{method}.png'
            if not cv2.imwrite(str(mask_path),mask.astype(np.uint8)*255):
                raise OSError(mask_path)
            entry = dict(mode=row.mode,source_id=row.source_id,group_id=row.group_id,
                         split=row.split,y_true=int(row.y_true), image_path=row.image_path,
                         method=method,score=scores[method],area_fraction=float(mask.mean()),
                         components=int(cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)[0]-1),
                         pixel_threshold=threshold if method=='mebin' else
                         settings.contrast_threshold if method=='contrast' else pixel_threshold,
                         mask_path=str(mask_path),has_defect_mask=defect is not None)
            if defect is not None:
                tp=int((mask & defect).sum()); fp=int((mask & ~defect).sum()); fn=int((~mask & defect).sum())
                entry.update(pixel_tp=tp,pixel_fp=fp,pixel_fn=fn,
                             iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                             dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
            rows.append(entry)
        if index < show_first:
            _plot(row,array,masks,contrast,threshold,target/f'{key}_comparison.png')
        if (index+1)%20 == 0 or index+1 == len(frame):
            print(f'{row.mode}: {index+1}/{len(frame)} mapas comparados.')
    return pd.DataFrame(rows)


def explore_saved_maps(heatmap_table, output_root, settings=None, show_first=8):
    """Reutiliza mapas históricos sem alegar calibração independente.

    Limiar absolute é o pixel_threshold histórico; contraste usa piso por imagem.
    Não gera AP/FPR novos nem muda pred_label. Destino deve ser novo.
    """
    settings = settings or MapSettings()
    required={'mode','image_path','map_path','pixel_threshold','y_true'}
    if heatmap_table.empty or not required.issubset(heatmap_table):
        raise ValueError('Tabela histórica vazia ou incompleta.')
    output = Path(output_root)
    output.mkdir(parents=True,exist_ok=False)
    all_rows = []
    for mode,frame in heatmap_table.groupby('mode'):
        frame = frame.copy()
        frame['source_id']=frame.image_path
        frame['group_id']=frame.image_path
        frame['split']='exploratory'
        if 'is_case' in frame:
            frame=frame[frame.is_case.astype(bool)]
        for index,row in frame.iterrows():
            array=validate_heatmap(np.load(row.map_path,allow_pickle=False))
            floor=max(float(np.ptp(array))*settings.scale_floor_ratio,1e-8)
            all_rows.append(_process(frame.loc[[index]],output,float(row.pixel_threshold),floor,
                                     settings,1 if len(all_rows)<show_first else 0))
    table=pd.concat(all_rows,ignore_index=True)
    table.to_csv(output/'exploratory_metrics.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(settings=asdict(settings),
        status='exploratory only; historical test references; no calibrated decisions',
        mebin_variant='per-image min/max, not exact paper reproduction'),indent=2),encoding='utf-8')
    return table


def evaluate_map_methods(manifest, output_root, settings=None, show_first=8):
    """Manifesto completo de mapas: split val/test ou reference/calibration/test.

    Colunas: mode, split, source_id, group_id, y_true, image_path, map_path.
    Opcional defect_path (máscara binária alinhada ao mapa).
    val é separado por grupo entre referência de escala e calibração conformal.
    O usuário garante que essas normais não participaram do treino do checkpoint.
    Calibração usa máximo por grupo para não contar fotos correlatas como amostras
    independentes. Decisão por grupo; métricas também exportadas por imagem.
    """
    from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix

    settings=settings or MapSettings()
    frame=manifest.copy()
    required={'mode','split','source_id','group_id','y_true','image_path','map_path'}
    if not required.issubset(frame):
        raise ValueError(f'Faltam colunas: {required-set(frame.columns)}')
    if frame[list(required)].isna().any().any() or not frame.y_true.isin([0,1]).all():
        raise ValueError('Manifesto possui campos vazios ou rótulos inválidos.')
    if frame.duplicated(['mode','image_path']).any() or frame.duplicated(['mode','map_path']).any():
        raise ValueError('Entradas duplicadas.')
    if not set(frame.split).issubset({'val','reference','calibration','test'}):
        raise ValueError('Split deve ser val/test ou reference/calibration/test.')
    if (frame[frame.split!='test'].y_true != 0).any():
        raise ValueError('Referência e calibração devem conter somente normais.')
    for mode,subset in frame.groupby('mode'):
        if 'val' in set(subset.split):
            if set(subset.split)-{'val','test'}:
                raise ValueError('Não misture val com reference/calibration.')
            groups=sorted(subset.loc[subset.split=='val','group_id'].unique())
            if len(groups)<2:
                raise ValueError('Exige ao menos 2 grupos normais de validação.')
            np.random.default_rng(settings.seed).shuffle(groups)
            reference=set(groups[:max(1,len(groups)//2)])
            selected=(frame['mode']==mode)&(frame.split=='val')
            frame.loc[selected,'split']=np.where(frame.loc[selected,'group_id'].isin(reference),'reference','calibration')
    # Origem/grupo não pode atravessar splits; mesma origem não pode mudar rótulo.
    if (frame.groupby(['mode','group_id']).split.nunique()>1).any() or (frame.groupby(['mode','source_id']).split.nunique()>1).any():
        raise ValueError('Vazamento de origem/grupo entre splits.')
    if (frame.groupby(['mode','source_id']).y_true.nunique()>1).any():
        raise ValueError('Rótulo inconsistente para uma origem.')
    if 'source_sha256' in frame and (frame.groupby(['mode','source_sha256']).split.nunique()>1).any():
        raise ValueError('Conteúdo de origem duplicado entre splits.')
    output=Path(output_root)
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'input_manifest.csv',index=False)
    tables,metrics,source_decisions,group_decisions,configurations=[],[],[],[],{}
    for mode,subset in frame.groupby('mode'):
        if set(subset.split)!={'reference','calibration','test'}:
            raise ValueError(f'{mode}: faltam referência/calibração/teste.')
        reference=subset[subset.split=='reference']
        values=np.concatenate([validate_heatmap(np.load(p,allow_pickle=False)).ravel() for p in reference.map_path])
        threshold=float(np.quantile(values,settings.pixel_quantile))
        floor=max(float(np.quantile(values,.95)-np.quantile(values,.05))*settings.scale_floor_ratio,1e-8)
        configurations[str(mode)]=dict(pixel_threshold=threshold,scale_floor=floor)
        table=_process(subset,output,threshold,floor,settings,show_first)
        tables.append(table)
        sources=table.groupby(['mode','method','split','source_id','group_id','y_true'],as_index=False).score.max()
        groups=sources.groupby(['mode','method','split','group_id'],as_index=False).agg(score=('score','max'),y_true=('y_true','max'))
        for method,current in groups.groupby('method'):
            cal=current[current.split=='calibration'].score.to_numpy()
            if len(cal)<math.ceil(1/settings.alpha)-1:
                print(f'AVISO {mode}/{method}: {len(cal)} grupos; resolução conformal={1/(len(cal)+1):.4f}, alpha={settings.alpha}.')
            test=current[current.split=='test'].copy()
            test['p_value']=conformal_pvalues(cal,test.score)
            test['pred_label']=(test.p_value<=settings.alpha).astype(int)
            group_decisions.append(test)
            # Cada imagem recebe o alarme do seu grupo: mesma unidade da calibração.
            image_test=sources[(sources.method==method)&(sources.split=='test')].merge(
                test[['group_id','p_value','pred_label']],on='group_id',validate='many_to_one')
            source_decisions.append(image_test)
            for unit,data in [('group',test),('source',image_test)]:
                tn,fp,fn,tp=confusion_matrix(data.y_true,data.pred_label,labels=[0,1]).ravel()
                both=data.y_true.nunique()==2
                metrics.append(dict(mode=mode,method=method,unit=unit,n=len(data),n_calibration_groups=len(cal),
                    alpha=settings.alpha,AP=float(average_precision_score(data.y_true,data.score)) if both else np.nan,
                    AUROC=float(roc_auc_score(data.y_true,data.score)) if both else np.nan,
                    TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
                    recall=tp/(tp+fn) if tp+fn else np.nan,FPR=fp/(fp+tn) if fp+tn else np.nan,
                    precision=tp/(tp+fp) if tp+fp else 0.))
    table=pd.concat(tables,ignore_index=True)
    comparison=pd.DataFrame(metrics)
    table.to_csv(output/'map_metrics.csv',index=False)
    comparison.to_csv(output/'comparison.csv',index=False)
    pd.concat(source_decisions,ignore_index=True).to_csv(output/'source_decisions.csv',index=False)
    pd.concat(group_decisions,ignore_index=True).to_csv(output/'group_decisions.csv',index=False)
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    for mode,data in pd.concat(group_decisions,ignore_index=True).groupby('mode'):
        fig,ax=plt.subplots(figsize=(7,5))
        try:
            for method,subset in data.groupby('method'):
                if subset.y_true.nunique()<2:
                    continue
                precision,recall,thresholds=precision_recall_curve(subset.y_true,subset.score)
                pd.DataFrame(dict(precision=precision[:-1],recall=recall[:-1],threshold=thresholds)).to_csv(
                    output/f'{mode}_{method}_pr_curve.csv',index=False)
                ax.plot(recall,precision,label=method)
            ax.set(xlabel='Recall por grupo',ylabel='Precisão',xlim=(0,1),ylim=(0,1))
            ax.legend()
            fig.tight_layout()
            fig.savefig(output/f'{mode}_precision_recall.png',dpi=120)
        finally:
            plt.close(fig)
    if 'pixel_tp' in table:
        annotated=table[(table.split=='test') & table.has_defect_mask]
        if not annotated.empty:
            summary=annotated.groupby(['mode','method'])[['pixel_tp','pixel_fp','pixel_fn']].sum()
            denom=summary.pixel_tp+summary.pixel_fp+summary.pixel_fn
            summary['micro_iou']=summary.pixel_tp/denom.replace(0,np.nan)
            summary['annotated_inputs']=annotated.groupby(['mode','method']).size()
            summary.to_csv(output/'annotated_pixel_summary.csv')
    (output/'protocol.json').write_text(json.dumps(dict(settings=asdict(settings),modes=configurations,
        mebin_variant='per-image min/max; no erosion by default; not exact paper reproduction',
        decision='split-conformal; group maximum; ties conservative; all transforms fixed before calibration',
        assumptions='independent exchangeable normal groups; unseen calibration data; no pixel independence claim',
        masks='spatial candidates, not calibrated pixel decisions; optional annotations only for evaluation',
        sources=['https://github.com/HUST-SLOW/MEBin','https://arxiv.org/abs/2104.08279']),indent=2),encoding='utf-8')
    return comparison,table
