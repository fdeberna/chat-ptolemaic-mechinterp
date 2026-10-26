"""Family-held-out base→LoRA threshold calibration, style L6/L28 only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from reconstruct_frozen_transfer_scores import metrics


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def select_threshold(y,s):
    unique=np.unique(s)
    # Strict score > threshold: midpoints plus boundary representatives cover
    # every classification partition, including all-positive/all-negative.
    candidates=np.unique(np.r_[0.,np.nextafter(unique[0],-np.inf),
        unique[:-1]+np.diff(unique)/2,unique[-1]])
    values=np.array([metrics(y,(s>t).astype(int),s)['balanced_accuracy'] for t in candidates])
    best=np.flatnonzero(values==values.max())
    chosen=min(best,key=lambda i:(abs(candidates[i]),candidates[i]))
    return float(candidates[chosen]),float(values[chosen]),pd.DataFrame({'threshold':candidates,'calibration_balanced_accuracy':values})


def main():
    start=time.monotonic();root=Path(__file__).resolve().parents[1]
    out=root/'results/comparisons/pilot_full/threshold_calibration_style_L6_L28_base_to_lora'
    out.mkdir(exist_ok=False);(out/'probes').mkdir();(out/'threshold_candidates').mkdir()
    sources={Path(__file__).resolve(),root/'scripts/reconstruct_frozen_transfer_scores.py'}
    def read(path):sources.add(path);return pd.read_csv(path)
    folds=read(root/'results/comparisons/pilot_full/cross_model_probe_transfer_folds.csv')
    selected=folds[(folds.target_variable=='style')&(folds.source_model=='base')&(folds.target_model=='lora')&folds.layer.isin([6,28])]
    stores={}
    for model in ('base','lora'):
        directory=root/f'results/activations/pilot_full/{model}'
        m=read(directory/'metadata.csv');path=directory/'activations.npz';sources.add(path)
        with np.load(path) as a:x=a['activations']
        stores[model]=(m,x)
    cols=['prompt_id','layer_index','style','template_family']
    assert stores['base'][0][cols].equals(stores['lora'][0][cols])
    rows=[];scores=[];splits=[]
    for row in selected.itertuples(index=False):
        layer,fold=int(row.layer),int(row.fold)
        train=set(json.loads(row.train_family_ids));test=set(json.loads(row.test_family_ids))
        ranked=sorted(train,key=lambda family:hashlib.sha256(f'0|{fold}|{family}'.encode()).hexdigest())
        calibration=set(ranked[:int(np.ceil(.25*len(train)))]);fit=train-calibration
        assert not fit&calibration and not train&test
        mapping=json.loads(row.label_mapping);assert mapping=={'modern':0,'premodern':1}
        bm,bx=stores['base'];lm,lx=stores['lora'];mask=bm.layer_index.to_numpy()==layer
        m=bm.loc[mask].reset_index(drop=True);base=bx[mask];lora=lx[mask]
        y=m['style'].map(mapping).to_numpy(dtype=int)
        fitmask=m.template_family.isin(fit).to_numpy();cmask=m.template_family.isin(calibration).to_numpy();tmask=m.template_family.isin(test).to_numpy()
        assert tmask.sum()==row.n_test and np.unique(y[fitmask]).size==np.unique(y[cmask]).size==np.unique(y[tmask]).size==2
        scaler=StandardScaler();z=scaler.fit_transform(base[fitmask])
        probe=LogisticRegression(solver='liblinear',C=1.,class_weight='balanced',random_state=0,max_iter=1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');probe.fit(z,y[fitmask])
        assert not caught, [str(w.message) for w in caught]
        cs=probe.decision_function(scaler.transform(lora[cmask]));threshold,calba,candidates=select_threshold(y[cmask],cs)
        ts=probe.decision_function(scaler.transform(lora[tmask]));prob=probe.predict_proba(scaler.transform(lora[tmask]))[:,1]
        original=metrics(y[tmask],(ts>0).astype(int),prob);calibrated=metrics(y[tmask],(ts>threshold).astype(int),prob)
        assert original['roc_auc']==calibrated['roc_auc']
        info=dict(layer=layer,fold=fold,fit_families=sorted(fit),calibration_families=sorted(calibration),test_families=sorted(test))
        splits.append(info)
        key=f'L{layer:02d}_fold{fold}'
        candidates.to_csv(out/'threshold_candidates'/f'{key}.csv',index=False)
        np.savez_compressed(out/'probes'/f'{key}.npz',coef=probe.coef_,intercept=probe.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,classes=probe.classes_,threshold=np.asarray(threshold),split_json=np.asarray(json.dumps(info)),label_mapping=np.asarray(json.dumps(mapping)))
        result=dict(layer=layer,fold=fold,n_fit=int(fitmask.sum()),n_calibration=int(cmask.sum()),n_test=int(tmask.sum()),threshold=threshold,calibration_balanced_accuracy=calba,zero_balanced_accuracy=original['balanced_accuracy'],calibrated_balanced_accuracy=calibrated['balanced_accuracy'],paired_difference=calibrated['balanced_accuracy']-original['balanced_accuracy'],zero_auc=original['roc_auc'],calibrated_auc=calibrated['roc_auc'],fit_iterations=int(probe.n_iter_[0]))
        rows.append(result)
        for partition,pmask,decision in [('calibration',cmask,cs),('outer_test',tmask,ts)]:
            for item,s,label in zip(m.loc[pmask].itertuples(index=False),decision,y[pmask]):
                scores.append(dict(layer=layer,fold=fold,partition=partition,prompt_id=item.prompt_id,template_family=item.template_family,y_true=int(label),decision_score=float(s),prediction_zero=int(s>0),prediction_calibrated=int(s>threshold),threshold=threshold))
    table=pd.DataFrame(rows);table.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    summary=table.groupby('layer')[['zero_balanced_accuracy','calibrated_balanced_accuracy','paired_difference','zero_auc','calibrated_auc']].mean().reset_index()
    summary.to_csv(out/'layer_metrics.csv',index=False)
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    config=dict(source='base',target='lora',label_mapping={'modern':0,'premodern':1},calibration_rule='ceil(25% of outer training families), sorted by SHA256 of 0|fold|family; identical across layers',threshold_rule='maximize calibration balanced accuracy; ties closest to zero then lower threshold; predict 1 iff score > threshold',probe=dict(solver='liblinear',C=1.,class_weight='balanced',random_state=0,max_iter=1000,tol=1e-4,fit_intercept=True,intercept_scaling=1.),sklearn=sklearn.__version__,numpy=np.__version__,elapsed_seconds=time.monotonic()-start,sources={str(p.relative_to(root)):digest(p) for p in sorted(sources)})
    (out/'manifest.json').write_text(json.dumps(config,indent=2))
    report='''# Nested family threshold calibration: style, base→LoRA

Outer test-family assignments are unchanged. In each outer training partition, 25% of families (rounded up) are deterministically reserved for target-domain calibration, identically across layers. Source scaler and probe fitting exclude calibration AND test families. Only the scalar threshold is selected on LoRA calibration scores. No refit follows calibration.

Probe settings match the existing transfer implementation: StandardScaler, liblinear logistic regression C=1, balanced training-class weights, seed 0, max_iter=1000, default tol=1e-4, penalized intercept with intercept_scaling=1. Positive class is premodern. Zero and calibrated thresholds use the same newly fitted probe and exactly the same outer test rows. Prior full-outer-training probe results are not the baseline here.

Threshold candidates cover all distinct calibration-score partitions plus zero; ties favor the threshold closest to zero, then the lower threshold. Candidates and split memberships are saved. AUC uses unchanged continuous probabilities, so it is exactly identical before/after thresholding for each fitted probe. Layer results are unweighted means of four fold metrics, not pooled AUC.

## Layer means

'''+summary.to_string(index=False)+'\n\n## Fold results\n\n'+table.to_string(index=False)+'''

Eight CPU fits only; no model loading or extraction. All fits completed without warnings. Source hashes, configuration, scores, probe/scaler parameters, threshold candidates and family membership are retained. These results describe one prespecified calibration split per outer fold; they do not establish threshold stability across alternative calibration partitions or statistical significance. The four fold differences are paired but their training sets overlap.

Reproduce with a fresh output directory using:
`PYTHONPATH=/tmp/probe-threshold-deps /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/calibrate_style_transfer_thresholds.py`
'''
    (out/'report.md').write_text(report)
    print(summary.to_string(index=False));print(table.to_string(index=False));print(out)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
