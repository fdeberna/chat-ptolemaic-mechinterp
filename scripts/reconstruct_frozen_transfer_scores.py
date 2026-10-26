"""Reconstruct saved style L6/L28 frozen transfer scores without fitting anything."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import rankdata


def metrics(y, prediction, score):
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(positive & (prediction == 1)))
    fp = int(np.sum(negative & (prediction == 1)))
    fn = int(np.sum(positive & (prediction == 0)))
    ranks = rankdata(score, method='average')
    n1, n0 = int(positive.sum()), int(negative.sum())
    return dict(accuracy=float(np.mean(y == prediction)),
                balanced_accuracy=float((np.mean(prediction[positive] == 1) + np.mean(prediction[negative] == 0))/2),
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                roc_auc=float((ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0)))


def main():
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = root/'results/comparisons/pilot_full/frozen_score_reconstruction_style_L6_L28'
    # A new directory protects both original results and prior reconstruction runs.
    out.mkdir(exist_ok=False)
    inputs = set()
    def source(path):
        inputs.add(path)
        return path
    reference = pd.read_csv(source(root/'results/comparisons/pilot_full/cross_model_probe_transfer_folds.csv'))
    layers_reference = pd.read_csv(source(root/'results/comparisons/pilot_full/cross_model_probe_transfer_layers.csv'))
    selected = reference[(reference.target_variable == 'style') & reference.layer.isin([6,28])]
    stores, folds = {}, {}
    for model in ('base','lora'):
        directory = root/f'results/activations/pilot_full/{model}'
        metadata = pd.read_csv(source(directory/'metadata.csv'))
        with np.load(source(directory/'activations.npz')) as arrays:
            activations = arrays['activations']
        assert len(metadata) == len(activations)
        stores[model] = metadata, activations
        folds[model] = pd.read_csv(source(root/f'results/probes/pilot_full/{model}/style/fold_metrics.csv'))
    columns = ['prompt_id','layer_index','style','template_family']
    assert stores['base'][0][columns].equals(stores['lora'][0][columns])
    records, comparisons = [], []
    for row in selected.itertuples(index=False):
        layer, fold = int(row.layer), int(row.fold)
        train, test = set(json.loads(row.train_family_ids)), set(json.loads(row.test_family_ids))
        assert not train & test
        cv = folds[row.source_model]
        cvrow = cv[(cv.layer_index == layer) & (cv.fold == fold)].iloc[0]
        assert set(json.loads(cvrow.train_template_families)) == train
        assert set(json.loads(cvrow.test_template_families)) == test
        mapping = json.loads(row.label_mapping)
        assert mapping == json.loads(cvrow.label_mapping) == {'modern':0,'premodern':1}
        path = source(root/f'results/probes/pilot_full/{row.source_model}/style/cv_coefficients/layer_{layer:03d}_fold_{fold:02d}.npz')
        metadata, activations = stores[row.target_model]
        layer_mask = metadata.layer_index.to_numpy() == layer
        layer_metadata = metadata.loc[layer_mask].reset_index(drop=True)
        assert layer_metadata.prompt_id.is_unique
        assert train | test == set(layer_metadata.template_family)
        test_mask = layer_metadata.template_family.isin(test).to_numpy()
        assert int(test_mask.sum()) == row.n_test
        x = activations[layer_mask][test_mask]
        with np.load(path) as probe:
            # Preserve NumPy/StandardScaler's in-place rounding at activation dtype.
            z = x.copy()
            z -= probe['scaler_mean']
            z /= probe['scaler_scale']
            decision = (z @ probe['coef'].T + probe['intercept']).reshape(-1)
        assert np.isfinite(decision).all()
        probability = expit(decision)
        prediction = (decision > 0).astype(int)
        test_metadata = layer_metadata.loc[test_mask]
        y = test_metadata['style'].map(mapping).to_numpy(dtype=int)
        result = metrics(y,prediction,probability)
        comparison = dict(layer=layer,fold=fold,source_model=row.source_model,target_model=row.target_model)
        for key,value in result.items():
            comparison[f'reconstructed_{key}'] = value
            comparison[f'original_{key}'] = float(getattr(row,key))
            comparison[f'delta_{key}'] = value-float(getattr(row,key))
        comparisons.append(comparison)
        for j, item in enumerate(test_metadata.itertuples(index=False)):
            records.append(dict(target_variable='style',layer=layer,source_model=row.source_model,
                target_model=row.target_model,fold=fold,prompt_id=item.prompt_id,
                template_family=item.template_family,label=item.style,y_true=int(y[j]),
                positive_class='premodern',decision_score=float(decision[j]),
                probability_positive=float(probability[j]),prediction=int(prediction[j]),
                decision_threshold=0.,probability_threshold=.5,
                coefficient_artifact=str(path.relative_to(root))))
    scores = pd.DataFrame(records)
    assert len(scores) == 960
    assert not scores.duplicated(['layer','source_model','target_model','prompt_id']).any()
    scores.to_csv(out/'out_of_fold_scores.csv',index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out/'fold_metric_comparison.csv',index=False)
    groupkeys = ['layer','source_model','target_model']
    aggregated = comparison.groupby(groupkeys)[[f'reconstructed_{m}' for m in ('accuracy','balanced_accuracy','f1','roc_auc')]].mean().reset_index()
    for metric in ('accuracy','balanced_accuracy','f1','roc_auc'):
        original = layers_reference[layers_reference.target_variable=='style'][groupkeys+[f'mean_{metric}']]
        aggregated = aggregated.merge(original,on=groupkeys,validate='one_to_one')
        aggregated[f'delta_mean_{metric}'] = aggregated[f'reconstructed_{metric}']-aggregated[f'mean_{metric}']
    aggregated.to_csv(out/'layer_metric_comparison.csv',index=False)
    inputs.add(Path(__file__).resolve())
    manifest = dict(method='Saved CV coefficients/scalers; frozen forward scoring only; no fitting or threshold selection',
        label_mapping={'modern':0,'premodern':1},layers=[6,28],scores=len(scores),fold_evaluations=len(comparison),
        elapsed_seconds=time.monotonic()-started,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        metric_definitions='Accuracy; mean class recall; positive-class F1; probability AUC via average-rank Mann-Whitney statistic. Fold means are unweighted, not pooled AUC.',
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    maximum = max(comparison[f'delta_{m}'].abs().max() for m in ('accuracy','balanced_accuracy','f1','roc_auc'))
    (out/'report.md').write_text(f'''# Frozen score reconstruction: style layers 6 and 28

Reconstructed all four source→target pairings, four recorded family folds, and 120 prompts per pairing/layer: 960 held-out scores, 32 fold evaluations. No model loading, refitting, threshold selection, or calibration.

Maximum absolute fold-metric discrepancy from original transfer CSV: {maximum:.17g}. See fold_metric_comparison.csv and layer_metric_comparison.csv for all differences. The source CV fold family lists and label mappings were checked against the transfer rows. NumPy scoring applies the saved source scaler and coefficients, preserving activation-dtype rounding during standardization; class 1 is premodern, with unchanged decision threshold 0 (probability 0.5).

out_of_fold_scores.csv contains decision scores, probabilities, predictions, labels, prompt/family IDs, fold IDs, source/target IDs, and coefficient paths. manifest.json records input/source hashes and versions. Original files remain unchanged. These are reconstructed scores, not recovered original transient scores; agreement of metrics supports compatibility but cannot prove coefficient identity with the original transfer fits or recover original score bit patterns.

AUC uses continuous probabilities and is evaluated within each fold; layer summaries are unweighted fold means. This reconstruction does not validate threshold recalibration or a universal additive score shift. No thresholds were optimized. Any future threshold selection must remain inside training/validation families, leaving outer test families untouched.

Reproduce from the project root (choose a fresh output directory in the script before repeating):
`/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/reconstruct_frozen_transfer_scores.py`
''')
    print(aggregated.to_string(index=False))
    print('maximum absolute fold-metric discrepancy:',maximum)
    print('outputs:',out)


if __name__ == '__main__':
    main()
