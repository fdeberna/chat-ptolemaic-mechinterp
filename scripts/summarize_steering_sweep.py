"""Summarize saved strength sweep and prefix diagnostics; no model inference."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steering/strength_sweep_003_01'


def main():
    config=json.loads((OUT/'config.json').read_text());gate=json.loads((OUT/'quality_gate.json').read_text())
    records=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
    expected=6+48+12*sum(g['proceed_to_01'] for g in gate['groups'])
    assert len(records)==expected and len({r['condition_id'] for r in records})==expected
    baseline={(r['model'],r['prompt_id']):r for r in records if r['k']==0}
    rows=[];tops=[]
    for r in records:
        assert r['run_identifier']==config['run_identifier']
        path=OUT/'prefill_logits'/f'{r["condition_id"]}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['prefill_logits_sha256']
        if not r['k']:continue
        b=baseline[r['model'],r['prompt_id']];assert r['prefix_input_ids']==b['prefix_input_ids']
        with np.load(path) as f:q=f['logits'].astype(np.float64)
        with np.load(OUT/'prefill_logits'/f'{b["condition_id"]}.npz') as f:p=f['logits'].astype(np.float64)
        lp=p-logsumexp(p);lq=q-logsumexp(q);kl=float(np.exp(lp)@(lp-lq))
        assert abs(kl-r['kl_unsteered_to_steered_nats'])<1e-12
        assert (int(p.argmax())!=int(q.argmax()))==r['argmax_changed']
        h=r['hook'];assert h['calls']==len(r['token_ids']) and h['prefill_calls']==1 and h['decode_calls']==len(r['token_ids'])-1
        assert np.isfinite(h['ratios']).all()
        row={k:r[k] for k in ['condition_id','model','target','layer','direction_source','prompt_id','k','gamma','activation_scale','identical_to_baseline','argmax_changed','kl_unsteered_to_steered_nats','actual_perturbation_ratio_min','actual_perturbation_ratio_median','actual_perturbation_ratio_max','generated_tokens','runtime_seconds']}
        row['abs_k']=abs(r['k']);row['prefill_perturbation_ratio']=h['ratios'][0];rows.append(row)
        for token in r['top_token_probabilities']:tops.append(dict(condition_id=r['condition_id'],**token))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(tops).to_csv(OUT/'top_token_probabilities.csv',index=False)
    summary=frame.groupby(['model','target','direction_source','k'],sort=True).agg(conditions=('condition_id','size'),fraction_identical=('identical_to_baseline','mean'),argmax_changes=('argmax_changed','sum'),median_kl_nats=('kl_unsteered_to_steered_nats','median'),max_kl_nats=('kl_unsteered_to_steered_nats','max'),median_prefill_ratio=('prefill_perturbation_ratio','median'),min_actual_ratio=('actual_perturbation_ratio_min','min'),max_actual_ratio=('actual_perturbation_ratio_max','max')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    stage=frame.groupby('abs_k').agg(conditions=('condition_id','size'),identical=('identical_to_baseline','sum'),fraction_identical=('identical_to_baseline','mean'),argmax_changes=('argmax_changed','sum'),min_kl=('kl_unsteered_to_steered_nats','min'),median_kl=('kl_unsteered_to_steered_nats','median'),max_kl=('kl_unsteered_to_steered_nats','max'),min_ratio=('actual_perturbation_ratio_min','min'),max_ratio=('actual_perturbation_ratio_max','max')).reset_index()
    stage.to_csv(OUT/'stage_summary.csv',index=False)
    examples=['# Baseline / negative / positive continuations for every prompt','', 'All six model/prompt combinations are shown. For each target use the largest strength actually tested, own trained direction; no selection by apparent behavioral effect. Full random-control and weaker-strength outputs remain in generations.jsonl/CSV.','']
    for model in ['base','lora']:
        for prompt in config['prompts']:
            pid=prompt['prompt_id'];b=baseline[model,pid]
            examples += [f'## {model}: {pid}', '',prompt['prompt_text'],'','### Baseline', '',b['continuation'],'']
            for target in ['stance','style']:
                cases=[r for r in records if r['model']==model and r['prompt_id']==pid and r['target']==target and r['direction_source']=='own']
                largest=max(abs(r['k']) for r in cases)
                for k in [-largest,largest]:
                    r=next(r for r in cases if r['k']==k)
                    examples += [f'### {target}, k={k:g}, gamma={r["gamma"]:.6g}, ID={r["condition_id"]}', '',r['continuation'],'']
    (OUT/'examples.md').write_text('\n'.join(examples))
    resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
    verification=dict(completed=expected,baselines=6,steered=len(frame),prefix_KL_independently_recomputed=True,hook_counts_verified=True,
        elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),generation_compute_seconds=sum(r['runtime_seconds'] for r in records),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resources),
        token_cap_count=sum(r['generated_tokens']==96 for r in records),failures=len(list(OUT.glob('failure_*.json'))))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2));shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
    review=json.loads((OUT/'quality_observations.json').read_text())
    report=['# Steering strength sweep: development pilot','',
        f'Completed {expected} conditions: six fresh shared baselines and {len(frame)} steered generations. Three prompts were fixed in config.json/prompts.csv before inference: daily sunrise/sunset, lunar phases, seasonal daylight. Their wording requests explanations and contains no arithmetic task. The base model sometimes introduces arithmetic anyway. All previous results remain unchanged.','',
        '## Method and quality gate','',
        'Reused the exact validated residual hook (hash checked), layer mapping (stance block 15 output / saved L16; style block 5 output / saved L6), final-prompt-token plus cached-new-token policy, packed NF4/FP16 SDPA setup, adapter, tokenizer, base revision, own raw full-data probe unit directions and seeded random controls. No cross-model directions, refitting, extraction, dependency changes or additional layers. Shared a_L values remain L16=73.83589058 and L6=31.03375833; gamma=k*a_L. All parameters are frozen on cuda:0 with no offloading, one model instance per sequential stage process. Intercepts and thresholds are not steering vectors.',
        'Six baselines are fresh because the new prompt texts do not match earlier conditions. Each is reused unchanged across targets/directions/strengths in this run. Original pilot_full probes were fitted on non-quantized activations; the generation arithmetic remains the established quantized pilot setup. This numerical mismatch remains a limitation.',
        'All 48 ±0.03 continuations were read in full (identical strings grouped with all condition IDs) before quality_gate.json was committed and before any ±0.1 generation. Quality review concerns coherence, relevance and repetition, not stance/style classification. Gates are per generating model/target and cover both own and random directions. These are informal development judgments, not validated independent ratings.',
        *[f"- {g['model']} / {g['target']}: {'proceeded' if g['proceed_to_01'] else 'stopped escalation'}. {g['reason']}" for g in gate['groups']],
        '', '## Distribution changes and output identity','',
        'KL is KL(p_unsteered || p_steered) in nats, evaluated from first-forward logits on the identical original prompt token IDs. Unsteered means the same generating model with no hook, including active LoRA for LoRA comparisons. Float64 log-softmax/norm calculations are used. Both top-five token lists and their union probabilities are saved; argmax flags compare these same-prefix distributions, not later diverged generation prefixes. Full FP32-valued logits are retained in per-condition NPZ files. An independent CPU logsumexp calculation reproduced every KL value.',
        stage.to_string(index=False),
        'summary.csv separates model, target, direction and signed k; metrics.csv retains every prompt/condition; top_token_probabilities.csv retains token IDs/text and both probabilities. Actual perturbation/residual norm ratios cover all processed tokens; prefill ratios are separately reported. Ratios vary because current residual norms differ from the fixed saved-base median scale and updates round into FP16.',
        '## Quality observations','',*[f'- {x}' for x in review['observations']],
        'Changed tokens, KL or argmax changes do not establish successful stance/style steering. Random controls can also change outputs. No probe scores are used as behavioral evidence, and no automated stance/style labels are supplied. One random control per model/layer and three development prompts do not establish causality or generalization. No strengths beyond ±0.1 were run.',
        '## Resources and files','',
        f"Summed stage elapsed time {verification['elapsed_seconds']/60:.2f} minutes; generation compute {verification['generation_compute_seconds']/60:.2f} minutes. Peak GPU allocation/reservation {verification['peak_allocated_bytes']/2**30:.2f}/{verification['peak_reserved_bytes']/2**30:.2f} GiB; peak host RSS {verification['peak_host_ram_bytes']/2**30:.2f} GiB. Allocator peaks exclude driver memory; RSS includes loading/file-backed memory. {verification['token_cap_count']}/{expected} outputs reached the 96-token cap. Failure records: {verification['failures']}.",
        'conditions/*.json and prefill_logits/*.npz are committed incrementally with unique IDs and hashes. Compatible conditions are skipped on resumption. Config, source hashes, code snapshots, low-strength quality gate and resource session logs are retained. generations.jsonl/CSV contain all continuations; examples.md contains baseline/negative/positive examples for every model/prompt, both targets at their largest tested strengths.',
        '```bash\nPYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/sweep_activation_steering.py --stage low\n# Read all low-strength continuations; preserve/create quality_gate.json before high stage.\nPYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/sweep_activation_steering.py --stage high\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/summarize_steering_sweep.py\n```',
        'GPU stage commands require access outside the GPU-blocking sandbox. Existing recorded gate decisions and completed conditions support exact resumption. No open-ended strength escalation was performed.']
    (OUT/'report.md').write_text('\n\n'.join(report).replace('\n\n\n\n','\n\n')+'\n')
    print(stage.to_string(index=False));print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
