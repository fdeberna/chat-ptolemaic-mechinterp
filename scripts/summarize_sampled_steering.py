"""Export all retained sampled steering outputs and an unlabeled blinded review."""
import hashlib
import json
import copy
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steering/sampled_own_random_k01'
FIELDS=['stance_label','stance_supporting_excerpt','earth_rotation','earth_rotation_supporting_excerpt',
        'earth_revolution_around_sun','earth_revolution_supporting_excerpt','style_label','style_supporting_excerpt',
        'semantic_notes','full_continuation_repetition','full_continuation_incoherence',
        'full_continuation_topic_drift','full_continuation_truncation','quality_notes','reviewer_id']


def window(text):
    words=list(re.finditer(r'\S+',text))
    return (text if len(words)<=80 else text[:words[79].end()]),min(len(words),80),len(words)


def main():
    config=json.loads((OUT/'config.json').read_text());gate=json.loads((OUT/'initial_quality_gate.json').read_text())
    steered=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
    baselines=[json.loads(p.read_text()) for p in (OUT/'reused_baselines').glob('*.json')]
    assert len(baselines)==24
    expected=set();initial=set();passed={(g['model'],g['target']) for g in gate['groups'] if g['continue_grid']}
    for m in ('base','lora'):
        for t in config['targets']:
            for d in ('own','random'):
                for k in (-.1,.1):
                    initial.add((m,t,d,k,'daily_sky',0))
                    for p in config['prompts']:
                        for seed in (0,1,2):
                            key=(m,t,d,k,p['prompt_id'],seed)
                            if (m,t) in passed or key in initial:expected.add(key)
    actual={(r['model'],r['target'],r['direction_source'],r['k'],r['prompt_id'],r['seed']) for r in steered}
    assert actual==expected and len(actual)==len(steered)==16+44*len(passed)
    bybase={(r['model'],r['prompt_id'],r['seed']):r for r in baselines}
    assert len(bybase)==24
    checks={}
    for m in ('base','lora'):
        v=json.loads((OUT/'verification'/f'{m}.json').read_text());assert v['no_hook_matches_reused_baseline'] and v['zero_matches_no_hook']
        checks[m]=dict(fresh_no_hook_matches_source=True,zero_matches_no_hook=True,zero_hook_layers=list(v['zero_hooks']['hook_states']))
    gate_hash=hashlib.sha256((OUT/'initial_quality_gate.json').read_bytes()).hexdigest()
    for r in steered:
        state=r['hook_states'][r['target']];assert state['calls']==r['generated_tokens'] and state['prefill_calls']==1 and state['decode_calls']==r['generated_tokens']-1
        assert np.isfinite(state['ratios']).all()
        b=bybase[r['model'],r['prompt_id'],r['seed']];assert b['input_ids']==r['input_ids']
        for k,v in config['generation'].items():assert r['generation_config'][k]==b['generation_config'][k]==v
        if r['quality_gate_sha256'] is not None:assert r['quality_gate_sha256']==gate_hash
    rows=sorted(baselines+steered,key=lambda r:r['condition_id'])
    assert len({r['condition_id'] for r in rows})==len(rows)
    runtime_configs={}
    for r in rows:
        r['mode']='sampling'
        gc=copy.deepcopy(r['generation_config'])
        gc.update(return_dict_in_generate=True,output_logits=False,output_scores=False,max_length=len(r['input_ids'])+200)
        runtime_configs[r['condition_id']]=dict(effective_generation_config=gc,model_kwargs={'logits_to_keep':1},note='Includes actual output-control overrides and prompt-dependent max_length; max_new_tokens remains 200.')
    (OUT/'effective_runtime_configs.json').write_text(json.dumps(runtime_configs,indent=2))
    with (OUT/'generations.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(OUT/'generations.csv',index=False)
    review=[];key=[];rng=np.random.default_rng(20261027)
    for index in rng.permutation(len(rows)):
        r=rows[index];rid=rng.bytes(12).hex();excerpt,scored_words,total_words=window(r['continuation'])
        review.append(dict(review_id=rid,prompt_context=r['prompt_text'],scoring_window_first_80_words=excerpt,scored_word_count=scored_words,full_continuation=r['continuation'],**{field:'' for field in FIELDS}))
        b=bybase[r['model'],r['prompt_id'],r['seed']]
        key.append(dict(review_id=rid,condition_id=r['condition_id'],kind=r['kind'],model=r['model'],prompt_id=r['prompt_id'],seed=r['seed'],target=r.get('target'),layer=r.get('layer'),direction_source=r.get('direction_source'),k=r['k'],gamma=r['gamma'],baseline_condition_id=b['condition_id'],total_generated_words=total_words,stopping_reason=r['stopping_reason'],generated_tokens=r['generated_tokens']))
    review_path=OUT/'blinded_review.csv'
    if review_path.exists():
        old=pd.read_csv(review_path,keep_default_na=False)
        assert all((old[f]=='').all() for f in FIELDS),'Preserve existing human labels; refuse review overwrite'
    pd.DataFrame(review).to_csv(review_path,index=False)
    with (OUT/'blinded_review.jsonl').open('w') as f:
        for r in review:f.write(json.dumps(r)+'\n')
    pd.DataFrame(key).to_csv(OUT/'condition_key.csv',index=False)
    instructions='''# Review delivery

Provide a separate evaluator ONLY rubric.md and blinded_review.csv/jsonl.
Do not provide condition_key.csv, initial examples, configurations, other result
files, or this generating-session history. Review IDs are shuffled opaque IDs.
The generating session was NOT blinded and did NOT supply semantic labels.
No genuinely separate evaluator was used; all review fields remain blank.
The initial unblinded quality gate is not a stance/style evaluation.
Support every provisional semantic label with excerpts from the first 80-word
window. Full-continuation quality is a separate assessment. Retain unclear and
poor cases. Three seeds are not independent prompts. Do not compute semantic
counts or paired effects from blank fields, token changes, or probe scores.
'''
    (OUT/'review_instructions.md').write_text(instructions)
    diagnostics=[]
    for r in steered:
        s=r['hook_states'][r['target']]
        diagnostics.append(dict(condition_id=r['condition_id'],model=r['model'],target=r['target'],direction_source=r['direction_source'],prompt_id=r['prompt_id'],seed=r['seed'],k=r['k'],gamma=r['gamma'],identical_tokens_to_baseline=r['identical_tokens_to_baseline'],prefill_perturbation_ratio=s['ratios'][0],min_actual_ratio=min(s['ratios']),median_actual_ratio=float(np.median(s['ratios'])),max_actual_ratio=max(s['ratios'])))
    pd.DataFrame(diagnostics).to_csv(OUT/'intervention_diagnostics.csv',index=False)
    resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
    metrics=dict(new_steered=len(steered),reused_baselines=24,total_review_rows=len(rows),planned_maximum_steered=192,stopped_families=[dict(model=g['model'],target=g['target']) for g in gate['groups'] if not g['continue_grid']],
        verification=checks,semantic_labels_supplied=False,independent_evaluator_used=False,identical_tokens_to_baseline=sum(r['identical_tokens_to_baseline'] for r in steered),
        new_generation_seconds=sum(r['runtime_seconds_new'] for r in steered),session_elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resources),
        failures=len(list(OUT.glob('failure_*.json'))),source_baselines_unchanged=True)
    (OUT/'verification_summary.json').write_text(json.dumps(metrics,indent=2));shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
    report=['# Bounded sampled own-direction/random steering experiment','',
      f'Completed {len(steered)} new steered continuations plus all 24 reused sampled baselines; {len(rows)} review rows. Maximum authorized grid is 192 new steered continuations. All 16 initial quality cases are retained, including poor cases; stopped families are not silently removed. No extra prompts, seeds, layers, strengths, cross-model directions, tuning, fitting or compatibility extraction was run.',
      '## Frozen design and checks','',
      'Latest four development prompts are unchanged: daily_sky, annual_sun, retrograde, lunar_appearance. Both generating models use their own raw full-data unit probe direction or the previously seeded random unit vector. Stance saved L16 is block15 residual output, style saved L6 is block5 residual output. Gamma=k*a_L with shared saved-base median scales L16=73.83589058, L6=31.03375833. k=±0.1; k=0 is the shared baseline. Intercept/thresholds are excluded. Prefill modifies only final prompt token; cached decoding modifies each newly processed token. Hooks clone outputs and are removed on context exit.',
      'Generation uses model.generate(), sampling temperature0.8/top_p0.9/top_k50, seeds0/1/2 independently reset for Python/NumPy/Torch/CUDA, max_new_tokens200, repetition_penalty1, no_repeat_ngram_size0, EOS/PAD151643 and KV cache. NF4 double quantization, FP16 base compute/storage and FP32 LoRA parameters, SDPA, one instance on cuda:0, no offload, frozen parameters and inference/eval mode are preserved. CUDA_LAUNCH_BLOCKING=1 matches successful baseline execution; timing includes synchronization overhead.',
      'All 24 baseline records pass matching prompt text/input IDs, full effective sampling configuration, model revision, adapter/tokenizer/source hashes, versions and seeds. Fresh first-prompt/seed0 no-hook samples reproduce the reused baseline exactly for BOTH models. Simultaneous gamma0 hooks at L6 and L16 also produce identical tokens to fresh no-hook samples for BOTH models. Verification decodes are stored separately and do not count as new steered grid conditions. Every steered record has verified one-prefill-plus-single-token-decode intervention counts and finite actual perturbation/residual ratios.',
      '## Initial quality gate','',
      'First fixed prompt daily_sky and seed0 were generated across both targets, own/random directions and both signs before continuation. All 16 full outputs and matching baselines appear in initial_examples.md. Quality decisions are unblinded development checks about substantial NEW deterioration relative to the matching baseline. Baseline defects alone do not constitute intervention damage.',
      *[f"- {g['model']} / {g['target']}: {'continued full family' if g['continue_grid'] else 'stopped after initial cases'}. {g['reason']}" for g in gate['groups']],
      '## Review and interpretation','',
      'rubric.md was frozen and hashed before inference. Both stance/style are to be scored for EVERY continuation using the first80 generated whitespace-delimited words; prompt context is not semantic evidence. First substantive answer, selected-quiz-answer/quotation rules, independent rotation/revolution fields, and separate full-continuation quality fields are preserved. A factual error does not imply premodern style; Earth rotation alone does not imply heliocentrism.',
      'No genuinely separate evaluator was used. Labels and supporting-excerpt fields are BLANK. The generating session is not claimed blinded. blinded_review.csv/jsonl contains only opaque shuffled IDs, prompt context, scoring window, full continuation and blank review fields; condition_key.csv is separate and must be withheld. Semantic counts, paired effects, unclear rates and quality rates are not computed without independent provisional labels. Three seeds are not independent prompts. Changed wording or probe scores do not demonstrate semantic steering.',
      '## Resources and files','',
      f"New steered generation compute {metrics['new_generation_seconds']/60:.2f} minutes; summed process elapsed {metrics['session_elapsed_seconds']/60:.2f} minutes including verification/loading (not the review gap). Peak GPU allocation/reservation {metrics['peak_allocated_bytes']/2**30:.2f}/{metrics['peak_reserved_bytes']/2**30:.2f} GiB; host RSS {metrics['peak_host_ram_bytes']/2**30:.2f} GiB. Allocator peaks exclude driver memory; RSS includes file-backed loading pages. Failure records {metrics['failures']}. {metrics['identical_tokens_to_baseline']}/{len(steered)} steered token lists equal their matched baseline; this is a technical diagnostic, not a semantic score.",
      'conditions/*.json commit each new generation atomically by unique condition ID; reused_baselines/*.json retain all baseline text with original IDs, paths and hashes. generations.jsonl/CSV contains the complete retained cohort. config.json, effective_generation_configs.json, direction_audit.json, baseline_source_index.json, verification/, rubric.md, initial_quality_gate.json, reconstruction/ and resource logs retain provenance and checks. intervention_diagnostics.csv records actual perturbation ratios. Original unquantized-probe/NF4-generation comparability remains pending and was not tested here.',
      '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/run_sampled_steering.py --stage initial\n# Read all initial examples and retain/create initial_quality_gate.json.\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/run_sampled_steering.py --stage full\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/summarize_sampled_steering.py\n```',
      'GPU commands require approved access outside the GPU-blocking sandbox. Compatible completed conditions and saved decisions are reused. All previous result directories are preserved.']
    (OUT/'report.md').write_text('\n\n'.join(report).replace('\n\n\n\n','\n\n')+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
