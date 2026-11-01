"""Verify the controlled repeat and export unlabeled blinded review artifacts."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np
import pandas as pd
from summarize_sampled_steering import FIELDS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/steering/reproduced_protocol_stance_strength_sweep'

def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());selection=cfg['selection'];preset=json.loads(Path(cfg['preset_path']).read_text())
 assert hashlib.sha256(Path(cfg['preset_path']).read_bytes()).hexdigest()==cfg['preset_sha256']
 new=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
 rows=json.loads((OUT/'reused_records.json').read_text())+new
 gate=json.loads((OUT/'quality_gate.json').read_text())
 expected={(m,p,s,d,k) for m in ['base','lora'] for p in selection['selected_prompt_ids'] for s in [0,1,2] for d,k in [(None,0.),('own',-.1),('own',.1),('random',-.1),('random',.1),('own',-.3),('own',.3),('random',-.3),('random',.3)]+([('own',-.5),('own',.5),('random',-.5),('random',.5)] if gate['models'][m]['proceed'] else [])}
 assert len(rows)==len(expected) and {(r['model'],r['prompt_id'],r['sample_id'],r['direction_source'],r['k']) for r in rows}==expected
 baselines={(r['model'],r['prompt_id'],r['sample_id']):r for r in rows if r['kind']=='baseline'};assert len(baselines)==12 and all(r['historical_exact_match'] for r in baselines.values())
 metrics=[]
 for r in rows:
  if abs(r['k'])>.1:assert r['run_identifier']==cfg['run_identifier']
  for k in ['temperature','top_p','top_k','max_new_tokens','repetition_penalty','no_repeat_ngram_size','do_sample','num_return_sequences']:assert r['generation_config'][k]==preset['generation_config'][k]
  if r['hook_state']:
   h=r['hook_state'];assert h['calls']==r['generated_tokens'] and h['prefill_calls']==1 and h['decode_calls']==r['generated_tokens']-1
   assert np.isfinite(h['ratios']).all()
   b=baselines[r['model'],r['prompt_id'],r['sample_id']]
   metrics.append(dict(condition_id=r['condition_id'],model=r['model'],prompt_id=r['prompt_id'],sample_id=r['sample_id'],sample_seed=r['sample_seed'],direction_source=r['direction_source'],k=r['k'],gamma=r['gamma'],baseline_id=b['condition_id'],identical_tokens_to_baseline=r['generated_token_ids']==b['generated_token_ids'],prefill_ratio=h['ratios'][0],min_ratio=min(h['ratios']),median_ratio=float(np.median(h['ratios'])),max_ratio=max(h['ratios'])))
 pd.DataFrame(metrics).to_csv(OUT/'intervention_diagnostics.csv',index=False)
 with (OUT/'generations.jsonl').open('w') as f:
  for r in sorted(rows,key=lambda r:r['condition_id']):f.write(json.dumps(r)+'\n')
 pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(OUT/'generations.csv',index=False)
 rng=np.random.default_rng(20261029);review=[];key=[];ordered=sorted(rows,key=lambda r:r['condition_id'])
 for index in rng.permutation(len(rows)):
  r=ordered[index];words=list(re.finditer(r'\S+',r['continuation']));window=r['continuation'] if len(words)<=80 else r['continuation'][:words[79].end()];rid=rng.bytes(12).hex()
  review.append(dict(review_id=rid,prompt_context=r['prompt_text'],scoring_window_first_80_words=window,scored_word_count=min(80,len(words)),full_continuation=r['continuation'],**{f:'' for f in FIELDS}))
  key.append(dict(review_id=rid,condition_id=r['condition_id'],kind=r['kind'],model=r['model'],prompt_id=r['prompt_id'],sample_id=r['sample_id'],sample_seed=r['sample_seed'],target='stance' if r['kind']=='steered' else None,layer=r['layer'],direction_source=r['direction_source'],k=r['k'],gamma=r['gamma'],baseline_id=baselines[r['model'],r['prompt_id'],r['sample_id']]['condition_id']))
 if (OUT/'blinded_review.csv').exists():
  old=pd.read_csv(OUT/'blinded_review.csv',keep_default_na=False);assert all((old[f]=='').all() for f in FIELDS),'Preserve existing annotations'
 pd.DataFrame(review).to_csv(OUT/'blinded_review.csv',index=False);pd.DataFrame(key).to_csv(OUT/'condition_key.csv',index=False)
 with (OUT/'blinded_review.jsonl').open('w') as f:
  for r in review:f.write(json.dumps(r)+'\n')
 lines=['# Controlled repeat: every baseline/steered continuation','', 'Development outputs; no semantic effects inferred from token changes. All cases retained.','']
 for m in ['base','lora']:
  for pid in selection['selected_prompt_ids']:
   for s in [0,1,2]:
    b=baselines[m,pid,s];lines += [f'## {m} / {pid} / sample{s}, seed{b["sample_seed"]}', '',b['prompt_text'],'','### Baseline','',b['continuation'],'']
    for d in ['own','random']:
     for k in [-.1,.1,-.3,.3]+([-.5,.5] if gate['models'][m]['proceed'] else []):
      r=next(r for r in rows if (r['model'],r['prompt_id'],r['sample_id'],r['direction_source'],r['k'])==(m,pid,s,d,k));lines += [f'### {d}, k={k:g}, gamma={r["gamma"]:.6g}', '',r['continuation'],'']
 (OUT/'examples.md').write_text('\n'.join(lines))
 resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
 result=dict(total_conditions=len(rows),reused_conditions=60,new_steered=len(new),elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resources),failures=len(list(OUT.glob('failure_*.json'))),identical_new_to_baseline=sum(r['identical_tokens_to_baseline'] for r in metrics if abs(r['k'])>.1))
 (OUT/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
 summary=pd.DataFrame(metrics).groupby(['model','direction_source','k']).agg(cases=('condition_id','size'),identical_fraction=('identical_tokens_to_baseline','mean'),median_prefill_ratio=('prefill_ratio','median'),median_token_ratio=('median_ratio','median'),max_token_ratio=('max_ratio','max')).reset_index()
 summary.to_csv(OUT/'condition_summary.csv',index=False)
 report=['# Bounded reproduced-protocol stance strength sweep','',f"Completed {len(new)} new steered generations; reused 12 baselines and 48 ±0.1 outputs without regeneration. Total {len(rows)} records. All expected conditions occur exactly once.",'',
 'Two development prompts questions_012 (retrograde) and astro_024 (yearly solar circuit); both models, original sample IDs 0/1/2 with their SHA256-derived RNG seeds. Saved L16 / block15 residual output, own raw unit directions and unchanged seeded random controls. Helio-positive own direction checked against raw probe coefficients; no intercept or threshold injected. Final prompt token then each cached newly processed token. Shared scale a_L=73.83589057971676; gamma=k*a_L. ±0.3 first, then ±0.5 only after quality review. No other settings or strengths tested.','',
 '## Quality gate','',json.dumps(gate['models'],indent=2),'',
 'quality_review.csv records unblinded development observations for every ±0.3 continuation relative to its matched baseline. Factual/stance changes alone are not incoherence. This gate is not independent behavioral labeling. All poor and ambiguous outputs are retained. Historical prompt selection did not establish uniformly opposite base/LoRA stances; response-informed development selection and ambiguity remain limitations.','',
 '## Technical checks and interpretation','',
 'Exact coverage, effective generation fields, finite perturbation ratios, prefill/decode hook counts and historical baseline matches verified. intervention_diagnostics.csv reports actual perturbation/residual norm ratios; condition_summary.csv groups them with token-identical fractions. Token changes do not establish semantic steering. One random direction per model is only a limited control. Three seeds are repeated realizations of two prompts, not independent prompts. Cross-configuration probe/NF4 compatibility remains unresolved and was not tested here.','',
 'All continuations are in examples.md and generations.jsonl. Shuffled blinded_review.csv/jsonl has blank first80-word rubric fields and a separate condition_key.csv; no independent evaluator was used. No semantic signed-effect claims are made. Existing source outputs were not changed.','',
 '## Observed quality and actual perturbation scale','',
 'The ±0.3 review found readable outputs without severe new pervasive collapse, allowing both models to proceed. Moderate LoRA digressions and a spacing glitch were recorded before escalation. At ±0.5, individual outputs show clearer relevance/formatting costs: base astro_024 sample2 own− becomes biblical-creation-style text (128a5e1c5efba9843401991b); LoRA questions_012 sample1 own+ becomes footnotes (a32cfe37137c9be5d05689e1); the same prompt/sample random+ becomes treatise headings and eclipse prose (82c1ba2e59ed65f7eb0effa7). These are case-level observations, not a claim that all high-strength outputs are incoherent. high_strength_quality_review.csv records all48 high-strength cases. No further escalation was attempted.','',
 'Across direction/sign condition groups, median prefill perturbation/residual ratios are approximately27.0–27.5% at |k|=.3 and45.0–45.8% at |k|=.5. Median per-generation median token ratios span27.8–30.9% and47.0–52.1%, respectively; maximum individual token ratios reach40.65% and67.30%. Thus k is a fraction of the saved activation scale, not a guarantee of the same fraction of every live residual. condition_summary.csv gives exact group values. All96 new cases differ token-for-token from baseline (0% identical), a technical observation only.','',
 '## Frozen settings and resources','',
 'Full preset configs/generation_reproduced_v1.json: pinned Qwen2.5-7B and final adapter, NF4 double quantization, FP16 compute, SDPA, batch1, inference/eval, single GPU instance, no offload. Plain tokenizer; temperature0.6, top_p0.9, top_k50, max_new_tokens150, repetition_penalty1.1, no_repeat_ngram_size4, caching. RNG reset per condition. CUDA_LAUNCH_BLOCKING=1. Source hashes, reconstruction scripts and atomic resumable per-condition records saved.','',
 f"New-stage elapsed {result['elapsed_seconds']/60:.2f} min; GPU allocated/reserved peaks {result['peak_allocated_bytes']/2**30:.2f}/{result['peak_reserved_bytes']/2**30:.2f} GiB; host peak RSS {result['peak_host_ram_bytes']/2**30:.2f} GiB. GPU peaks exclude driver/context; RSS includes loading/file-backed memory. Failures {result['failures']}. New token-identical cases {result['identical_new_to_baseline']}/{len(new)}.",'',
 '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/sweep_reproduced_protocol_stance.py --stage low\n# Inspect all low-stage outputs and record quality_gate.json before high:\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/sweep_reproduced_protocol_stance.py --stage high\nPYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/report_reproduced_stance_strength_sweep.py\n```']
 (OUT/'report.md').write_text('\n'.join(report)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
