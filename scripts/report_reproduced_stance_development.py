"""Verify the controlled repeat and export unlabeled blinded review artifacts."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np
import pandas as pd
from summarize_sampled_steering import FIELDS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/steering/reproduced_protocol_stance_development'

def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());selection=cfg['selection'];preset=json.loads(Path(cfg['preset_path']).read_text())
 assert hashlib.sha256(Path(cfg['preset_path']).read_bytes()).hexdigest()==cfg['preset_sha256']
 rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
 expected={(m,p,s,d,k) for m in ['base','lora'] for p in selection['selected_prompt_ids'] for s in [0,1,2] for d,k in [(None,0.),('own',-.1),('own',.1),('random',-.1),('random',.1)]}
 assert len(rows)==60 and {(r['model'],r['prompt_id'],r['sample_id'],r['direction_source'],r['k']) for r in rows}==expected
 baselines={(r['model'],r['prompt_id'],r['sample_id']):r for r in rows if r['kind']=='baseline'};assert len(baselines)==12 and all(r['historical_exact_match'] for r in baselines.values())
 zero=[json.loads(p.read_text()) for p in (OUT/'verification').glob('*.json')];assert len(zero)==2
 for r in zero:assert r['generated_token_ids']==baselines[r['model'],r['prompt_id'],r['sample_id']]['generated_token_ids']
 metrics=[]
 for r in rows:
  assert r['run_identifier']==cfg['run_identifier']
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
 for index in rng.permutation(60):
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
     for k in [-.1,.1]:
      r=next(r for r in rows if (r['model'],r['prompt_id'],r['sample_id'],r['direction_source'],r['k'])==(m,pid,s,d,k));lines += [f'### {d}, k={k:g}, gamma={r["gamma"]:.6g}', '',r['continuation'],'']
 (OUT/'examples.md').write_text('\n'.join(lines))
 resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
 result=dict(total_conditions=60,new_baselines=12,steered=48,baseline_exact_historical_matches=12,zero_hook_token_matches=2,labels_supplied=False,independent_evaluator=False,reserved_confirmation_prompts=96,
  elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),generation_seconds=sum(r['runtime_seconds'] for r in rows)+sum(r['runtime_seconds'] for r in zero),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resources),failures=len(list(OUT.glob('failure_*.json'))),identical_steered_tokens_to_baseline=sum(r['identical_tokens_to_baseline'] for r in metrics))
 (OUT/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
 report=['# Frozen-protocol development stance-steering repeat','',
  '## Frozen default','',
  'configs/generation_reproduced_v1.json is the frozen default for subsequent generation experiments, documented in configs/GENERATION_PROTOCOL.md. This runner loads the entire preset rather than recreating old defaults. It preserves the model revision, final adapter/tokenizer hashes, NF4 double quantization, FP16 compute, SDPA, sampling temperature0.6/top_p0.9/top_k50, length150, repetition_penalty1.1 and no_repeat_ngram_size4. EOS/PAD151643, no chat template, inference/eval mode, batch1, one GPU instance and no offload. CUDA_LAUNCH_BLOCKING=1 matches the successful reproduction setup. Older archived runners are not rewritten, preserving their provenance.',
  '## Historical screen and selection','',
  'All five historical samples were inspected for ten candidate prompts under base, LoRA500 and LoRA1000:150 existing responses, without generation. screening_plan.json records the pool before inspection and its bounded amendment; historical_screened_samples.jsonl archives all screened cases and screening_notes.csv records decisions. LoRA500 is a different run/export and is historical context only, not a new steering model or a monotonic training-dose comparison.',
  '**The strict selection objective was not met:** none of the ten screened prompts gave a clean stable five-of-five base-heliocentric versus LoRA-geocentric contrast. We did not reinterpret historical vocabulary, objections, quotations, rotation-only claims or missing answers as geocentric stance.',
  'The two best available component-level development comparisons were selected: questions_012 (retrograde) and astro_024 (yearly solar circuit). Base retrograde answers consistently discuss Sun-centered planetary orbits; final-LoRA answers often instead give Earth-centered mechanisms or leave global stance unresolved. The yearly-circuit pair distinguishes explicit Earth-around-Sun revolution in some base answers from axial rotation/heavenly-revolution explanations in LoRA. Earth rotation alone is not heliocentrism. These are partially interpretable comparisons, not a uniformly opposite binary stance set. The full limitations are in screening_report.md.',
  'prompt_partitions.csv reserves96 other prompts for later confirmation, excluding the ten screened prompts and six additional previously exposed replay prompts. None of the reserved prompts was generated here. Selection was response-informed development work; this is not an untouched final evaluation set.',
  '## Controlled repeat','',
  'Two prompts × two generating models × historical sample_ids0/1/2. The first three sample IDs were fixed regardless of favorable historical answers. Actual RNG seeds are the original SHA256-derived seeds with base_seed1234, not literal seeds0/1/2. Three sampled realizations are not three independent prompts. Each call returns one sequence; the preset historical five-sample count is an outer-loop policy, and this explicitly bounded test uses three of those IDs.',
  'Stance saved L16 / decoder block15 output, own trained raw unit direction and the existing model-specific seeded random unit control, k=−0.1/0/+0.1, gamma=k*a_L with unchanged a_L=73.83589057971676. Gamma is ±7.383589057971676. No intercept/threshold is injected. Token policy is unchanged: final prompt token during prefill, then every newly processed cached token. Raw directions and positive class heliocentric are verified against saved probe coefficients. No additional layers, strengths, adapters, seed search, fitting, training or NF4 compatibility extraction.',
  f'Completed {result["steered"]} steered cases and {result["new_baselines"]} fresh baselines. **All12 fresh baselines exactly reproduce their selected historical answers.** Two zero-hook checks match their corresponding no-hook baseline token-for-token. Hook counts and finite actual perturbation ratios are verified. Each completed generation is atomically checkpointed and resumable.',
  '## Interpretation','',
  'This repeats the technical steering test under the empirically reproduced behavioral protocol. It establishes that the new test is anchored to matching historical baselines; it does not by itself establish a semantic stance effect or prove the previous null/ambiguous result was caused by the mismatched setup. Prompt set, seed policy and several decoding settings change together relative to the earlier pilot, so individual causes cannot be isolated.',
  'All60 continuations are in examples.md/generations.jsonl. A shuffled blinded_review.csv/jsonl and separate condition_key.csv use the prior frozen first80-generated-word rubric, including separate Earth rotation/revolution, stance/style, supporting excerpts and full-continuation quality. No separate evaluator was used: fields remain blank and the generating session is not claimed blinded. No semantic effect counts are invented. Changed words or probe scores are not semantic steering evidence. Baseline selection ambiguity remains a material limitation for interpreting any future labels.',
  '## Resources and reproduction','',
  f'Elapsed {result["elapsed_seconds"]/60:.2f} minutes; generation/zero-check compute {result["generation_seconds"]/60:.2f} minutes. Peak GPU allocation/reservation {result["peak_allocated_bytes"]/2**30:.2f}/{result["peak_reserved_bytes"]/2**30:.2f} GiB; host RSS {result["peak_host_ram_bytes"]/2**30:.2f} GiB. Allocator peaks exclude driver/context; host RSS includes loading/file-backed memory. Failures {result["failures"]}. Token-identical steered/baseline cases {result["identical_steered_tokens_to_baseline"]}/48 are a technical diagnostic only.',
  '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/run_reproduced_protocol_stance.py\nPYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/report_reproduced_stance_development.py\n```',
  'GPU execution needs approved access outside the GPU-blocking sandbox. Frozen preset/source hashes and completed cases protect compatible resumption. All previous results and original source answers are preserved. No confirmation experiment was launched.']
 (OUT/'report.md').write_text('\n\n'.join(report).replace('\n\n\n\n','\n\n')+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
