"""Validate Stage3A coverage and export review artifacts without judging text."""
from pathlib import Path
import csv,hashlib,json,random,re,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/adapter_ablation/stage3a'
FIELDS=['stance_label','stance_supporting_excerpt','earth_rotation','earth_rotation_supporting_excerpt','earth_revolution_around_sun','earth_revolution_supporting_excerpt','style_label','style_supporting_excerpt','semantic_notes','full_continuation_repetition','full_continuation_incoherence','full_continuation_topic_drift','full_continuation_truncation','quality_notes','reviewer_id']
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
 expected={(c,p,s) for c in cfg['conditions'] for p in cfg['selection']['selected_prompt_ids'] for s in [0,1,2]}
 assert len(rows)==60 and {(r['condition'],r['prompt_id'],r['sample_id']) for r in rows}==expected
 assert len({r['condition_id'] for r in rows})==60
 for r in rows:
  assert r['run_identifier']==cfg['run_identifier'] and r['disabled_saved_layers']==cfg['conditions'][r['condition']]
  assert len(r['disabled_modules'])==7*len(r['disabled_saved_layers'])
  group=[a for a in rows if a['prompt_id']==r['prompt_id'] and a['sample_id']==r['sample_id']]
  assert len({a['sample_seed'] for a in group})==1 and all(a['input_ids']==r['input_ids'] and a['generation_config']==r['generation_config'] for a in group)
 for path,digest in cfg['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 (OUT/'generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda x:x['condition_id'])))
 csvout('generations.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows])
 rng=random.Random(20261101);ordered=sorted(rows,key=lambda r:r['condition_id']);rng.shuffle(ordered);review=[];key=[]
 for r in ordered:
  rid=f'{rng.getrandbits(96):024x}';words=list(re.finditer(r'\S+',r['continuation']));window=r['continuation'] if len(words)<=80 else r['continuation'][:words[79].end()]
  review.append(dict(review_id=rid,prompt_context=r['prompt_text'],scoring_window_first_80_words=window,scored_word_count=min(80,len(words)),full_continuation=r['continuation'],**{k:'' for k in FIELDS}))
  refs={a['condition']:a['condition_id'] for a in rows if a['prompt_id']==r['prompt_id'] and a['sample_id']==r['sample_id']}
  key.append(dict(review_id=rid,condition_id=r['condition_id'],condition=r['condition'],prompt_id=r['prompt_id'],sample_id=r['sample_id'],sample_seed=r['sample_seed'],disabled_saved_layers=json.dumps(r['disabled_saved_layers']),base_reference=refs['base'],intact_lora_reference=refs['intact_lora']))
 if (OUT/'blinded_review.jsonl').exists():
  old=[json.loads(x) for x in (OUT/'blinded_review.jsonl').read_text().splitlines()];assert all(not r.get(k) for r in old for k in FIELDS),'Preserve supplied annotations'
 (OUT/'blinded_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review));csvout('blinded_review.csv',review);csvout('condition_key.csv',key)
 examples=['# Stage3A: all continuations','', 'Development outputs; no behavioral labels assigned.','']
 for pid in cfg['selection']['selected_prompt_ids']:
  for s in [0,1,2]:
   group=[r for r in rows if r['prompt_id']==pid and r['sample_id']==s];examples += [f'## {pid} / sample{s}', '',group[0]['prompt_text'],'']
   for c in cfg['conditions']:
    r=next(r for r in group if r['condition']==c);examples += [f'### {c}', '',r['continuation'],'']
 (OUT/'examples.md').write_text('\n'.join(examples))
 resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
 result=dict(complete=True,total_conditions=60,reused=sum(bool(r['reused_condition_id']) for r in rows),new_generations=sum(not r['reused_condition_id'] for r in rows),stage3b_run=False,stage3c_run=False,labels_supplied=False,elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),peak_gpu_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_gpu_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resources),failures=len(list(OUT.glob('failure_*.json'))))
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 text=f'''# Stage3A single-layer ablation pilot

Complete:60 conditions; {result['new_generations']} new outputs and {result['reused']} reused references.
Four preselected, previously exposed development prompts (questions_002,
questions_004, questions_012, astro_010), three matched historical sample IDs and
five conditions: base, intact LoRA, and LoRA with L6/L16/L28 adapters disabled.
Reserved confirmation prompts were untouched. These prompts were fixed by topic
and exposure status, without new response screening; assessability remains to be
established by scoring. They retain historical question/continuation wording.

The Step2-validated scoped controller disables all seven adapter modules of each
selected block on all prompt/decode tokens, before prefill with a fresh KV cache.
Base disables all196 adapter modules, equivalent to global disabling in Step2.
Frozen NF4/FP16/SDPA loading, FP32 adapter weights, original tokenization and derived
seeds, sampling T0.6/top_p0.9/top_k50,150 tokens, repetition penalty1.1 and 4-gram
repetition constraint were retained. Each call resets RNGs. Reused references match
the frozen preset hash, input token IDs, generation config and actual seed.

Coverage, paired inputs/settings/seeds, exact disabled module sets and restoration
were checked. Original results are preserved. All continuations, including poor or
ambiguous text, are retained in examples.md and generations.jsonl. No semantic or
quality labels have been invented. Blinded review fields are blank for a separate
reviewer, with the frozen first80-word rubric and condition_key.csv kept separately.
The generating session is not claimed to be blinded. Three seeds are decoding
realizations, not independent prompts.

Elapsed {result['elapsed_seconds']/60:.2f}min; GPU allocated/reserved peaks
{result['peak_gpu_allocated_bytes']/2**30:.2f}/{result['peak_gpu_reserved_bytes']/2**30:.2f}GiB;
host RSS {result['peak_host_ram_bytes']/2**30:.2f}GiB. GPU allocator peaks exclude
context/driver memory; host RSS includes loading/file-backed memory. Recorded
failures:{result['failures']}.

Stages3B and3C remain unexecuted. Review the paired base/intact references and
ablation answers for assessability, stance, separate Earth-motion claims, style
and quality before deciding whether larger groups warrant testing. A null
single-layer effect would not exclude distributed adaptation. No efficacy claim
or decision to expand is made before review.

## Reproduction

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/run_ablation_stage3a.py
python -B scripts/report_ablation_stage3a.py
```

The runner resumes compatible atomic checkpoints. Preserve development_selection.json,
experiment_config.json, frozen preset, rubric, reconstruction scripts, source
hashes, condition records and the original referenced baseline artifacts. The
full configuration and resource records are saved alongside this report.
'''
 (OUT/'report.md').write_text(text);shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
