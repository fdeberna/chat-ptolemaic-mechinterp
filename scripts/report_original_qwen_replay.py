"""Verify historical/current generation comparisons and write a provenance report."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/generation_diagnostics/original_answers_10_prompt_replay'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def table(frame):
    def cell(x):return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in frame.itertuples(index=False,name=None)])

def main():
    config=json.loads((OUT/'config.json').read_text());rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
    expected={(m,p) for m in ['base','lora'] for p in config['prompt_ids']}
    assert len(rows)==20 and {(r['model'],r['prompt_id']) for r in rows}==expected
    for path,digest in config['sources'].items():assert sha(path)==digest,path
    metrics=[]
    for r in rows:
        assert r['run_identifier']==config['run_identifier'] and r['sample_id']==0
        old=r['historical_record'];assert old['sample_seed']==r['sample_seed'] and old['prompt']==r['prompt']
        assert r['exact_text_match']==(old['completion']==r['completion'])
        assert r['generated_tokens']==len(r['generated_token_ids'])<=old['max_new_tokens']
        gc=r['effective_generation_config']
        for field in ['max_new_tokens','temperature','top_p','do_sample','repetition_penalty','no_repeat_ngram_size']:assert gc[field]==old[field]
        assert gc['num_return_sequences']==1 and old['num_return_sequences']==5
        assert gc['top_k']==50
        metrics.append(dict(model=r['model'],prompt_id=r['prompt_id'],sample_seed=r['sample_seed'],exact_text_match=r['exact_text_match'],whitespace_normalized_match=' '.join(old['completion'].split())==' '.join(r['completion'].split()),common_prefix_characters=r['common_prefix_characters'],historical_characters=len(old['completion']),current_characters=len(r['completion'])))
    frame=pd.DataFrame(metrics).sort_values(['model','prompt_id']);frame.to_csv(OUT/'validated_comparison_metrics.csv',index=False)
    parity={m:json.loads((OUT/'parity'/f'{m}.json').read_text()) for m in ['base','lora']}
    assert all(x['token_ids_identical'] and x['text_identical'] for x in parity.values())
    resource_rows=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
    summary=dict(primary_outputs=20,distinct_prompts=10,historical_sample_id=0,exact_text_matches=int(frame.exact_text_match.sum()),matches_by_model=frame.groupby('model').exact_text_match.sum().to_dict(),same_environment_original_helper_parity={m:dict(token_ids_identical=True,text_identical=True) for m in parity},
        generation_compute_seconds=sum(r['runtime_seconds'] for r in rows),parity_compute_seconds=sum(r['runtime_seconds'] for r in parity.values()),process_elapsed_seconds=sum(r['elapsed_seconds'] for r in resource_rows),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resource_rows),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resource_rows),peak_host_ram_bytes=max(r['peak_host_ram_bytes'] for r in resource_rows),
        failures=len(list(OUT.glob('failure_*.json'))),input_source_hashes_unchanged=True,
        original_repo_checkout_head=subprocess.check_output(['git','-C',config['original_repo'],'rev-parse','HEAD'],text=True).strip())
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2))
    settings=pd.DataFrame([
        ['Prompt text','Exact historical cleaned prompt text','Same historical text','Different neutral development stems'],
        ['Sampling','true','true','true (sampled arm)'],
        ['max_new_tokens',150,150,200],
        ['temperature',.6,.6,.8],
        ['top_p',.9,.9,.9],
        ['top_k','Not recorded; script inherits model/library default',50,50],
        ['repetition_penalty',1.1,1.1,1.0],
        ['no_repeat_ngram_size',4,4,0],
        ['Seeds','base1234; recorded per-prompt/sample hash seed','Exact saved sample_seed for sample0','0,1,2 per prompt'],
        ['Samples per prompt','5 outer calls, each returns1','Recorded sample0 only','3 independent seeded calls'],
        ['Prompt formatting','Plain tokenizer(prompt), fast tokenizer, no chat template','Same','Same plain tokenization'],
        ['Quantization/compute/attention','Source loader: NF4 double / FP16 / SDPA','Same established setup','Same established setup'],
        ['Model / adapter','Qwen2.5-7B / final exported adapter path','Same model ID/final path; cached revision pinned','Same current pinned model/adapter'],
        ['Historical revision/versions','Not stored in answer records','Current revision/versions recorded','Current revision/versions recorded'],
    ],columns=['Setting','Historical evidence','Matched replay','Recent baseline diagnostic'])
    settings.to_csv(OUT/'settings_comparison.csv',index=False)
    short=frame[['prompt_id','model','exact_text_match','sample_seed']]
    conclusion=('The current working NF4 runner reproduces all20 selected historical saved answers exactly (10/10 base and10/10 LoRA). Both original-helper controls also match the current wrapper token-for-token. No runner mismatch was detected in this bounded replay.' if summary['exact_text_matches']==20 else f"The current runner exactly reproduces {summary['exact_text_matches']}/20 historical saved answers. Both same-environment original-helper controls match the current wrapper token-for-token; historical mismatches therefore require provenance/environment investigation rather than being assumed a wrapper bug.")
    lines=['# Original Qwen answer reproduction: ten prompts, base and LoRA','',conclusion,'',
      '## What was recovered','',
      'The original repository now contains scripts/qlora/generate_eval_qwen.py and evaluation/generation_outputs_qwen/qwen25_base_eval.jsonl / qwen25_lora1000_eval.jsonl. Each saved file contains560 answers: four categories ×28 prompts ×5 samples. Both files consistently record150 new tokens, temperature0.6, top_p0.9, sampling enabled, repetition_penalty1.1, no_repeat_ngram_size4 and base seed1234. These are verified saved-run arguments, not merely script or README defaults. The LoRA records name the same final adapter export used here; the filename lora1000 does not mean a different checkpoint path was silently loaded.',
      'Earlier local audits lacked these files in the then-accessible checkout. This audit uses the newly accessible material at original-repository HEAD '+summary['original_repo_checkout_head']+'. The earlier supplied single-prompt script defaults (0.8, penalty1, no n-gram restriction) did not establish these actual batch-run overrides.',
      '## Selection and method','',
      'Ten prompts were fixed by topical coverage rather than favorable answers: astro_000/005/010/015/022, declarative_000 and questions_002/004/009/012. Topics cover retrograde motion, brightness, inner planets, varying speed, solar-year motion, daily Earth motion and cosmological framing. Exact text and historical IDs/seeds are in selected_prompts.csv. Only recorded sample_id0 is replayed for each model; this is20 primary generations, not a replay of all five samples or all1120 historical answers.',
      'Original clean_prompt_line/load_prompts are used to verify number-prefix removal and exact prompt wording. Original make_sample_seed reconstructs each saved seed using SHA256 of base_seed|prompt_file|prompt_id|sample_id; all20 match. Actual sample_seed values are reused, not arbitrary seeds0/1/2. Historical num_return_sequences=5 denotes five outer calls: generate_completion itself generates one sequence per call. Passing5 directly into model.generate would be a different experiment; this replay keeps one sequence.',
      'Current loading remains the established pinned Qwen2.5-7B '+config['revision']+', final adapter, NF4 double quantization, FP16 base compute/storage, FP32 adapter parameters as loaded, SDPA, batch1, one PEFT-wrapped model instance, inference/eval mode and no CPU offload. Adapter disabled for base, enabled for LoRA. No steering hooks are present, and no fitting, extraction or dependency changes were run.',
      'Current wrapper uses model.generate with restored recorded arguments. On astro_000 for each model, the original repository generate_completion function is also invoked on the same loaded model/tokenizer/settings and seed; its token IDs and text match the wrapper. The framework automatically defaults logits_to_keep to1 in the original helper under Transformers4.57.3; explicitly passing1 in the current wrapper does not change these parity results. CUDA_LAUNCH_BLOCKING=1 preserves the working execution mode after the previous asynchronous CUDA issue; its overhead is included in timings.',
      '## Settings comparison','',table(settings),
      'The README batch examples use temperature0.8, length200 and n-gram3; saved answer metadata instead records0.6/150/4. This comparison follows the answer metadata. Current inherited top_k resolves to50; its historical effective value was not serialized. The original script does not set top_k. Historical model revision, package versions, GPU and adapter-content hash are not saved, so they remain unknown rather than guessed. Current versions: '+json.dumps(config['versions'],sort_keys=True)+'.',
      '## Results','',table(short),
      'Exact match means the full decoded continuation string equals the saved historical continuation, including whitespace and punctuation. Historical token IDs were not saved; historical token-for-token equality is not claimed. Same-environment original-helper/current-wrapper parity does compare token IDs. Historical/current continuations for every selected model/prompt pair appear in examples.md; raw outputs and original records are retained in conditions/*.json / comparisons.jsonl.',
      '## Interpretation and limits','',
      'These results separate implementation from experimental choices. The bounded replay supports using the current runner to reproduce the selected original answers when actual prompt text, per-sample seeds and recorded decoding arguments are restored. The recent experiments changed all of those, including disabling the original anti-repetition controls; they should not have been expected to produce the same continuations. This is not a one-factor causal test of which individual setting explains a quality difference.',
      'Exact reproduction does not establish that the answers are scientifically correct, coherent, or clean behavioral exemplars. Original saved answers themselves include truncated passages, historical/excerpt-like text and—in some source samples—unsolicited quizzes. The report does not add automated stance/style labels or treat text equality as evidence of steering. No tuning or further generations were launched after this bounded check.',
      '## Resources and reproduction','',
      f"Primary generation compute {summary['generation_compute_seconds']/60:.2f} minutes; two parity controls {summary['parity_compute_seconds']/60:.2f} minutes; summed process elapsed {summary['process_elapsed_seconds']/60:.2f} minutes. Peak GPU allocation/reservation {summary['peak_allocated_bytes']/2**30:.2f}/{summary['peak_reserved_bytes']/2**30:.2f} GiB, host RSS {summary['peak_host_ram_bytes']/2**30:.2f} GiB. Allocator peaks exclude driver/context memory; RSS includes loading/file-backed pages. Failure records: {summary['failures']}.",
      'config.json records sources/hashes, current revision, numerical setup, versions and selected historical arguments. source_snapshots/ retains source helper, replay/report scripts and original answer JSONLs. conditions/ commits each result atomically and supports resumption; parity/ retains the two control outputs. settings_comparison.csv, validated_comparison_metrics.csv and verification.json are machine-readable checks. Previous results and the original repository were not modified.',
      '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/reproduce_original_qwen_answers.py\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/report_original_qwen_replay.py\n```',
      'The GPU command requires approved access outside the GPU-blocking sandbox. Compatible completed cases are reused.']
    (OUT/'report.md').write_text('\n\n'.join(lines).replace('\n\n\n\n','\n\n')+'\n');shutil.copy2(__file__,OUT/'source_snapshots'/Path(__file__).name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
