"""Verify and report completed diagnostic plus explicit provisional text review."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/generation_diagnostics/recovered_defaults_comparison'


def main():
    config=json.loads((OUT/'config.json').read_text())
    rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
    assert len(rows)==32 and len({r['condition_id'] for r in rows})==32
    expected={(m,p['prompt_id'],mode,seed) for m in ('base','lora') for p in config['prompts'] for mode,seeds in [('greedy',[None]),('sampling',[0,1,2])] for seed in seeds}
    assert {(r['model'],r['prompt_id'],r['mode'],r['seed']) for r in rows}==expected
    review=json.loads((OUT/'quality_review.json').read_text())
    assert set(review['conditions'])=={r['condition_id'] for r in rows}
    configurations={};verified={}
    for condition in ('base','lora'):
        check=json.loads((OUT/'verification'/f'{condition}.json').read_text())
        assert check['token_ids_identical'] and check['manual_tokens']==check['generate_tokens']
        verified[condition]=dict(token_ids_identical=True,tokens=len(check['manual_tokens']),manual_runtime_seconds=check['manual_runtime_seconds'])
    for row in rows:
        assert row['no_intervention_hooks_verified'] and row['run_identifier']==config['run_identifier']
        gc=row['generation_config'];assert gc['top_k']==50 and gc['max_new_tokens']==200 and gc['use_cache'] and gc['eos_token_id']==gc['pad_token_id']==151643
        assert gc['repetition_penalty']==1 and gc['no_repeat_ngram_size']==0
        if row['mode']=='sampling':assert gc['temperature']==.8 and gc['top_p']==.9 and row['seed'] in (0,1,2)
        assert len(row['generated_token_ids'])==row['generated_tokens']<=200
        assert row['input_ids']==json.loads((OUT/'tokenization.json').read_text())[row['model']]['input_ids'][row['prompt_id']]
        effective=copy.deepcopy(gc)
        effective.update(return_dict_in_generate=True,output_logits=row['prompt_id']=='daily_sky' and row['mode']=='greedy',output_scores=False,max_length=len(row['input_ids'])+200)
        configurations[row['condition_id']]=dict(effective_generation_config=effective,model_kwargs={'logits_to_keep':1},explanation='Includes actual generate() output-control overrides and prompt-dependent maximum length; top_k remains inherited. Output-logit retention is verification-only and does not alter token selection.')
    (OUT/'effective_runtime_configs.json').write_text(json.dumps(configurations,indent=2))
    records=[]
    for row in rows:
        records.append(dict(condition_id=row['condition_id'],model=row['model'],prompt_id=row['prompt_id'],mode=row['mode'],seed=row['seed'],stopping_reason=row['stopping_reason'],generated_tokens=row['generated_tokens'],**review['conditions'][row['condition_id']]))
    pd.DataFrame(records).to_csv(OUT/'quality_review.csv',index=False)
    resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
    metrics=dict(outputs=32,successful_manual_verifications=verified,failed_attempts=len(list(OUT.glob('failure_*.json'))),
        comparison_compute_seconds=sum(r['runtime_seconds'] for r in rows),all_session_elapsed_seconds=sum(x['elapsed_seconds'] for x in resources),
        peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in resources),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in resources),peak_host_ram_bytes=max(x['peak_host_ram_bytes'] for x in resources),
        eos_stops=sum(r['stopping_reason']=='eos' for r in rows),token_cap_stops=sum(r['stopping_reason']=='max_new_tokens' for r in rows))
    (OUT/'verification_summary.json').write_text(json.dumps(metrics,indent=2));shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
    table=['| Model / decoding | Provisional quality summary |','|---|---|']+[f'| {key} | {text} |' for key,text in review['model_mode_summary'].items()]
    report=['# Bounded unsteered recovered-default comparison','',
      'Completed 32 comparison outputs: four fixed new development prompts × base/LoRA × one greedy plus sampling seeds 0,1,2. Two successful manual-loop verification decodes are separate; their generate() counterparts are reused as daily-motion greedy outputs. An initial failed session also completed an unsaved manual pass before generate() failed. No steering, probe fitting, extraction or dependency changes. Historical command-line overrides remain unknown; supplied script defaults are recovered settings, not proof of the historical invocation.',
      '## Manual-loop compatibility','',
      *[f'- {model}: exact manual/generate token identity for {value["tokens"]} tokens on the fixed daily-motion prompt.' for model,value in verified.items()],
      'No token divergence occurred in either successful verification, so no after-divergence logits were compared. Checks cover one prompt per model, not all possible inputs. The initial asynchronous attempt failed with a CUDA unknown error in generate(), without a reported OOM. A single fresh-process retry with CUDA_LAUNCH_BLOCKING=1 completed the experiment. Root cause is unresolved; no library/model/decoding changes were made. Successful timings include synchronous-launch overhead and should not be compared directly with earlier asynchronous pilot timings.',
      '## Effective settings and tokenization','',
      'Pinned Qwen2.5-7B revision '+config['revision']+', existing adapter, NF4 double quantization, FP16 base compute/storage with FP32 LoRA parameters, SDPA, inference/eval mode, batch1, one GPU model instance. Base uses disabled adapters; LoRA active. All parameters are frozen and on cuda:0, with no CPU offload. Decoder/final-norm hook absence is asserted before generation. The separate unquantized-probe versus NF4-generation compatibility question was not tested.',
      'Supplied-script tokenizer selection is followed: base snapshot for base, adapter-local tokenizer for LoRA, use_fast=True, tokenizer(prompt, return_tensors="pt") defaults, no chat template. EOS/PAD are tokenizer IDs 151643; no BOS is prepended. Earlier steering used adapter-local tokenizer for both models and explicitly add_special_tokens=True/right padding. For these prompts the saved token IDs agree between sources; default special-token handling is equivalent. The supplied script prints prompt+continuation; this diagnostic saves continuation-only text with prompt/input token IDs separately.',
      'Both regimes use max_new_tokens=200, repetition_penalty=1.0, no_repeat_ngram_size=0, use_cache=True, num_beams=1. Greedy has do_sample=False and no temperature/top-p warpers; sampling has do_sample=True, temperature=0.8, top_p=0.9. **Inherited top_k=50**, not 0, is retained. Python/NumPy/Torch/CUDA RNGs are reset separately for every sampled condition. Seeds are 0,1,2. EOS or the 200-token cap stops generation; no resource-driven length cap was introduced.',
      'effective_generation_configs.json contains complete resolved configurations before output-control overrides. effective_runtime_configs.json includes actual return-dict/output-logit flags, logits_to_keep=1 model kwarg, and prompt-dependent max_length. Output logits are retained only for the two generate() verification calls; output_scores=False throughout. This does not alter token selection. Relevant library versions: '+json.dumps(config['versions'],sort_keys=True)+'.',
      '## Quality review','', '\n'.join(table),
      'Every continuation was read in full. quality_review.json/CSV records coherence, relevance, repetition/recursive dialogue, unsolicited exercises, factual issues, truncation and stance/style assessability per condition. These are provisional development observations, not validated behavioral labels. Factual errors are distinguished from unintelligible writing; repeated quiz format is distinguished from exact sentence looping. Conflicting multiple-choice options can make a global stance label ambiguous even when an opening answer is explicit.',
      '## Decision','',review['baseline_decision'],review['smallest_remaining_issue'],
      'No steering was resumed, no extra seeds/prompts/decoding values were searched, and no historical invocation was inferred from defaults.',
      '## Resources, artifacts and reproduction','',
      f"Comparison generate() compute {metrics['comparison_compute_seconds']/60:.2f} minutes; summed sessions {metrics['all_session_elapsed_seconds']/60:.2f} minutes, including loading, verification and failed attempt. Peak GPU allocation/reservation {metrics['peak_allocated_bytes']/2**30:.2f}/{metrics['peak_reserved_bytes']/2**30:.2f} GiB; peak host RSS {metrics['peak_host_ram_bytes']/2**30:.2f} GiB. GPU allocator peaks exclude driver/context; RSS includes file-backed/model-loading memory. EOS stops {metrics['eos_stops']}/32, token-cap stops {metrics['token_cap_stops']}/32. A cap stop indicates truncation risk; per-condition notes distinguish syntactically incomplete endings.",
      'examples.md contains all 32 continuation-only outputs. conditions/*.json are atomic resumable checkpoints with prompt text/input IDs, continuation token IDs, model/adapter state, seeds, generation settings, stop reason, timings and run ID. Source/model/adapter/tokenizer hashes and versions are in config.json. execution_notes.json and failure/resource logs preserve the CUDA retry evidence. All previous results remain unchanged.',
      '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/diagnose_unsteered_decoding.py\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/summarize_unsteered_diagnostic.py\n```',
      'The GPU command requires approved access outside the GPU-blocking sandbox and skips compatible completed conditions. The summarizer uses the retained provisional quality review; it does not invent labels.']
    (OUT/'report.md').write_text('\n\n'.join(report).replace('\n\n\n\n','\n\n')+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
