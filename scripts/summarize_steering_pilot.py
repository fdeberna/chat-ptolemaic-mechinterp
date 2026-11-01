"""Verify and describe saved steering generations, without behavioral labeling."""
import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steering/pilot_L16_stance_L6_style'


def main():
    config=json.loads((OUT/'config.json').read_text())
    shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name)
    snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{config["revision"]}'
    provenance_paths=[snapshot/'config.json',snapshot/'generation_config.json',
        ROOT/'results/activations/pilot_full/base/manifest.json',
        ROOT/'results/activations/pilot_full/lora/manifest.json',
        ROOT/'results/full_strength_nf4_fp16/activations/series_config.json',
        Path(config['adapter'])/'tokenizer_config.json',Path(config['adapter'])/'adapter_config.json']
    provenance={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),content=json.loads(p.read_text())) for p in provenance_paths}
    (OUT/'additional_provenance.json').write_text(json.dumps(provenance,indent=2))
    records=[json.loads(p.read_text()) for p in sorted((OUT/'conditions').glob('*.json'))]
    assert len(records)==300
    assert len({r['condition_id'] for r in records})==300
    expected=set()
    for model in ('base','lora'):
        for prompt in config['prompts']:
            pid=prompt['prompt_id'];expected.add((model,pid,None,None,0.))
            for target in ('stance','style'):
                for source in ('base','lora','random'):
                    for k in config['strengths']:expected.add((model,pid,target,source,k))
    actual={(r['model'],r['prompt_id'],r['target'],r['direction_source'],r['k']) for r in records}
    assert actual==expected
    baseline={(r['model'],r['prompt_id']):r for r in records if r['target'] is None}
    ratios=[];comparisons=[]
    for r in records:
        assert r['run_identifier']==config['run_identifier']
        assert r['generated_tokens']==len(r['token_ids'])<=96
        if r['target']:
            h=r['hook'];assert h['calls']==r['generated_tokens'] and h['prefill_calls']==1 and h['decode_calls']==r['generated_tokens']-1
            assert np.isfinite(h['ratios']).all() and min(h['ratios'])>0
            ratios+=h['ratios']
            b=baseline[r['model'],r['prompt_id']]
            common=0
            for a,c in zip(r['token_ids'],b['token_ids']):
                if a!=c:break
                common+=1
            comparisons.append(dict(condition_id=r['condition_id'],model=r['model'],target=r['target'],direction_source=r['direction_source'],k=r['k'],prompt_id=r['prompt_id'],identical_to_baseline=r['token_ids']==b['token_ids'],common_prefix_tokens=common,actual_perturbation_ratio_min=r['actual_perturbation_ratio_min'],actual_perturbation_ratio_median=r['actual_perturbation_ratio_median'],actual_perturbation_ratio_max=r['actual_perturbation_ratio_max']))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'token_comparison_diagnostics.csv',index=False)
    summary=comparison.groupby(['model','target','direction_source','k']).agg(identical_to_baseline=('identical_to_baseline','sum'),conditions=('condition_id','size'),median_common_prefix_tokens=('common_prefix_tokens','median'),median_actual_perturbation_ratio=('actual_perturbation_ratio_median','median')).reset_index()
    summary.to_csv(OUT/'token_comparison_summary.csv',index=False)
    key=pd.read_csv(OUT/'condition_key.csv');review=pd.read_csv(OUT/'blinded_review.csv',keep_default_na=False)
    # Provide prompt context for relevance without revealing conditions. Reviewers
    # must still label the continuation only; the condition key stays separate.
    review=review.drop(columns=['prompt_text'],errors='ignore').merge(key[['review_id','prompt_text']],on='review_id',validate='one_to_one')
    review=review[['review_id','prompt_text','continuation','generated_stance','generated_style','coherence','relevance','repetition','notes']]
    assert len(review)==300 and all((review[c]=='').all() for c in ['generated_stance','generated_style','coherence','relevance','repetition','notes'])
    review.to_csv(OUT/'blinded_review.csv',index=False)
    instructions='''Review only the generated continuation for stance and style; the prompt is context for relevance, not text to score. Keep condition_key.csv hidden from reviewers. Stance options: geocentric / heliocentric / mixed / unclear. Style options: modern / premodern / mixed / unclear. Do not infer a label from an astronomy topic alone. Mark unclear when the continuation supplies insufficient evidence. Record descriptive observations for coherence, relevance and repetition. No suitable independent, established text-labeling rubric/evaluator was found; these are development review instructions, not a validated scoring instrument. All fields remain blank for independent review. No probe scores are behavioral evidence.\n'''
    (OUT/'review_instructions.md').write_text(instructions)
    # Prespecified compact illustrative selection, with both signs and baselines.
    choices=[('stance','base','base','wanderer_notebook'),('style','lora','lora','comet_visitors')]
    examples=['# Illustrative continuations','', 'Development examples only; no behavioral labels or favorable-example selection. Full conditions and all 300 continuations are saved.','']
    for target,model,source,pid in choices:
        examples += [f'## {target}, generating model {model}, direction {source}, prompt {pid}', '',baseline[model,pid]['prompt_text'],'']
        for k in (0.,-.01,.01):
            r=baseline[model,pid] if k==0 else next(x for x in records if x['model']==model and x['prompt_id']==pid and x['target']==target and x['direction_source']==source and x['k']==k)
            examples += [f'### k={k:g}, condition {r["condition_id"]}', '', r['continuation'],'']
    (OUT/'selected_examples.md').write_text('\n'.join(examples))
    smoke=json.loads((OUT/'smoke.json').read_text());checks=json.loads((OUT/'numerical_checks.json').read_text())
    assert len(checks)==4 and all(c['hook_vs_saved_hidden_location_max_difference']==0 and c['cleanup_logits_max_difference']==0 and c['unintended_positions_nonzero']==0 for c in checks)
    resources=[json.loads(p.read_text()) for p in OUT.glob('resources_*.json')]
    gpu_peak=max(x['peak_allocated_bytes'] for x in resources);gpu_reserved=max(x['peak_reserved_bytes'] for x in resources);ram_peak=max(x['peak_host_ram_bytes'] for x in resources)
    metrics=dict(completed=300,baselines=12,steered=288,review_rows=300,numerical_checks=4,all_hook_counts_correct=True,total_generated_tokens=sum(r['generated_tokens'] for r in records),generation_compute_seconds=sum(r['runtime_seconds'] for r in records),session_elapsed_seconds=sum(r['elapsed_seconds'] for r in resources),peak_gpu_allocated_bytes=gpu_peak,peak_gpu_reserved_bytes=gpu_reserved,peak_host_ram_bytes=ram_peak,identical_to_baseline=int(comparison.identical_to_baseline.sum()),stopped_before_token_cap=sum(r['generated_tokens']<96 for r in records),perturbation_ratio_min=float(min(ratios)),perturbation_ratio_max=float(max(ratios)),max_injection_relative_rounding_error=max(c['delta_relative_error'] for c in checks),max_raw_probe_score_absolute_error=max(c['score_absolute_error'] for c in checks))
    (OUT/'verification.json').write_text(json.dumps(metrics,indent=2))
    lines=['# Activation-steering feasibility and development pilot','',
      'Completed 300 unique grid conditions: 12 shared unsteered baselines and 288 steered generations; six development prompts, stance layer 16 and style layer 6, both generating models, both trained directions plus one seeded random direction per layer/model, and k = ±0.003, ±0.01. No behavioral labels have been assigned.','',
      '## Implementation and provenance','',
      'Branch: steering. No AGENTS.md instructions were found. The original extraction code selects the last non-padding token (no mean pooling). Qwen saved layer 16 is model.layers[15] output and layer 6 is model.layers[5] output: residual after attention and MLP additions, before the next block input normalization. These are not attention-submodule outputs or the final model norm. Runtime checks compare hooked tensors with hidden_states[layer] exactly for both models. Applying the learned last-token direction at newly decoded tokens is a generalization assumption, not a change of hook location.',
      'Full-data pilot_full probe coefficients are converted w=v/scale and u=w/||w||; all four directions are finite, nonzero, dimension 3584, and exactly agree with saved raw directions. Stance positive class is heliocentric, style positive class is premodern. Intercepts and calibrated thresholds never enter steering.',
      f'One fixed base-store median activation norm per layer is shared across models/directions: layer 16 a_L={config["scales"]["stance"]:.8g}; layer 6 a_L={config["scales"]["style"]:.8g}. Gamma=k*a_L. Directions use matched Euclidean requested perturbation norms. The fractions are development choices, not optimized values. Actual perturbation/residual norm ratios are retained per hook call, including prefill and every cached decode token.',
      'The established working generation loading setup is pinned Qwen2.5-7B revision '+config['revision']+', packed NF4 with double quantization, FP16 compute/storage for base operations and FP32 LoRA parameters as loaded by PEFT, SDPA with the existing default enabled backends. The FP32 JVP wrapper is not used. Model parameters are frozen, evaluation/inference mode is active, cache is enabled, and no CPU offloading occurs. All parameters are on cuda:0. One PEFT-wrapped instance serves both conditions: disable_adapter() for base, normal active intrinsic alpha/r=2 adapter for LoRA; no merging or adapter rescaling. Adapter/tokenizer and probe/store hashes, package versions and reconstruction code are retained.',
      'Numerical comparability limitation: original pilot_full probe/store manifests document load_in_4bit=false and dtype=auto; they do not establish a fully pinned original revision/actual dtype. This generation run uses the separately validated NF4/FP16 setup that fits 8 GB. Hook location and dimensional coordinates match, but original probe-fit arithmetic is not identical. No old activations are represented as newly matched NF4 measurements, and no probes were refitted.',
      'Six new plain continuation prompts are saved with topics/families. No earlier generation driver or independent text-labeling rubric was found in this repository. No instruction/chat template is introduced; tokenizer equality between the adapter export and base snapshot is checked on these prompts. The overlap audit found at most three contiguous shared words with any fitting prompt. This is a lexical development-data audit, not a semantic independence guarantee.','',
      '## Numerical checks','',
      f'Exactly two complete smoke decodes were run: no-hook baseline (reused in the grid) and a zero-hook verification decode. Next-token logits differed by {smoke["no_hook_vs_zero_max_logits_difference"]:g}, tolerance 1e-6; full greedy token lists matched exactly. The zero-hook verification is not a 301st grid condition.',
      'At each target location for both models, only the intended final prompt token changes; other positions remain bit-identical. The requested delta is gamma*u in float32, rounded into the actual FP16 residual. Equality is checked within a dtype-derived rounding bound, not asserted as exact in finite precision. The associated raw probe-score change is checked against gamma*||w|| with a Cauchy-Schwarz error bound from the measured rounding error. The score check uses the base-trained direction for each target; it is strictly an intervention implementation check.',
      f'Maximum checked relative perturbation-vector rounding error: {metrics["max_injection_relative_rounding_error"]:.4g}; maximum raw probe-score absolute discrepancy: {metrics["max_raw_probe_score_absolute_error"]:.4g}. numerical_checks.json retains expected/actual values and bounds. Normal and deliberate-exception cleanup restored hook counts; no-hook logits after removal matched exactly.',
      'Every steered generation has one prefill hook call at the final non-padding prompt token and generated_tokens−1 cached single-token calls. The last emitted token is not forwarded again. Hook updates clone the output rather than mutate it; actual tensor/tuple output structures are preserved. All logits are checked finite.','',
      '## Resources and completion','',
      f'GPU: {smoke["gpu"]}. Peak allocated/reserved {gpu_peak/2**30:.2f}/{gpu_reserved/2**30:.2f} GiB; peak host RSS {ram_peak/2**30:.2f} GiB. Completed grid generation compute {metrics["generation_compute_seconds"]/60:.2f} minutes; summed process sessions {metrics["session_elapsed_seconds"]/60:.2f} minutes (includes loading/checks and the extra smoke validation decode). Smoke-only and full-grid processes were sequential, never simultaneous model instances. Allocator peaks exclude driver/context memory; host RSS includes file-backed/loading memory.',
      f'Total generated tokens: {metrics["total_generated_tokens"]}. Conditions stopping before the 96-token cap: {metrics["stopped_before_token_cap"]}/300. Steered conditions exactly identical to their shared baseline: {metrics["identical_to_baseline"]}/288. These are token diagnostics, not behavioral effect labels. Actual per-token perturbation/residual norm ratios range from {metrics["perturbation_ratio_min"]:.5g} to {metrics["perturbation_ratio_max"]:.5g}.','',
      '## Behavioral inspection and limits','',
      'Blinded review rows have randomized IDs, prompt context and continuations; stance/style/coherence/relevance/repetition fields remain blank. The separate key exposes conditions and should be withheld by the review organizer. Score the continuation, not prompt content. Labels are not automated, and probe scores are never used as behavioral evidence.',
      'A small representative selection is in selected_examples.md. The base-model smoke continuation already adopted a multiple-choice exercise format; other base continuations developed numerical problems or extraneous assumptions. The LoRA comet baseline described a comet as a cloud of vapor and attributed its disappearance to the Earth’s shadow, illustrating questionable astronomical content. These are observed development-quality problems, not assigned stance/style labels. The 96-token cap may truncate explanations. Inspect all continuations before drawing behavioral conclusions.',
      'One random direction per model/layer is only a preliminary control. Small development prompts, greedy decoding, one numerical setup, and one random control do not support final causal conclusions. Positive probe-score changes do not establish behavioral effects; null behavioral effects do not establish causal irrelevance. No final held-out set, fitting, sensitivity analysis, dataset extraction or training was run.','',
      '## Files and reproduction','',
      'generations.jsonl / generations.csv: all conditions, prompt text, continuations, model/adapter state, direction/seed, layer, k/gamma/scale, token counts, timings and provenance. conditions/*.json are atomic checkpoints keyed by unique condition IDs. config.json pins compatible resumes; previous experiment directories are preserved. directions.npz, direction_audit.json, development_prompts.csv, numerical_checks.json, smoke.json and resource session files retain implementation evidence. token_comparison_diagnostics.csv is descriptive and does not assign behavioral labels.','',
      '```bash',
      'PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/pilot_activation_steering.py',
      '/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/summarize_steering_pilot.py',
      '```',
      'The first command requires GPU access outside the GPU-blocking sandbox. Existing compatible completed conditions are skipped. No dependency installation or upgrade was performed for this pilot.']
    (OUT/'report.md').write_text(('\n\n'.join(lines)+'\n').replace('\n\n\n\n','\n\n'))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
