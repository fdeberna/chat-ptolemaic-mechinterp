# Pilot Full Probe Summary

This is a linear-probe diagnostic. Linear decodability means the target label is recoverable by a linear separator from hidden states; it does not establish causal use by the language model during generation. Steering and activation patching are separate later experiments.

## Dataset and Activations

- Dataset size: 120 prompts.
- Template families: 30.
- Model conditions: base Qwen and LoRA-enabled Qwen.
- Activation shape: 29 layers including layer 0 embedding output, 28 transformer blocks, hidden dimension 3584.
- No generated text is involved in this stage.

## Cross-Validation

- Grouped stratified cross-validation by `template_family`.
- Train/test family overlap was checked and must be zero for every fold.
- The same held-out template families are used for base and LoRA metrics.

## Best Decoding Layers

- base stance: layer 16, balanced accuracy 0.833, ROC AUC 0.917, base/LoRA direction cosine at that layer 0.601.
- lora stance: layer 18, balanced accuracy 0.855, ROC AUC 0.928, base/LoRA direction cosine at that layer 0.570.
- base style: layer 6, balanced accuracy 1.000, ROC AUC 1.000, base/LoRA direction cosine at that layer 0.678.
- lora style: layer 6, balanced accuracy 0.982, ROC AUC 0.999, base/LoRA direction cosine at that layer 0.678.

## Transfer

- Strongest cross-model transfer: style layer 8 base->lora balanced accuracy 0.974, ROC AUC 0.998.
- Weakest cross-model transfer: stance layer 0 base->lora balanced accuracy 0.500, ROC AUC 0.500.

## Stance vs Style

- Diagnostic comparison table: `results/comparisons/pilot_full/stance_style_summary.csv`.
- Mean stance direction cosine: 0.641.
- Mean style direction cosine: 0.538.

## Warnings

- Layer 0 is the embedding output and should be interpreted cautiously.
- Transformer layers should not be treated as independent statistical samples.
- Some retention ratios are NaN where within-model performance was near chance.


## Steering Pilot — Closed (2026-11-01)

The reproduced-protocol stance-steering pilot is closed without further tuning or a
reserved-prompt confirmation run. It found prompt-specific suppression of expressed
Earth-motion claims, but no reliable bidirectional semantic stance effect across the
two models and two development prompts. This does not establish that the directions
cannot steer, or that the probed features are causally irrelevant.

### Final findings

The final scored set contains 156 unique outputs: two prompts (`questions_012`,
retrograde; `astro_024`, annual solar circuit), base and final LoRA, sample IDs
0/1/2, and 13 conditions (one baseline plus own/random directions at signed
k = ±0.1, ±0.3, ±0.5). All IDs, text, first-80-word windows, required rubric fields,
and exact coverage validated. The 12 baselines are reused in paired comparisons,
not counted as fresh evidence at every strength. Evaluator labels and excerpts
were preserved; no new judging was performed during analysis.

- **Base retrograde:** own-negative heliocentric counts fall from baseline 3/3 to
  2/3 at −0.3 and 0/3 at −0.5; random-negative remains 3/3. At −0.5, all three
  own-negative responses are labeled unclear, with Earth revolution unmentioned,
  rather than geocentric or denying revolution. These three responses have no
  supplied incoherence/topic-drift flags and remain modern in style. Own-positive
  is not monotonic: heliocentric counts are 2/3, 3/3, 2/3 at +0.1/+0.3/+0.5.
- **Base annual solar circuit:** the heliocentric counts by signed strength match
  the random control in aggregate; seed-level assignments differ. Most outputs
  are unclear, and there is no geocentric conversion.
- **LoRA annual solar circuit:** every own-negative magnitude changes baseline
  rotation affirmed 3/3 to unmentioned 3/3, without denial. Own-positive retains
  rotation on 2/3, 2/3, 1/3 as strength increases; it does not improve on baseline.
  No steered LoRA annual response affirms Earth-around-Sun revolution.
- **LoRA overall:** none of its 78 outputs is labeled heliocentric. Two newly
  geocentric labels occur at own +0.3, opposite the intended positive stance sign.
  Own-negative does not consistently induce new geocentric labels.
- **Interpretability:** across all outputs, stance counts are heliocentric 39,
  geocentric 7, unclear 110, mixed 0. At own |k|=0.5, 21/24 are unclear and 12/24
  have supplied incoherence flags, versus 16/24 and 5/24 at |k|=0.1. These pooled
  counts describe outputs, not independent prompts. Random controls also change
  relevance/style/quality. Repetition is flagged on 0/156 outputs. No own-steered
  output is labeled as denying either Earth-motion field: suppression is omission,
  not demonstrated denial.

### Interpretation limits

Only two response-selected development prompts and three paired decoding
realizations were evaluated; seeds are not independent prompts. Historical
vocabulary, quotation and attributed models are not automatically endorsements.
Earth rotation alone is not heliocentrism; unmentioned is not denied. Mixed,
unclear, conflicting and unmentioned categories were retained without an ordinal
stance score. No significance tests or confirmed winning strength are reported.
One seeded random direction per model is a limited control, not a null distribution.

Higher strengths often reduce direct-answer relevance or introduce formatting and
style changes; this is not uniform incoherence across all outputs. Quality changes
are compared with matched baselines, not attributed to steering solely because a
baseline was poor. Semantic labels use the first 80 generated words; full-text
quality is separate. Reviewer independence cannot be established from the supplied
reviewer ID alone. Two truncation=no labels differ from token-cap stopping metadata;
these remain review flags, not automatically annotation errors.

The probe directions were fitted on nonquantized activations, while generation used
NF4; the pending numerical-compatibility extraction was not performed. Successful
hooks, positive probe-score changes and changed tokens do not establish behavioral
steering. Absence of a general effect here does not demonstrate causal irrelevance.

### Frozen setup and reproduction record

- Protocol: [`configs/generation_reproduced_v1.json`](../configs/generation_reproduced_v1.json)
  and [`configs/GENERATION_PROTOCOL.md`](../configs/GENERATION_PROTOCOL.md).
  Pinned Qwen2.5-7B/final adapter, NF4 double quantization, FP16 compute, SDPA,
  plain-text tokenization, sampling temperature 0.6/top_p 0.9/top_k 50,
  max_new_tokens 150, repetition_penalty 1.1, no_repeat_ngram_size 4. Original
  SHA256-derived per-sample RNG seeds, batch size 1, cache enabled, inference mode,
  one GPU model, no CPU offload. Historical invocation overrides remain unknown;
  these settings reproduced the saved selected historical baselines exactly.
- Intervention: `src/ptolemaic_mechinterp/interventions/steering.py`, saved L16 =
  decoder block 15 residual output (matching last-nonpadding-token probes).
  Raw own direction u = (v / scaler_scale) / ||v / scaler_scale||; positive class
  heliocentric. No intercept or threshold injected. Hook final prompt token, then
  every cached newly processed token. Shared a_L = 73.83589057971676 and gamma=k*a_L.
  At |k|=.5 the median prefill perturbation/residual norm ratio is about 45–46%,
  not necessarily 50% of each live residual. One unchanged seeded random vector
  per model, matched Euclidean perturbation norms.
- Frozen rubric: [`configs/steering_scoring_rubric_v1.md`](../configs/steering_scoring_rubric_v1.md),
  copied byte-for-byte from the experiment's `rubric.md` (SHA256
  `d6affa10b842da75e227ca937363aaf028e2017d5203c296c90b09557cdc8555`).
- Historical replay: `scripts/reproduce_original_qwen_answers.py` and
  `results/generation_diagnostics/original_answers_10_prompt_replay/`.
  Selection and first reproduced test: `scripts/prepare_reproduced_stance_development.py`,
  `scripts/run_reproduced_protocol_stance.py`,
  `scripts/report_reproduced_stance_development.py`, and
  `results/steering/reproduced_protocol_stance_development/` (60 reused records,
  development selection, prompt partitions and historical source references).
- Strength sweep: `scripts/sweep_reproduced_protocol_stance.py --stage low|high`
  and `scripts/report_reproduced_stance_strength_sweep.py`. Preserve their helper
  scripts, including `run_sampled_steering.py` and `summarize_sampled_steering.py`.
  Output root: `results/steering/reproduced_protocol_stance_strength_sweep/`.
  `quality_gate.json` records the ±0.3 review before ±0.5; all poor outputs remain.
  The sweep added 96 generations, reusing 60 records; elapsed 54.3 minutes, peak
  GPU allocated/reserved 5.60/6.09 GiB, host RSS 4.60 GiB, no failures.
- Analysis: `scripts/analyze_reproduced_stance_strength_scoring.py`. Canonical
  output: `results/steering/reproduced_protocol_stance_strength_sweep/scored_analysis_20261030/`.
  Preserve `report.md`, `data_report.md`, `validation_results.json`,
  `analysis_checks.json`, `annotation_review_flags.csv`, joined annotations,
  counts, paired tables, PNG/PDF plots and `provenance.json`. The older
  `scored_analysis/` directory is an explicitly marked incomplete plotting attempt,
  not the final analysis.
- Raw evidence in the sweep root: `blinded_review_scored.jsonl`, original
  `blinded_review.jsonl`, separate `condition_key.csv`, `generations.jsonl`,
  per-condition checkpoints, intervention diagnostics, frozen configuration,
  reuse manifest, resource records, source hashes and reconstruction scripts.
  Direction vectors: `results/steering/pilot_L16_stance_L6_style/directions.npz`;
  source probes: `results/probes/pilot_full/{base,lora}/stance/full_data_coefficients/layer_016_full_data.npz`.
  Keep those local artifacts and the pinned external model/adapter/tokenizer
  caches; paths and hashes alone are not a backup of the data.

CPU-only analysis reproduction (refuses a nonempty destination; no inference):

```bash
/home/fdebe/miniconda3/envs/copernicus2/bin/python -B \
  scripts/analyze_reproduced_stance_strength_scoring.py \
  --output-dir /tmp/steering-scoring-fresh-analysis
```

Generation commands remain documented in the sweep report for provenance only;
closing this experiment does not authorize another run. Earlier steering setups
and cumulative interpolation results remain separately labeled, not pooled here.
Generated steering outputs remain local and ignored by Git. Configurations,
rubric, code and this summary are reviewable repository files; reproducing analysis
requires retaining the referenced local artifacts. No reserved prompts, generation,
extraction, fitting or tuning were run for closure.

### Closure checks and diff review

- CPU-only synthetic hook checks passed: tensor/tuple outputs, final-prefill and
  cached-token placement, no input mutation, gamma=0 equivalence, counts, wrapper
  behavior and hook removal on exceptions. CUDA was disabled.
- AST parsing passed for the 19 pending experiment scripts and steering module;
  frozen configuration JSON and rubric hash checks passed.
- Saved evidence rechecked: 156 unique records with exact coverage, unchanged
  text/labels/excerpts, matching provenance input hashes and count denominators.
  No generation, model loading, extraction, fitting or new judging was performed.
- `git diff --check` passed. No pending file is at least 1 MiB, no generation
  artifacts are exposed by the ignore rules, and no new unrelated changes were
  introduced. Four roughly 71 MiB bootstrap arrays are already tracked from
  earlier work and unchanged; closure does not delete or rewrite them. Ignored
  bytecode caches and the marked incomplete analysis attempt are left intact.
- The requested pytest subset could not run: pytest is not installed in the
  available environments. No dependencies were installed. The direct CPU and
  artifact checks above passed; this is not a claim that the pytest suite passed.
- Machine-readable record: `results/steering/reproduced_protocol_stance_strength_sweep/closure_20261101.json`.
