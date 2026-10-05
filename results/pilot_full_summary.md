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
