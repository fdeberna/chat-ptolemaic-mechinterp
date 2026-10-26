# Probe Direction Stability

This analysis uses saved activations only. It does not rerun activation extraction, modify model weights, run activation steering, or generate new model text.

## Method Checks

- Prompt-level rows: 120.
- Template families: 30.
- Base and LoRA prompt IDs, labels, template families, and layer indices were required to align exactly before analysis.
- Bootstrap resampling is at the template-family level with replacement; duplicated families are expanded to duplicated prompt rows.
- Probe directions are converted from standardized coefficients to raw activation coordinates before L2 normalization.
- Label orientation is required to be stance geocentric=0/heliocentric=1 and style modern=0/premodern=1.
- Layer 0 is excluded from aggregate conclusions.

## Primary Bootstrap Summary

- stance: mean base-within cosine 0.703; mean LoRA-within cosine 0.706; mean matched base-vs-LoRA cosine 0.631; mean alignment gap 0.074.
- stance: largest gap at layer 28 (0.245); smallest gap at layer 1 (-0.281).
- style: mean base-within cosine 0.802; mean LoRA-within cosine 0.813; mean matched base-vs-LoRA cosine 0.538; mean alignment gap 0.270.
- style: largest gap at layer 26 (0.423); smallest gap at layer 2 (-0.093).

## Full-Data Probe Reference

- stance base: mean bootstrap-vs-full cosine 0.821 across nonzero layers (median layer value 0.826).
- stance lora: mean bootstrap-vs-full cosine 0.825 across nonzero layers (median layer value 0.828).
- style base: mean bootstrap-vs-full cosine 0.884 across nonzero layers (median layer value 0.886).
- style lora: mean bootstrap-vs-full cosine 0.888 across nonzero layers (median layer value 0.895).

## Regularization Sensitivity

- stance: OOB balanced accuracy ranged from 0.793 to 0.834; within-model direction mean cosine ranged from 0.708 to 0.747 over the selected layers and C grid.
- style: OOB balanced accuracy ranged from 0.965 to 0.986; within-model direction mean cosine ranged from 0.813 to 0.878 over the selected layers and C grid.

## Feature-Correlation Context

- Candidate-layer activation matrices are prompt-by-hidden-state matrices. Effective rank and PCA counts are descriptive redundancy diagnostics only; they do not identify causal model features.
- Effective rank range across candidate layers and model conditions: 40.7 to 60.3.

## Interpretation

Use `alignment_gap = mean(within-model direction stability) - mean(base-vs-LoRA alignment)` descriptively. Values near 0 mean the between-model direction difference is comparable to ordinary probe refitting variability. Larger positive values mean base and LoRA directions differ more than within-model bootstrap refits. This is not a formal statistical test.
