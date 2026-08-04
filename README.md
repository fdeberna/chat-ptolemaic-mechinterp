# chat-ptolemaic-mechinterp

This repository is a mechanistic follow-up to the original `chat-ptolemaic` research project.
The original project fine-tuned Qwen2.5-7B with LoRA on pre-Copernican astronomical texts. The
behavioral result was that fine-tuning strongly shifted the model toward premodern astronomical
language and explanatory frameworks, but did not consistently shift its factual stance toward
geocentrism.

Original paper: [https://arxiv.org/abs/2605.30415](https://arxiv.org/abs/2605.30415)

Original repository: [https://github.com/fdeberna/chat-ptolemaic](https://github.com/fdeberna/chat-ptolemaic)

## Research Motivation

The central question is why the LoRA adapter changed style and explanatory framing more reliably
than factual cosmological stance. This codebase compares:

1. the base Qwen model;
2. the same model with the existing LoRA adapter enabled.

The initial implementation is a minimal vertical slice: prompt validation, hidden-state extraction,
activation storage, and grouped layer-wise linear probes for binary cosmological stance.

## Mechanistic Hypotheses

- New representation: fine-tuning may create or sharpen a representation of geocentric stance.
- Latent-mode amplification: the base model may already contain a premodern mode that the adapter
  amplifies without changing factual readout reliably.
- Routing/readout change: the adapter may alter how existing features influence next-token behavior.
- Mostly stylistic adaptation: the adapter may primarily shift diction, attribution, and explanatory
  style while leaving stance representations mostly intact.

The first probe pipeline is correlational. Any apparent layer-wise decodability must be followed by
causal interventions such as activation steering, patching, and selective adapter ablations.

## Installation

Use Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[accelerate]"
python -m pip install -e ".[quantization]"
```

Model weights, Hugging Face caches, and LoRA adapter files are not committed to Git. Configure local
paths in YAML or environment variables. `.env.example` contains only safe placeholders.

## Local Model References

This repo is configured to reuse the original `chat-ptolemaic` artifacts in place, without copying
weights or adapter files. The expected sibling layout is:

```text
Models/
  chat-ptolemaic/
    outputs/qwen25-7b-astronomy-qlora-run2_04212026_FINAL/
  chat-ptolemaic-mechinterp/
```

The default configs point to:

```yaml
model_name_or_path: Qwen/Qwen2.5-7B
adapter_path: ../chat-ptolemaic/outputs/qwen25-7b-astronomy-qlora-run2_04212026_FINAL
```

The adapter directory includes tokenizer files, so LoRA runs load the tokenizer from that directory
and the base model weights from `Qwen/Qwen2.5-7B`.

## Layout

```text
configs/                    YAML configs for model loading, extraction, and probes
data/prompts/               JSONL prompt datasets
src/ptolemaic_mechinterp/   reusable package code
scripts/                    command-line entry points
tests/                      CPU-only unit tests
results/                    generated activations, probes, and comparisons
```

## Prompt Format

Each JSONL row is validated as:

```python
prompt_id: str
text: str
template_family: str
stance: str | None
style: str | None
framework: str | None
attribution: str | None
metadata: dict[str, Any]
```

The first supported probe target is `stance`, with labels `geocentric` and `heliocentric`. The
example dataset in `data/prompts/example_prompts.jsonl` is deliberately small and only demonstrates
the pipeline.

## Example Commands

Base model extraction:

```bash
python scripts/extract_activations.py \
  --config configs/extraction.yaml \
  --condition base
```

LoRA extraction:

```bash
python scripts/extract_activations.py \
  --config configs/extraction.yaml \
  --condition lora
```

Probe training:

```bash
python scripts/train_probes.py \
  --config configs/probes.yaml \
  --activations results/activations/base
```

Model comparison:

```bash
python scripts/compare_models.py \
  --base-results results/probes/base.csv \
  --lora-results results/probes/lora.csv
```

## Activation Storage

The minimal store uses inspectable files:

- `activations.npz`: compressed NumPy activation matrix;
- `metadata.csv`: one row per prompt and layer;
- `manifest.json`: dimensions, layer numbering, condition, and extraction config.

Layer 0 is explicitly the embedding output returned by Hugging Face `hidden_states`. Transformer
block outputs are layers 1 through N.

## Reproducibility Notes

- Do not use row-level random splits for probes. The implemented probe path uses grouped splits by
  `template_family` to avoid template leakage.
- Tests use synthetic tensors and mock objects, so they do not download Qwen or require CUDA.
- Keep prompt-generation code, labels, model condition, adapter path, and extraction config attached
  to saved activation manifests.
- Treat probe results as descriptive until intervention experiments establish causal relevance.

## Current Limitations

- The example prompts are not scientifically meaningful.
- The first probe task is binary cosmological stance only.
- Activation steering, patching, cross-model patching, probe-transfer evaluation, LoRA projection,
  and selective adapter ablation are represented as lightweight interfaces or placeholders.
- The storage layer is intentionally simple and may later be replaced by Zarr or HDF5 for larger
  runs.

## Planned Experiments

- Controlled prompt generation and labeling.
- Hidden-activation extraction for base and LoRA model conditions.
- Linear probes for stance, historical style, explanatory framework, and attribution.
- Training probes on one model condition and evaluating on the other.
- Activation steering and activation patching.
- LoRA weight and activation analysis.
- Selective adapter ablations by layer or module.
- Behavioral evaluation of generated outputs.
