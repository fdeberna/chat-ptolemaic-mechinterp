# Data

This directory stores prompt datasets and split definitions. The example JSONL file is only a
pipeline smoke-test dataset; it is not intended to support scientific claims.

Prompt records are JSON objects with:

- `prompt_id`
- `text`
- `template_family`
- nullable categorical labels: `stance`, `style`, `framework`, `attribution`
- `metadata`, a free-form JSON object

Generated activations and probe outputs should be written under `results/`, not committed.

