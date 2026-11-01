# Default generation protocol

`generation_reproduced_v1.json` is the frozen default for subsequent generation
experiments unless the user explicitly authorizes a labeled override. It contains
the complete generation configuration, model revision, NF4/FP16 loading recipe,
tokenization, seed derivation, versions and replay provenance. It reproduced ten
historical sample-0 answers for both base and the final adapter exactly.

Key settings: sampling, temperature 0.6, top_p 0.9, inherited-and-recorded top_k 50,
150 new tokens, repetition_penalty 1.1, no_repeat_ngram_size 4, one returned
sequence per call, tokenizer EOS/PAD, no chat template. Historical five samples
are five separately seeded calls, not a generation batch. Reuse the historical
seed policy with base seed 1234 and the exact prompt-file label and sample ID.

Do not silently edit this version. Freeze a new labeled version for an authorized
change. New runners should load this preset and verify its settings and hashes.
Older scripts retain their original defaults to preserve earlier-run provenance;
they are not the default protocol for new experiments.
