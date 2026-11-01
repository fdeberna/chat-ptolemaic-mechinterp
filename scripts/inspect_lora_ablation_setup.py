"""Step 1 only: inspect LoRA metadata and source files without tensor/model loading.

Uses the standard library only. Does not import torch, transformers, PEFT or
bitsandbytes, instantiate a model, execute a forward pass, or implement ablation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results/adapter_ablation/step1_inspection')
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use a fresh output directory; existing artifacts are preserved')
    preset_path = ROOT / 'configs/generation_reproduced_v1.json'
    preset = json.loads(preset_path.read_text())
    adapter = Path(preset['adapter'])
    adapter_config_path = adapter / 'adapter_config.json'
    weights_path = adapter / 'adapter_model.safetensors'
    config = json.loads(adapter_config_path.read_text())
    snapshot = Path.home() / ('.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/'
                              f'snapshots/{preset["revision"]}')
    base_config_path = snapshot / 'config.json'
    base = json.loads(base_config_path.read_text())
    expected_targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'}
    versions = {name: importlib.metadata.version(name) for name in preset['versions']}
    assert versions == preset['versions'], 'Environment differs from frozen versions'
    for path in [adapter_config_path, weights_path, adapter / 'tokenizer.json',
                 snapshot / 'generation_config.json']:
        assert sha(path) == preset['source_hashes'][str(path)], f'Frozen source differs: {path}'
    assert config['peft_type'] == 'LORA' and config['bias'] == 'none'
    assert not config['lora_bias'] and not config['use_dora'] and not config['use_rslora']
    assert not config['modules_to_save'] and not config['rank_pattern'] and not config['alpha_pattern']
    assert not config.get('target_parameters') and not config.get('layer_replication')
    assert set(config['target_modules']) == expected_targets
    assert base['num_hidden_layers'] == 28 and base['hidden_size'] == 3584
    rank, alpha = config['r'], config['lora_alpha']
    scaling = alpha / rank
    with weights_path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        assert length < weights_path.stat().st_size - 8
        header_bytes = handle.read(length)
        header = json.loads(header_bytes)
    tensor_meta = {k: v for k, v in header.items() if k != '__metadata__'}
    grouped = {}
    tensors, outside, non_lora = [], [], []
    pattern = re.compile(r'^(base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+))\.lora_([AB])\.weight$')
    for name, info in sorted(tensor_meta.items()):
        match = pattern.fullmatch(name)
        row = dict(tensor_key=name, stored_dtype=info['dtype'],
                   shape=json.dumps(info['shape']), parameter_count=math.prod(info['shape']),
                   data_start=info['data_offsets'][0], data_end=info['data_offsets'][1])
        tensors.append(row)
        if not match:
            outside.append(name)
            if '.lora_' not in name:
                non_lora.append(name)
            continue
        module, block, component, projection, factor = match.groups()
        grouped.setdefault(module, {})[factor] = info
        assert 0 <= int(block) < 28 and projection in expected_targets
    assert not outside and not non_lora, 'Unexpected saved modules need explicit review'
    modules = []
    head_dim = base['hidden_size'] // base['num_attention_heads']
    dimensions = {
        'q_proj': (3584, base['num_attention_heads'] * head_dim),
        'k_proj': (3584, base['num_key_value_heads'] * head_dim),
        'v_proj': (3584, base['num_key_value_heads'] * head_dim),
        'o_proj': (3584, 3584), 'gate_proj': (3584, base['intermediate_size']),
        'up_proj': (3584, base['intermediate_size']),
        'down_proj': (base['intermediate_size'], 3584),
    }
    for module, factors in grouped.items():
        assert set(factors) == {'A', 'B'}
        match = re.search(r'layers\.(\d+)\.(self_attn|mlp)\.(\w+)$', module)
        block, component, projection = match.groups()
        block = int(block)
        inputs, outputs = dimensions[projection]
        assert factors['A']['shape'] == [rank, inputs]
        assert factors['B']['shape'] == [outputs, rank]
        assert factors['A']['dtype'] == factors['B']['dtype'] == 'F32'
        modules.append(dict(saved_layer=block+1, block_index=block,
                            base_module_path=module.removeprefix('base_model.model.'),
                            expected_peft_module_path=module,
                            component=component, projection=projection,
                            input_features=inputs, output_features=outputs,
                            rank=rank, alpha=alpha, intrinsic_scaling=scaling,
                            lora_A_shape=json.dumps(factors['A']['shape']),
                            lora_B_shape=json.dumps(factors['B']['shape']),
                            stored_dtype='F32', adapter_bias=False,
                            base_bias_retained=projection in ['q_proj', 'k_proj', 'v_proj'],
                            parameter_count=rank*(inputs+outputs),
                            A_tensor_key=module+'.lora_A.weight',
                            B_tensor_key=module+'.lora_B.weight',
                            expected_A_runtime_parameter=module+'.lora_A.default.weight',
                            expected_B_runtime_parameter=module+'.lora_B.default.weight'))
    modules.sort(key=lambda r: (r['block_index'], r['component'], r['projection']))
    assert len(modules) == 196 and len(tensors) == 392
    layers = []
    for block in range(28):
        selected = [r for r in modules if r['block_index'] == block]
        assert len(selected) == 7 and {r['projection'] for r in selected} == expected_targets
        layers.append(dict(saved_layer=block+1, block_index=block,
                           base_block_path=f'model.layers.{block}',
                           expected_peft_block_path=f'base_model.model.model.layers.{block}',
                           hidden_state_location=('block residual after attention+MLP, before final model.norm'
                                                  if block == 27 else 'block residual after attention+MLP, before next block input norm'),
                           saved_hidden_state_location=('final model.norm(block27 output)' if block == 27
                                                        else f'hidden_states[{block+1}] = block{block} residual output'),
                           module_count=7, attention_modules=4, mlp_modules=3,
                           tensor_count=14,
                           adapter_parameters=sum(r['parameter_count'] for r in selected),
                           planned_single_block_condition=block+1 in [6,16,28]))
    # Read installed source as text only; no library imports or runtime module tree.
    peft_root = Path(importlib.metadata.distribution('peft').locate_file('peft'))
    tf_root = Path(importlib.metadata.distribution('transformers').locate_file('transformers'))
    references = [
        (ROOT/'scripts/run_reproduced_protocol_stance.py', 55, 67, 'Validated frozen loader and parameter freezing'),
        (ROOT/'src/ptolemaic_mechinterp/activations/extraction.py', 104, 118, 'Last non-padding token extraction'),
        (peft_root/'tuners/lora/bnb.py', 509, 564, 'NF4 PEFT forward: base-only disabled branch and scaled BA enabled branch'),
        (peft_root/'tuners/tuners_utils.py', 1511, 1530, 'Per-module enable_adapters; requires_grad side effects'),
        (peft_root/'tuners/lora/layer.py', 183, 210, 'Rank, alpha, dropout and scaling initialization'),
        (tf_root/'models/qwen2/modeling_qwen2.py', 122, 141, 'Attention projections and base QKV biases'),
        (tf_root/'models/qwen2/modeling_qwen2.py', 34, 48, 'MLP projections'),
        (tf_root/'models/qwen2/modeling_qwen2.py', 207, 251, 'Block residual composition'),
        (tf_root/'models/qwen2/modeling_qwen2.py', 263, 269, 'Hidden-state recording target'),
        (tf_root/'models/qwen2/modeling_qwen2.py', 378, 397, 'Final norm after all blocks'),
        (tf_root/'utils/generic.py', 1020, 1038, 'Capture block input then outputs'),
        (tf_root/'utils/generic.py', 1092, 1103, 'Replace final saved hidden state with normalized last_hidden_state'),
    ]
    reference_rows = []
    excerpts = []
    for path, start, end, purpose in references:
        reference_rows.append(dict(path=str(path), start_line=start, end_line=end,
                                   purpose=purpose, sha256=sha(path)))
        lines=path.read_text().splitlines()
        excerpts += [f'## {purpose}', f'{path}:{start}-{end}', '```',
                     *[f'{i+1}: {lines[i]}' for i in range(start-1, min(end,len(lines)))], '```', '']
    save_csv(out/'module_inventory.csv', modules)
    save_csv(out/'tensor_inventory.csv', tensors)
    save_csv(out/'layer_map.csv', layers)
    save_json(out/'adapter_tensor_header.json', header)
    save_json(out/'outside_block_inventory.json', dict(outside_block_tensor_keys=outside,
                                                      non_lora_tensor_keys=non_lora,
                                                      modules_to_save=config['modules_to_save'],
                                                      bias_training=config['bias']))
    conditions = []
    for name, disabled in [('base_adapter_disabled', list(range(1,29))),
                           ('intact_lora', []), ('ablate_L6',[6]),
                           ('ablate_L16',[16]), ('ablate_L28',[28])]:
        conditions.append(dict(condition=name, disabled_saved_layers=disabled,
                               disabled_block_indices=[x-1 for x in disabled],
                               disabled_expected_peft_modules=[m['expected_peft_module_path'] for m in modules
                                                              if m['saved_layer'] in disabled],
                               disabled_module_count=7*len(disabled),
                               global_disable=name=='base_adapter_disabled'))
    spec = dict(status='specification_only_not_implemented_or_runtime_validated',
                scope='ABLATION_PLAN.md step1 only',
                equation='y = base_NF4_linear(x) + m_block * (alpha/r) * B(A(x)); m=0 in selected blocks, 1 elsewhere',
                intrinsic_scaling=scaling, weight_orientation='A[r,in], B[out,r]; delta_W=2*(B@A)',
                base_bias='Retain original QKV biases and all base parameters',
                token_scope='All prompt token positions and every decoding step; persistent parameter-contribution ablation, not token-local residual steering',
                parameter_scope='Only adapter contribution disabled; base block, residuals, norms and all other adapter parameters/scalings unchanged. Downstream inputs may change.',
                dropout='Configured0.05; eval mode makes dropout identity',
                conditions=conditions,
                all_blocks_vs_all_adapter=dict(structural_equivalence=True,
                    evidence='All saved trainable adapter tensors are A/B in blocks0–27; bias=none; modules_to_save=null; no outside adapters',
                    numerical_equivalence='Deferred to step2; not tested here'),
                proposed_later_control='Scoped per-module enable_adapters(False) on exact inventory paths; never merge/unmerge/requantize. Validate loaded unmerged vanilla LoRA first.',
                restoration_caveat='enable_adapters(True) calls set_adapter and may re-enable gradients. Snapshot/restore flags, active adapters, scalings and requires_grad; all parameters must remain frozen. No controller implemented here.',
                cache_policy='Apply condition before prefill and keep through generation; start each condition with a fresh KV cache; never reuse cache across ablations',
                runtime_path_status='Expected names derived from checkpoint and PEFT source, not verified against a loaded model; adapter name default is loader-derived expectation',
                numerical_validation='Not run; step2 is out of scope')
    save_json(out/'intervention_spec.json', spec)
    save_json(out/'source_references.json', reference_rows)
    (out/'source_excerpts.md').write_text('\n'.join(excerpts))
    for path, name in [(preset_path,'generation_reproduced_v1.json'),
                       (adapter_config_path,'adapter_config.json'),
                       (base_config_path,'base_model_config.json'),
                       (ROOT/'ABLATION_PLAN.md','ABLATION_PLAN.md'),
                       (ROOT/'configs/GENERATION_PROTOCOL.md','GENERATION_PROTOCOL.md')]:
        shutil.copyfile(path, out/name)
    sources = [preset_path, adapter_config_path, weights_path, base_config_path,
               snapshot/'generation_config.json', adapter/'tokenizer.json',
               ROOT/'ABLATION_PLAN.md', Path(__file__),
               ROOT/'results/activations/pilot_full/base/manifest.json',
               *[r[0] for r in references]]
    results = dict(step=1, complete=True, model_loaded=False, tensors_materialized=False,
                   forward_passes=0, ablation_implemented=False,
                   adapter_header_bytes=length, tensor_count=len(tensors),
                   lora_modules=len(modules), blocks=len(layers), modules_per_block=7,
                   adapter_parameters=sum(r['parameter_count'] for r in modules),
                   adapter_file_bytes=weights_path.stat().st_size,
                   stored_adapter_dtype='F32', intrinsic_scaling=scaling,
                   outside_block_tensors=0, non_lora_tensors=0,
                   shapes_and_complete_AB_pairs_valid=True,
                   source_hashes_match_frozen_preset=True,
                   versions_match_frozen_preset=True,
                   runtime_and_numerical_checks='Deferred to step2',
                   base_training_revision='adapter_config.revision is null; historical training revision not established. Current generation snapshot is pinned.')
    save_json(out/'validation_results.json', results)
    save_json(out/'provenance.json', dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                 source_sha256={str(path):sha(path) for path in dict.fromkeys(sources)},
                 header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                 installed_versions=versions, model_revision=preset['revision'],
                 inspection_method='stdlib JSON/CSV/source reading and safetensors header parsing only; full adapter file streamed solely for checksum, never deserialized',
                 outputs_not_runtime_configuration=True))
    report = f'''# LoRA ablation: step1 inspection

**Complete: metadata/source inspection only.** No model or tensors loaded, no
ablation implemented, no forwards, numerical checks, prompt selection or generation.

## Adapter inventory

The frozen adapter has **196 LoRA linear modules /392 A+B tensors**, exactly seven
in each of28 decoder blocks: self_attn q_proj/k_proj/v_proj/o_proj and mlp
gate_proj/up_proj/down_proj. There are **{results['adapter_parameters']:,} adapter
parameters**, stored FP32. Rank16, alpha32, intrinsic scale **2**; dropout0.05 is
inactive in evaluation mode. bias=none, lora_bias=false, no DoRA/RSLoRA, no rank or
alpha overrides, no modules_to_save or saved tensors outside decoder blocks.
Original base Q/K/V biases must remain active and untouched.

| Projection | A shape | B shape |
| --- | --- | --- |
| q_proj / o_proj |16×3584|3584×16|
| k_proj / v_proj |16×3584|512×16|
| gate_proj / up_proj |16×3584|18944×16|
| down_proj |16×18944|3584×16|

Shapes and complete A/B pairs checked directly in the safetensors header; weight
values were not materialized. module_inventory.csv enumerates every exact saved
key, expected runtime path, rank/scale, dimensions and parameter count.

## Layer and intervention definition

| Saved label | Decoder block index | Ablation location | Modules disabled |
| --- | --- | --- | --- |
| L6 |5|model.layers.5|all7 attention+MLP adapters|
| L16 |15|model.layers.15|all7 attention+MLP adapters|
| L28 |27|model.layers.27|all7 attention+MLP adapters|

For selected blocks remove the entire adapter term: base_NF4_linear(x) +
0×2×B(A(x)). Elsewhere retain2×B(A(x)). Do not zero base weights, biases,
normalizations or residuals; do not remove the block. Unselected parameters and
scales remain fixed, though their inputs can change as a consequence of ablation.
No residual direction, steering gamma, intercept or threshold is involved.

**Token scope differs from steering:** block-level parameter-contribution ablation
applies to all prompt tokens and every decode step, not just the final prompt token.
Apply before prefill, keep throughout the condition, and start a fresh KV cache for
each condition. It does not inherit the earlier token-local residual hook policy.

Saved activations use the last non-padding token. Saved layers1–27 correspond to
post-block attention+MLP residual outputs. Saved **L28 is final RMSNorm(block27
output)**, not the raw block27 output. Layer0 is the embedding input to block0.
This follows installed Qwen2 hidden-state recording and final-state replacement;
the ablation target for L28 nevertheless remains all LoRA modules in block27.
See layer_map.csv and source_excerpts.md for exact references.

All-block ablation and global adapter disabling are **structurally equivalent for
this export** because no other adapter-trained parameters were saved. Their logits
and token equivalence are not established here; that belongs to step2.

## Frozen configuration and proposed control

Preserve generation_reproduced_v1.json unchanged: pinned Qwen2.5-7B snapshot
{preset['revision']}, existing final adapter/tokenizer,
NF4 double quantization, FP16 base compute, SDPA; batch1, inference/eval, all
parameters frozen, GPU-only single instance, no CPU offload, cache enabled. Sampling
T0.6/top_p0.9/top_k50,150 new tokens, repetition_penalty1.1,
no_repeat_ngram_size4, plain tokenizer and original derived RNG seeds. Adapter
weights are stored FP32; the validated loader retains FP32 LoRA operations and
casts their updates to the base result dtype in PEFT's quantized forward. This is
source/config provenance, not a new measurement of loaded dtypes or placement.
The base config's BF16 default is overridden by the validated FP16 loader.
Generic configs/model.yaml defaults to unquantized/auto and is not a substitute.

Installed PEFT's4-bit forward supports a per-module disabled branch that calls
only base_layer(x). A later scoped controller can use enable_adapters(False) on
these exact modules, with an unmerged-state guard and guaranteed restoration.
Re-enabling can alter requires_grad, so restoration must preserve frozen flags as
well as active adapters/scaling. No controller is implemented or exercised here.
Do not merge/unmerge or change installed packages; reject unexpected merged state.

## Evidence and remaining boundaries

The adapter config, full adapter-file checksum, tokenizer and pinned generation
config match frozen hashes; installed library versions match the preset. Module
paths are expected from saved names/source, **not live named_modules() inspection**.
No actual packed NF4 quantization state, device placement, memory or runtime
behavior was measured here. The exact model revision used for training is not
proven by adapter_config (revision=null); the current generation revision is pinned.
These boundaries must not be described as successful step2 numerical validation.

## Outputs and reproduction

- module_inventory.csv (196 rows), tensor_inventory.csv (392), layer_map.csv (28).
- outside_block_inventory.json, adapter_tensor_header.json: exhaustive saved-key inventory.
- intervention_spec.json: declarative conditions and semantics only, not runnable ablation.
- Frozen generation, adapter/base config and plan copies; source_references.json,
  source_excerpts.md, provenance.json and validation_results.json.
- inspect_lora_ablation_setup.py: this metadata-only inspection script snapshot.

```bash
/home/fdebe/miniconda3/envs/copernicus2/bin/python -B \\
  scripts/inspect_lora_ablation_setup.py --output-dir /tmp/ablation-step1-fresh
```

Use a fresh destination. Step2 implementation/numerical checks and all later
behavioral work remain unexecuted. The reference plan stays untracked and unchanged.
'''
    (out/'report.md').write_text(report)
    shutil.copyfile(__file__, out/Path(__file__).name)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
