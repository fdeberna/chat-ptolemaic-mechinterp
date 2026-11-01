"""Bounded step2 GPU technical checks only; no behavioral generation grid."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import resource
import shutil
import time
import traceback

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from ptolemaic_mechinterp.interventions.lora_ablation import (
    ablate_lora_blocks, adapter_state, validate_ablation_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'results/adapter_ablation/step2_validation')
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use a fresh directory; validation does not overwrite previous results')
    tick = time.perf_counter()
    assert os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
    preset_path = ROOT/'configs/generation_reproduced_v1.json'
    preset = json.loads(preset_path.read_text())
    step1 = ROOT/'results/adapter_ablation/step1_inspection'
    prior = json.loads((step1/'provenance.json').read_text())
    for path, digest in prior['source_sha256'].items():
        assert sha(path) == digest, f'Step1 source changed: {path}'
    inventory = list(csv.DictReader((step1/'module_inventory.csv').open()))
    versions = {name: importlib.metadata.version(name) for name in preset['versions']}
    assert versions == preset['versions']
    old = ROOT/'results/steering/reproduced_protocol_stance_development/generations.jsonl'
    prompt = next(json.loads(line) for line in old.read_text().splitlines()
                  if json.loads(line)['prompt_id'] == 'questions_012'
                  and json.loads(line)['model'] == 'lora'
                  and json.loads(line)['sample_id'] == 0
                  and json.loads(line)['kind'] == 'baseline')
    config = dict(step=2, purpose='technical validation only', preset=preset,
                  prompt_id=prompt['prompt_id'], prompt_text=prompt['prompt_text'],
                  input_ids=prompt['input_ids'], seed=prompt['sample_seed'],
                  selected_saved_layers=[6,16,28], greedy_cache_tokens=8,
                  technical_decoding_override={'do_sample':False,'max_new_tokens':8},
                  thresholds={'logit_atol':1e-5,'logit_rtol':1e-5,'linear_atol':1e-6,'linear_rtol':1e-6},
                  token_scope='all prompt positions and all cached decoding steps',
                  sources={str(p):sha(p) for p in [preset_path, Path(__file__),
                      ROOT/'src/ptolemaic_mechinterp/interventions/lora_ablation.py',
                      step1/'module_inventory.csv', old]})
    atomic(out/'config.json',config)
    shutil.copy2(preset_path, out/preset_path.name)
    shutil.copy2(step1/'module_inventory.csv',out/'module_inventory.csv')
    shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2(ROOT/'src/ptolemaic_mechinterp/interventions/lora_ablation.py',out/'lora_ablation.py')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert torch.cuda.is_available(), 'GPU unavailable; do not substitute CPU/offloading'
    torch.cuda.reset_peak_memory_stats()
    model = None
    tests, operations, generations = [], [], []
    def resources():
        return dict(elapsed_seconds=time.perf_counter()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def compare(label, a, b, atol=1e-5, rtol=1e-5):
        a, b = a.double(), b.double()
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        diff = a-b
        row = dict(check=label, exact=torch.equal(a,b), max_abs=float(diff.abs().max()),
                   relative_l2=float(diff.norm()/b.norm()) if b.norm() else None,
                   denominator='L2 norm of reference logits', atol=atol, rtol=rtol,
                   passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
        tests.append(row)
        atomic(out/'checks_incremental.json',tests)
        assert row['passed'], f'Equivalence failed: {row}'
        return row
    try:
        adapter = Path(preset['adapter'])
        snapshot = Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{preset["revision"]}'
        for p in [adapter/'adapter_model.safetensors', adapter/'adapter_config.json',adapter/'tokenizer.json']:
            assert sha(p) == preset['source_hashes'][str(p)]
        tok = AutoTokenizer.from_pretrained(adapter,use_fast=True,local_files_only=True)
        base_tok = AutoTokenizer.from_pretrained(snapshot,use_fast=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        if base_tok.pad_token is None:base_tok.pad_token=base_tok.eos_token
        for t in [tok,base_tok]:
            assert t(prompt['prompt_text'])['input_ids']==prompt['input_ids']
            assert t.pad_token_id==t.eos_token_id==151643
        quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                    torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},
                    attn_implementation='sdpa',low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
        assert model.get_base_model().config._attn_implementation=='sdpa'
        modules = validate_ablation_inventory(model,inventory)
        initial = adapter_state(model,modules)
        assert all(not r['disabled'] for r in initial['modules'].values())
        atomic(out/'initial_adapter_state.json',initial)
        addresses={name:(p.data_ptr(),tuple(p.shape),str(p.dtype)) for name,p in model.named_parameters()}
        # Hash packed weights and quantization tensors without retaining CPU copies.
        def packed_digest():
            h=hashlib.sha256()
            def add(label,t):
                h.update(label.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
                h.update(t.detach().contiguous().cpu().numpy().tobytes())
            for name,module in modules.items():
                weight=module.base_layer.weight
                assert weight.quant_type=='nf4' and weight.quant_state.nested
                assert module.base_layer.compute_dtype==torch.float16
                add(name+'.packed',weight.data)
                q=weight.quant_state
                for level in [q,q.state2]:
                    if level is not None:
                        for field in ['absmax','code','offset']:
                            value=getattr(level,field,None)
                            if isinstance(value,torch.Tensor):add(name+field,value)
                        h.update(str((level.shape,level.dtype,level.blocksize,level.quant_type)).encode())
            return h.hexdigest()
        packed_before=packed_digest()
        runtime=[]
        for name,module in modules.items():
            runtime.append(dict(name=name,type=type(module).__module__+'.'+type(module).__name__,
                                base_weight_dtype=str(module.base_layer.weight.dtype),
                                compute_dtype=str(module.base_layer.compute_dtype),
                                A_dtype=str(module.lora_A['default'].weight.dtype),
                                B_dtype=str(module.lora_B['default'].weight.dtype),
                                quant_type=module.base_layer.weight.quant_type,nested=True,
                                device=str(module.base_layer.weight.device)))
        atomic(out/'runtime_module_inventory.json',runtime)
        encoded={k:v.to('cuda:0') for k,v in tok(prompt['prompt_text'],return_tensors='pt').items()}
        def reset_seed():
            seed=prompt['sample_seed'];random.seed(seed);np.random.seed(seed)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        @torch.inference_mode()
        def forward(label):
            reset_seed();torch.cuda.synchronize();start=time.perf_counter()
            output=model(**encoded,use_cache=False,logits_to_keep=1,return_dict=True)
            logits=output.logits.detach().cpu();del output
            torch.cuda.synchronize()
            print(f'FORWARD {label}: {time.perf_counter()-start:.2f}s',flush=True)
            return logits
        @contextmanager
        def global_base():
            before=adapter_state(model,modules)
            try:
                with model.disable_adapter():yield
            finally:
                for name,p in model.named_parameters():
                    if p.requires_grad != before['requires_grad'][name]:p.requires_grad_(before['requires_grad'][name])
                assert adapter_state(model,modules)==before
        reference=forward('intact_reference')
        with ablate_lora_blocks(model,inventory,[]):empty=forward('empty_ablation')
        compare('empty versus intact',empty,reference)
        with ablate_lora_blocks(model,inventory,list(range(1,29))):all_disabled=forward('all196_disabled')
        assert adapter_state(model,modules)==initial
        with global_base():base_logits=forward('global_adapter_disabled')
        compare('all196 disabled versus global base',all_disabled,base_logits)
        compare('intact restored after all/global',forward('restored_after_all'),reference)

        @contextmanager
        def observe(label):
            handles=[];base_outputs={}
            def base_hook(name):
                def hook(module,args,output):base_outputs[name]=output
                return hook
            def parent_hook(name):
                def hook(module,args,output):
                    b=base_outputs.pop(name);disabled=bool(module.disable_adapters)
                    if disabled:expected=b
                    else:
                        x=args[0].to(module.lora_A['default'].weight.dtype)
                        delta=module.lora_B['default'](module.lora_A['default'](module.lora_dropout['default'](x)))*module.scaling['default']
                        expected=b+delta.to(b.dtype)
                    error=float((output.float()-expected.float()).abs().max())
                    passed=bool(torch.allclose(output,expected,atol=1e-6,rtol=1e-6))
                    operations.append(dict(condition=label,module=name,disabled=disabled,
                        shape=list(output.shape),max_abs_error=error,exact=torch.equal(output,expected),
                        adapter_update_norm=float((output.float()-b.float()).norm()),passed=passed))
                    assert passed, f'{label}/{name}: linear branch mismatch'
                return hook
            try:
                for name,module in modules.items():
                    handles += [module.base_layer.register_forward_hook(base_hook(name)),module.register_forward_hook(parent_hook(name))]
                yield
                assert not base_outputs
            finally:
                for handle in handles:handle.remove()
                base_outputs.clear()
        selective={}
        for layer in [6,16,28]:
            label=f'ablate_L{layer}'
            with ablate_lora_blocks(model,inventory,[layer]) as state:
                active=adapter_state(model,modules)
                assert sum(v['disabled'] for v in active['modules'].values())==7
                for name,v in active['modules'].items():
                    assert v['disabled']==(name in state['disabled_modules'])
                    assert v['scaling']==initial['modules'][name]['scaling']
                    assert v['active_adapters']==initial['modules'][name]['active_adapters']
                assert not any(p.requires_grad for p in model.parameters())
                atomic(out/(label+'_state.json'),active)
                with observe(label):selective[label]=forward(label)
            assert adapter_state(model,modules)==initial
            compare(label+' restored intact',forward(label+'_restored'),reference)
            group=[r for r in operations if r['condition']==label]
            assert len(group)==196 and sum(r['disabled'] for r in group)==7
            assert all(r['adapter_update_norm']==0 for r in group if r['disabled'])
            assert all(r['adapter_update_norm']>0 for r in group if not r['disabled'])
            atomic(out/'linear_checks_incremental.json',operations)
        class ForcedInterruption(Exception):pass
        try:
            with ablate_lora_blocks(model,inventory,[6,16,28]):
                assert sum(v.disable_adapters for v in modules.values())==21
                raise ForcedInterruption('Technical exception cleanup test')
        except ForcedInterruption:pass
        assert adapter_state(model,modules)==initial
        compare('forced exception restored intact',forward('exception_restored'),reference)
        # Fail-closed guard checks without forwarding or changing base weights.
        guard_results=[]
        for label,bad_inventory,layers,exc in [('duplicate layers',inventory,[6,6],ValueError),
                      ('out of range',inventory,[0],ValueError),
                      ('missing module',inventory[:-1],[6],ValueError)]:
            try:
                with ablate_lora_blocks(model,bad_inventory,layers):raise AssertionError('guard did not reject')
            except exc:guard_results.append(dict(check=label,passed=True))
            assert adapter_state(model,modules)==initial
        # Eight tokens only: no sampling/behavioral scoring and no reused cache.
        gc=GenerationConfig.from_dict(copy.deepcopy(preset['generation_config']))
        gc.do_sample=False;gc.max_new_tokens=8
        gc.temperature=None;gc.top_p=None;gc.top_k=None
        atomic(out/'technical_greedy_generation_config.json',gc.to_dict())
        @torch.inference_mode()
        def generate(label):
            calls={name:[] for name in modules};handles=[]
            def inspect(name):
                def hook(module,args,kwargs):
                    assert not any(p.requires_grad for p in model.parameters())
                    calls[name].append(dict(tokens=int(args[0].shape[1]),disabled=bool(module.disable_adapters)))
                return hook
            try:
                for name,module in modules.items():handles.append(module.register_forward_pre_hook(inspect(name),with_kwargs=True))
                reset_seed();torch.cuda.synchronize();start=time.perf_counter()
                seq=model.generate(**encoded,generation_config=copy.deepcopy(gc),return_dict_in_generate=True,
                                   output_scores=False,output_logits=False,logits_to_keep=1)
                tokens=seq.sequences[0,encoded['input_ids'].shape[1]:].tolist();del seq
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                assert len(tokens)==8, 'Short EOS would leave cache coverage incomplete; stop rather than expand'
                for name,rows in calls.items():
                    assert len(rows)==8 and rows[0]['tokens']==encoded['input_ids'].shape[1]
                    assert all(row['tokens']==1 for row in rows[1:])
                    assert all(row['disabled']==modules[name].disable_adapters for row in rows)
                record=dict(condition=label,tokens=tokens,continuation=tok.decode(tokens,skip_special_tokens=True),
                            runtime_seconds=elapsed,per_module_calls=calls,
                            all_modules_prefill_calls=196,all_modules_decode_calls=196*7)
                generations.append(record);atomic(out/(label+'_cache.json'),record)
                print(f'CACHE {label}:8 tokens in {elapsed:.2f}s',flush=True)
                return tokens
            finally:
                for h in handles:h.remove()
        intact_tokens=generate('intact')
        with ablate_lora_blocks(model,inventory,[]):empty_tokens=generate('empty')
        assert empty_tokens==intact_tokens
        with ablate_lora_blocks(model,inventory,list(range(1,29))):all_tokens=generate('all_disabled')
        with global_base():base_tokens=generate('global_base')
        assert all_tokens==base_tokens
        for layer in [6,16,28]:
            with ablate_lora_blocks(model,inventory,[layer]):generate(f'L{layer}')
            assert adapter_state(model,modules)==initial
        restored_tokens=generate('restored_intact')
        assert restored_tokens==intact_tokens
        assert adapter_state(model,modules)==initial
        assert {name:(p.data_ptr(),tuple(p.shape),str(p.dtype)) for name,p in model.named_parameters()}==addresses
        packed_after=packed_digest();assert packed_before==packed_after
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        np.savez(out/'next_token_logits.npz',intact=reference.numpy(),empty=empty.numpy(),
                 all_disabled=all_disabled.numpy(),global_base=base_logits.numpy(),
                 **{k:v.numpy() for k,v in selective.items()})
        with (out/'linear_checks.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(operations[0]));writer.writeheader();writer.writerows(operations)
        result=dict(passed=True,forward_equivalence_checks=tests,
                    selective_linear_checks=len(operations),all_selective_linear_checks_exact=all(r['exact'] for r in operations),
                    guard_checks=guard_results,exception_restoration=True,
                    initial_and_final_adapter_states_equal=True,all_parameters_frozen=True,
                    parameter_addresses_shapes_dtypes_unchanged=True,
                    packed_NF4_and_quantization_tensor_sha256_before=packed_before,
                    packed_NF4_and_quantization_tensor_sha256_after=packed_after,
                    packed_weights_and_quantization_unchanged=True,
                    cached_runs=len(generations),total_generated_tokens=sum(len(r['tokens']) for r in generations),
                    empty_intact_token_match=True,all_global_base_token_match=True,
                    restored_intact_token_match=True,all_cache_counts_valid=True,
                    hooks_removed=True,behavioral_grid_run=False,**resources())
        atomic(out/'validation_results.json',result)
        report=['# LoRA block ablation: Step2 technical validation','',
                '**PASS.** This was a bounded implementation/technical check, not a behavioral steering experiment.','',
                '## Scope and implementation','',
                'Scoped ablate_lora_blocks disables all seven PEFT LoRA branches in each selected block with public per-module enable_adapters(False). It rejects unexpected module inventories/types, adapter variants, merged state, ranks/scales/dimensions, training mode and unfrozen parameters. Exact initial adapter flags, active adapters, scales and requires_grad states restore on normal and exceptional exits. No base weights, biases, norms or residuals are removed; no merge, requantization, residual hooks or dependency changes. Conditions apply to all tokens, starting before prefill with fresh caches.','',
                '## Results','',
                f'All {len(tests)} logit equivalence/restoration checks passed; exact matches: {sum(x["exact"] for x in tests)}/{len(tests)}. Logit tolerances were fixed before execution: atol=rtol=1e-5; both max absolute error and relative L2/reference-norm errors are in checks_incremental.json. Exactly196 runtime modules matched Step1. Each L6/L16/L28 condition disabled exactly7 and retained189 adapters/scales. {len(operations)} same-operation linear checks passed (exact={result["all_selective_linear_checks_exact"]}); selected outputs equal their base-layer outputs and unselected outputs equal base+scaled LoRA on the identical inputs. Selected update norms are zero; unselected update norms are nonzero. No comparison assumes downstream activations stay unchanged.','',
                'Empty ablation matches intact LoRA; all196 disabled matches global adapter-disabled base. Intact logits recover after every selective condition, all/global disabling and a forced exception. Invalid duplicate/out-of-range/missing-module requests fail closed. Forced merged-state/variant mutation was not performed; those guards are code-level checks.','',
                f'{len(generations)} fresh-cache greedy calls ×8 tokens = {result["total_generated_tokens"]} diagnostic tokens, all saved. Empty/intact, all-disabled/global-base, and restored/intact token sequences match. All196 modules were checked at prefill and each of7 cached token-processing steps for every call, including L6/L16/L28. The eighth generated token is not fed back because generation stops there. No extra prompt, seed, decoding search or behavioral assessment.','',
                'All parameters remain frozen; adapter states and parameter addresses/shapes/dtypes restore. Packed NF4 weights and quantization tensors have matching streamed SHA256 fingerprints before/after. Every observer hook is removed.','',
                '## Configuration and resources','',
                f'Existing questions_012 prompt/sample0; exact saved input IDs and SHA256-derived seed. One PEFT-wrapped pinned Qwen2.5-7B instance; base evaluated through adapter disabling. NF4 double quantization, FP16 compute/storage, FP32 LoRA factors, SDPA, all parameters on cuda:0, batch1, inference/eval, no offload. Frozen preset hashes and library versions matched. Only the explicitly authorized diagnostic decoding override differs: greedy,8 tokens. Inactive sampling fields cleared; other generation settings retained. Elapsed{result["elapsed_seconds"]:.1f}s; peak GPU allocated/reserved {result["peak_allocated_bytes"]/2**30:.2f}/{result["peak_reserved_bytes"]/2**30:.2f}GiB; host RSS{result["peak_host_ram_bytes"]/2**30:.2f}GiB. GPU allocator excludes driver/context; RSS includes loading/file-backed memory.','',
                '## Limits and next boundary','',
                'These checks establish technical selective removal/restoration under the inspected configuration and this one short input. They do not establish stance/style effects, behavior on longer prompts, causal uniqueness of a block, or additive effects across blocks. No prompt selection or Step3 grid was launched. Existing steering conclusions remain unchanged.','',
                '## Artifacts and reproduction','',
                'config.json, runtime_module_inventory.json, initial/selective state snapshots, checks_incremental.json, linear_checks.csv, next_token_logits.npz, per-condition cache JSON, packed-state hashes, validation_results.json, resources.json and script/controller snapshots. No full activations or dequantized weights are saved.','',
                '```bash\nCUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/validate_lora_ablation.py --output-dir /tmp/lora-ablation-step2-fresh\n```','',
                'Use a fresh destination; no overwriting or automatic retry after failure/OOM. This command reruns only this bounded technical check, not a behavioral grid.']
        (out/'report.md').write_text('\n'.join(report)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),**resources()))
        raise
    finally:
        atomic(out/'resources.json',dict(gpu=torch.cuda.get_device_name(0),
                     placement=sorted({str(p.device) for p in model.parameters()}) if model else [],**resources()))


if __name__=='__main__':main()
