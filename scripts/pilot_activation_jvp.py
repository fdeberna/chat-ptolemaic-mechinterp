"""Bounded forward-AD pilot with unchanged NF4/FP16/SDPA extraction settings."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default='results/activation_jvp_pilot')
    args=parser.parse_args();out=Path(args.output_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot.json').exists(): raise RuntimeError('Choose a new output directory; existing pilot is preserved')
    started=time.perf_counter()
    spec=json.loads(Path('results/full_strength_nf4_fp16/activations/series_config.json').read_text())
    adapter=Path(spec['adapter_path'])
    assert sha(adapter/'adapter_model.safetensors')==spec['adapter_sha256']
    assert sha(adapter/'tokenizer.json')==spec['tokenizer_sha256']
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(spec['seed'])
    torch.cuda.reset_peak_memory_stats()
    md=pd.read_csv('results/full_strength_nf4_fp16/activations/t_0p5000/metadata.csv')
    prompts=md[md.layer_index==0].reset_index(drop=True)
    assert prompts.prompt_id.tolist()==spec['prompt_ids']
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    lengths=[len(tokenizer(text,add_special_tokens=True)['input_ids']) for text in prompts.prompt_text]
    short=0;longest=int(np.argmax(lengths))
    snapshot=Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{spec['model_revision']}"
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=spec['quant_type'],
        bnb_4bit_use_double_quant=spec['double_quant'],bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,
        quantization_config=config,device_map={'':0},attn_implementation=spec['attention'],low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    model.eval();model.requires_grad_(False)
    core=model.get_base_model().model
    loras=[module for module in model.modules() if isinstance(module,LoraLayer)]
    def scale(t):
        # Pass the dual tensor directly; the older CLI helper's float(t) would destroy AD.
        # Native set_scale preserves alpha/r (or alpha/sqrt(r)) and is noncumulative.
        for module in loras:
            for name in module.active_adapters: module.set_scale(name,t)
    handles=[];buffer={};token_index=0
    def hook(layer):
        def capture(module,inputs,output):
            tensor=output[0] if isinstance(output,tuple) else output
            pooled=tensor[0,token_index,:]
            primal,tangent=torch.autograd.forward_ad.unpack_dual(pooled)
            assert primal.grad_fn is None
            buffer[layer]=(primal.detach().float().cpu().numpy().copy(),
                None if tangent is None else tangent.detach().float().cpu().numpy().copy())
        return capture
    for i in range(27): handles.append(core.layers[i].register_forward_hook(hook(i+1)))
    handles.append(core.norm.register_forward_hook(hook(28)))
    def run(index,t,jvp=False):
        nonlocal token_index
        buffer.clear()
        encoded=tokenizer(prompts.iloc[index].prompt_text,return_tensors='pt',add_special_tokens=True)
        token_index=int(encoded['attention_mask'].sum())-1
        encoded={k:v.to('cuda:0') for k,v in encoded.items()}
        torch.cuda.synchronize();tick=time.perf_counter()
        if jvp:
            # no_grad disables reverse recording, but unlike inference_mode permits forward AD.
            with torch.no_grad(),torch.autograd.forward_ad.dual_level():
                scalar=torch.tensor(t,dtype=torch.float32,device='cuda:0')
                dual=torch.autograd.forward_ad.make_dual(scalar,torch.ones_like(scalar))
                scale(dual)
                result=core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
                del result
        else:
            scale(t)
            with torch.inference_mode():
                result=core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
                del result
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        primal=np.stack([buffer[layer][0] for layer in range(1,29)])
        tangent=None
        if jvp:
            if any(buffer[layer][1] is None for layer in range(1,29)):
                raise RuntimeError('Missing forward tangent at a retained layer')
            tangent=np.stack([buffer[layer][1] for layer in range(1,29)])
        del encoded
        return primal,tangent,elapsed
    report=dict(extraction_specification=spec,ad_method='torch.autograd.forward_ad, scalar dual tangent dt=1 under no_grad',
        model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        parameter_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters())),
        lora_parameter_dtype_counts=dict(Counter(str(p.dtype) for name,p in model.named_parameters() if 'lora_' in name)),
        devices=sorted({str(p.device) for p in model.parameters()}),
        actual_attention_implementation=core.config._attn_implementation,
        actual_4bit_compute_dtypes=sorted({str(m.compute_dtype) for m in model.modules() if hasattr(m,'compute_dtype')}),
        gpu=torch.cuda.get_device_name(0),versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')},
        short_prompt_id=str(prompts.iloc[short].prompt_id),short_tokens=lengths[short],
        longest_prompt_id=str(prompts.iloc[longest].prompt_id),longest_tokens=lengths[longest],
        token_length_min=min(lengths),token_length_mean=float(np.mean(lengths)),token_length_max=max(lengths),
        stage='ordinary short forward',status='running',measurements=[])
    def measure(label,operation):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        try: return operation()
        finally:
            torch.cuda.synchronize()
            report['measurements'].append(dict(stage=label,elapsed_seconds=time.perf_counter()-tick,
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
                current_host_rss_bytes=psutil.Process().memory_info().rss,
                peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    fd_rows=[]
    try:
        ordinary,_,cold=measure('ordinary short cold',lambda:run(short,.5))
        _,_,warm=measure('ordinary short warm',lambda:run(short,.5))
        report['ordinary_short_cold_seconds']=cold;report['ordinary_short_warm_seconds']=warm
        np.savez_compressed(out/'short_ordinary.npz',pooled=ordinary,layers=np.arange(1,29),t=.5,
            prompt_id=np.asarray(report['short_prompt_id']))
        with np.load('results/full_strength_nf4_fp16/activations/t_0p5000/activations.npz') as a:
            old=a['activations'].reshape(120,29,3584)[short,1:]
            report['ordinary_vs_saved_max_abs_diff']=float(np.max(np.abs(ordinary-old)))
            assert report['ordinary_vs_saved_max_abs_diff']==0.
        # An isolated actual LoRA projection verifies that t=0 does not bypass its tangent.
        report['stage']='zero-scale adapter branch verification'
        with torch.no_grad():
            inputs=core.embed_tokens(torch.tensor([tokenizer(prompts.iloc[short].prompt_text)['input_ids']],device='cuda:0'))
            projection=core.layers[0].self_attn.q_proj
            adapter_name=projection.active_adapters[0]
            scale(1.);intrinsic=projection.scaling[adapter_name]
            direct=projection.lora_B[adapter_name](projection.lora_A[adapter_name](
                inputs.to(projection.lora_A[adapter_name].weight.dtype)))*intrinsic
            with torch.autograd.forward_ad.dual_level():
                zero=torch.tensor(0.,dtype=torch.float32,device='cuda:0')
                scale(torch.autograd.forward_ad.make_dual(zero,torch.ones_like(zero)))
                projected=projection(inputs)
                _,tangent=torch.autograd.forward_ad.unpack_dual(projected)
                assert tangent is not None and float(tangent.norm())>0
                expected=direct.to(tangent.dtype)
                report['zero_scale_check']=dict(module='layer0.self_attn.q_proj',intrinsic_scale=float(intrinsic),
                    tangent_norm=float(tangent.norm()),max_abs_diff_from_intrinsic_LoRA_branch=float((tangent-expected).abs().max()))
            del inputs,projection,direct,projected,tangent,expected
        report['stage']='short full-model JVP at t=0.5'
        primal,derivative,elapsed=measure('short full-model JVP',lambda:run(short,.5,True))
        report['jvp_short_seconds']=elapsed
        report['jvp_primal_vs_ordinary_max_abs_diff']=float(np.max(np.abs(primal-ordinary)))
        np.savez_compressed(out/'short_jvp.npz',pooled=primal,derivative=derivative,layers=np.arange(1,29),t=.5)
        report['stage']='centered finite differences on short prompt'
        for step in (.001,.005,.01,.02,.04):
            plus,_,_=run(short,.5+step);minus,_,_=run(short,.5-step)
            fd=(plus.astype(np.float64)-minus.astype(np.float64))/(2*step)
            for layer in range(28):
                j=derivative[layer].astype(np.float64);f=fd[layer];jn=np.linalg.norm(j);fn=np.linalg.norm(f)
                fd_rows.append(dict(layer=layer+1,step_size=step,jvp_norm=jn,fd_norm=fn,
                    relative_error_denominator_jvp=float(np.linalg.norm(f-j)/jn) if jn>1e-12 else np.nan,
                    cosine=float(f@j/(fn*jn)) if min(fn,jn)>1e-12 else np.nan,
                    jvp_near_zero=jn<=1e-12,fd_near_zero=fn<=1e-12))
        pd.DataFrame(fd_rows).to_csv(out/'centered_fd_checks.csv',index=False)
        report['stage']='longest prompt JVP'
        normal,_,normal_time=measure('ordinary longest',lambda:run(longest,.5))
        primal,derivative,elapsed=measure('longest JVP',lambda:run(longest,.5,True))
        report['ordinary_longest_seconds']=normal_time;report['jvp_longest_seconds']=elapsed
        report['longest_primal_max_abs_diff']=float(np.abs(primal-normal).max())
        np.savez_compressed(out/'longest_jvp.npz',pooled=primal,derivative=derivative,layers=np.arange(1,29),t=.5)
        report['stage']='short full-model JVP at t=0'
        primal,derivative,elapsed=measure('short zero JVP',lambda:run(short,0.,True))
        np.savez_compressed(out/'short_zero_jvp.npz',pooled=primal,derivative=derivative,layers=np.arange(1,29),t=0.)
        report['zero_full_derivative_norm']=float(np.linalg.norm(derivative))
        report['status']='completed'
    except (NotImplementedError,RuntimeError) as error:
        report['status']='blocked'
        report['error_type']=type(error).__name__;report['error']=str(error)
        report['traceback']=traceback.format_exc()
        (out/'failure_traceback.txt').write_text(report['traceback'])
        print(report['traceback'],flush=True)
    finally:
        scale(.5)
        for handle in handles: handle.remove()
        buffer.clear();del core,model,loras
        gc.collect();torch.cuda.empty_cache()
        report['elapsed_seconds']=time.perf_counter()-started
        report['peak_host_ram_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['final_allocated_vram_bytes']=torch.cuda.memory_allocated()
        (out/'pilot.json').write_text(json.dumps(report,indent=2,sort_keys=True))
        print(json.dumps({k:v for k,v in report.items() if k not in ('extraction_specification','traceback')},indent=2),flush=True)


if __name__=='__main__': main()
