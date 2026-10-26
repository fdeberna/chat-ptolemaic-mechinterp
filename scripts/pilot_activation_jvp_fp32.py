"""Bounded FP32 arithmetic validation with unchanged effective NF4 weight values."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
from types import MethodType

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import psutil
import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from pilot_activation_jvp_math import sha


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def frozen_digest(core):
    """Stream packed matrices and quant-state tensors; retain at most one host matrix."""
    h=hashlib.sha256()
    for name,module in core.named_modules():
        if not isinstance(module,bnb.nn.Linear4bit): continue
        h.update(name.encode());h.update(tensor_digest(module.weight.data).encode())
        state=module.weight.quant_state
        for value in (state.absmax,state.code,state.offset,state.state2.absmax,state.state2.code):
            if value is not None: h.update(tensor_digest(value).encode())
        h.update(str((state.dtype,state.shape,state.blocksize,state.quant_type)).encode())
    return h.hexdigest()


def reconstructed_weight(module):
    # Preserve EXACT prior FP16-reconstructed values, not a different FP32 NF4 decode.
    decoded=bnb.functional.dequantize_4bit(module.weight.data,module.weight.quant_state)
    if decoded.dtype!=torch.float16: raise RuntimeError('Unexpected NF4 reconstruction dtype')
    return decoded.float()


@contextmanager
def fp32_path(core,audit):
    original_linear=bnb.nn.Linear4bit.forward
    original_embedding=core.embed_tokens.forward
    original_sdpa=torch.nn.functional.scaled_dot_product_attention
    def linear(module,x,*args,**kwargs):
        if x.dtype!=torch.float32: raise RuntimeError(f'Base linear input is {x.dtype}')
        w=reconstructed_weight(module)
        bias=None if module.bias is None else module.bias.float()
        # Native forward AD uses this same temporary fixed W for primal and input tangent.
        result=torch.nn.functional.linear(x,w,bias)
        del w,bias
        return result
    def embedding(module,*args,**kwargs):
        return original_embedding(*args,**kwargs).float()
    def sdpa(q,k,v,*args,**kwargs):
        for name,value in (('Q',q),('K',k),('V',v)):
            p,t=torch.autograd.forward_ad.unpack_dual(value)
            audit.add(('SDPA_'+name,str(p.dtype),None if t is None else str(t.dtype)))
            if p.dtype!=torch.float32 or (t is not None and t.dtype!=torch.float32):
                raise RuntimeError('SDPA input or tangent not FP32')
        result=original_sdpa(q,k,v,*args,**kwargs)
        p,t=torch.autograd.forward_ad.unpack_dual(result)
        audit.add(('SDPA_output',str(p.dtype),None if t is None else str(t.dtype)))
        return result
    bnb.nn.Linear4bit.forward=linear
    core.embed_tokens.forward=MethodType(embedding,core.embed_tokens)
    torch.nn.functional.scaled_dot_product_attention=sdpa
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True), \
             torch.autocast(device_type='cuda',enabled=False):
            yield
    finally:
        bnb.nn.Linear4bit.forward=original_linear
        core.embed_tokens.forward=original_embedding
        torch.nn.functional.scaled_dot_product_attention=original_sdpa


def fd_diagnostics(derivative,plus,minus,h,label):
    fd=(plus.astype(np.float64)-minus.astype(np.float64))/(2*h)
    rows=[]
    for layer in range(28):
        j=derivative[layer].astype(np.float64);f=fd[layer]
        jn=float(np.linalg.norm(j));fn=float(np.linalg.norm(f))
        rows.append(dict(prompt=label,layer=layer+1,step_size=h,jvp_norm=jn,fd_norm=fn,
            cosine=float(j@f/(jn*fn)) if min(jn,fn)>1e-12 else np.nan,
            relative_error_denominator_jvp=float(np.linalg.norm(f-j)/jn) if jn>1e-12 else np.nan))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default='results/activation_jvp_fp32_pilot')
    args=parser.parse_args();out=Path(args.output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot.json').exists(): raise RuntimeError('Existing pilot results are preserved; choose a new directory')
    started=time.perf_counter();report=dict(status='running',measurements=[],stage='loading')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    spec=json.loads(Path('results/full_strength_nf4_fp16/activations/series_config.json').read_text())
    adapter=Path(spec['adapter_path']);assert sha(adapter/'adapter_model.safetensors')==spec['adapter_sha256']
    assert sha(adapter/'tokenizer.json')==spec['tokenizer_sha256']
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    metadata=pd.read_csv('results/full_strength_nf4_fp16/activations/t_0p5000/metadata.csv')
    prompts=metadata[metadata.layer_index==0].reset_index(drop=True)
    assert prompts.prompt_id.tolist()==spec['prompt_ids']
    lengths=[len(tokenizer(x,add_special_tokens=True)['input_ids']) for x in prompts.prompt_text]
    longest=int(np.argmax(lengths));assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    snapshot=Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{spec['model_revision']}"
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,
        quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    model.eval();model.requires_grad_(False);core=model.get_base_model().model
    loading_peak=torch.cuda.max_memory_allocated()
    # No full embedding promotion and no vocabulary-head invocation. Bound temporary decode/cast.
    largest=max(m.in_features*m.out_features for m in core.modules() if isinstance(m,bnb.nn.Linear4bit))
    free,total=torch.cuda.mem_get_info()
    estimate=largest*6+256*1024**2 # one half decode + one float W + generous activation/workspace margin
    report.update(specification=spec,loading_peak_allocated_bytes=loading_peak,
        loading_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        memory_estimate=dict(largest_matrix_elements=largest,half_plus_float_weight_bytes=largest*6,
                            estimated_additional_bytes=estimate,available_bytes=free),
        embedding_strategy='Gather frozen FP16 rows, promote selected activations only; no whole embedding promotion',
        reconstruction='Unchanged NF4 quant_state decodes FP16 values; exact promotion to FP32 per operation',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),autocast=False,math_sdpa=True,
        prompt_short=str(prompts.iloc[0].prompt_id),prompt_longest=str(prompts.iloc[longest].prompt_id),
        token_lengths=dict(short=lengths[0],longest=lengths[longest],mean=float(np.mean(lengths)),minimum=min(lengths)),
        model_parameters_frozen=all(not p.requires_grad for p in model.parameters()))
    handles=[];audit=set();buffer={};token_index=0
    loras=[m for m in core.modules() if isinstance(m,LoraLayer)]
    def scale(t):
        for module in loras:
            for name in module.active_adapters:module.set_scale(name,t)
    def audit_hook(name):
        def hook(module,inputs,output):
            values=[]
            if inputs and isinstance(inputs[0],torch.Tensor) and inputs[0].is_floating_point():values.append(('input',inputs[0]))
            y=output[0] if isinstance(output,tuple) else output
            if isinstance(y,torch.Tensor) and y.is_floating_point():values.append(('output',y))
            for side,value in values:
                p,t=torch.autograd.forward_ad.unpack_dual(value)
                audit.add((name+':'+side,str(p.dtype),None if t is None else str(t.dtype)))
                if p.dtype!=torch.float32 or (t is not None and t.dtype!=torch.float32):
                    raise RuntimeError(f'Non-FP32 audited value: {name}:{side}')
        return hook
    def pool(layer):
        def hook(module,inputs,output):
            output=output[0] if isinstance(output,tuple) else output
            p,t=torch.autograd.forward_ad.unpack_dual(output[0,token_index,:])
            assert p.grad_fn is None and (t is None or t.grad_fn is None)
            buffer[layer]=(p.detach().cpu().numpy().copy(),None if t is None else t.detach().cpu().numpy().copy())
        return hook
    def measure(label,operation):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        try:return operation()
        finally:
            torch.cuda.synchronize()
            report['measurements'].append(dict(stage=label,seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                host_rss_bytes=psutil.Process().memory_info().rss,peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    def run(index,t,jvp=False):
        nonlocal token_index
        buffer.clear();encoded=tokenizer(prompts.iloc[index].prompt_text,return_tensors='pt',add_special_tokens=True)
        token_index=int(encoded['attention_mask'].sum())-1
        encoded={k:v.to('cuda:0') for k,v in encoded.items()}
        with torch.no_grad(),fp32_path(core,audit):
            if jvp:
                with torch.autograd.forward_ad.dual_level():
                    scalar=torch.tensor(t,dtype=torch.float32,device='cuda:0')
                    scale(torch.autograd.forward_ad.make_dual(scalar,torch.ones_like(scalar)))
                    result=core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
                    del result
            else:
                scale(t);result=core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True);del result
        h=np.stack([buffer[i][0] for i in range(1,29)])
        d=np.stack([buffer[i][1] for i in range(1,29)]) if jvp else None
        if not np.isfinite(h).all() or (d is not None and not np.isfinite(d).all()):raise RuntimeError('Nonfinite pooled outputs')
        return h,d
    all_fd=[]
    try:
        if free<estimate:raise RuntimeError('Memory estimate insufficient before any promotion; stop without OOM retry')
        report['packed_state_digest_before']=frozen_digest(core)
        report['stage']='FP32 linear reference validation'
        records=[]
        for name in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
            module=core.layers[0].get_submodule(name).base_layer
            x=torch.randn(1,lengths[0],module.in_features,device='cuda:0',dtype=torch.float32)*.1
            dx=torch.randn_like(x)*.1
            with torch.no_grad(),fp32_path(core,audit),torch.autograd.forward_ad.dual_level():
                weight=reconstructed_weight(module)
                bias=None if module.bias is None else module.bias.float()
                expected=torch.nn.functional.linear(x,weight,bias)
                expected_d=torch.nn.functional.linear(dx,weight,None)
                dual=torch.autograd.forward_ad.make_dual(x,dx)
                actual=module(dual);p,d=torch.autograd.forward_ad.unpack_dual(actual)
                ep=float((p-expected).abs().max());ed=float((d-expected_d).abs().max())
                assert ep==0. and ed==0.
                records.append(dict(module=name,input_shape=list(x.shape),weight_shape=list(weight.shape),
                    primal_max_abs_error=ep,tangent_max_abs_error=ed,dtype=str(p.dtype)))
                del weight,bias,expected,expected_d,actual,p,d,dual
            del x,dx,module
        report['linear_validation']=records
        for name,module in core.named_modules():
            if name and (isinstance(module,(torch.nn.Embedding,torch.nn.Linear,LoraLayer)) or
                         'layernorm' in name or name in ('norm','rotary_emb') or name.endswith('self_attn') or
                         (name.startswith('layers.') and name.count('.')==1) or name.endswith('act_fn')):
                handles.append(module.register_forward_hook(audit_hook(name)))
        for i in range(27):handles.append(core.layers[i].register_forward_hook(pool(i+1)))
        handles.append(core.norm.register_forward_hook(pool(28)))
        report['stage']='short FP32 ordinary and JVP'
        measure('short ordinary cold',lambda:run(0,.5))
        ordinary,_=measure('short ordinary warm',lambda:run(0,.5))
        h,d=measure('short JVP',lambda:run(0,.5,True))
        report['short_primal_max_abs_error']=float(np.abs(h-ordinary).max());assert np.array_equal(h,ordinary)
        np.savez_compressed(out/'short_jvp.npz',pooled=h,derivative=d,layers=np.arange(1,29),t=.5)
        report['stage']='short FP32 finite differences'
        for step in (.04,.02,.01,.005,.001):
            plus,_=run(0,.5+step);minus,_=run(0,.5-step)
            all_fd+=fd_diagnostics(d,plus,minus,step,'short')
        pd.DataFrame(all_fd).to_csv(out/'centered_fd_checks.csv',index=False)
        # Require TWO adjacent modest steps to pass EVERY layer, not just an aggregate median.
        df=pd.DataFrame(all_fd)
        good=[]
        for step in (.04,.02,.01,.005,.001):
            group=df[df.step_size==step]
            good.append(bool((group.cosine>.999).all() and (group.relative_error_denominator_jvp<.02).all()))
        report['short_success_screen']='two adjacent steps: every layer cosine >0.999 and relative error <2%'
        report['short_steps_passing_screen']=dict(zip(map(str,(.04,.02,.01,.005,.001)),good))
        success=any(a and b for a,b in zip(good[:-1],good[1:]));report['short_success']=success
        if success:
            report['stage']='longest reduced FP32 validation'
            ordinary,_=measure('longest ordinary',lambda:run(longest,.5))
            h,d=measure('longest JVP',lambda:run(longest,.5,True))
            report['longest_primal_max_abs_error']=float(np.abs(h-ordinary).max());assert np.array_equal(h,ordinary)
            np.savez_compressed(out/'longest_jvp.npz',pooled=h,derivative=d,layers=np.arange(1,29),t=.5)
            for step in (.01,.005,.001):
                plus,_=run(longest,.5+step);minus,_=run(longest,.5-step)
                all_fd+=fd_diagnostics(d,plus,minus,step,'longest')
            pd.DataFrame(all_fd).to_csv(out/'centered_fd_checks.csv',index=False)
            report['stage']='zero-strength FP32 validation'
            ordinary,_=measure('zero ordinary',lambda:run(0,0.))
            h,d=measure('zero JVP',lambda:run(0,0.,True))
            assert np.array_equal(h,ordinary) and (np.linalg.norm(d.astype(np.float64),axis=1)>0).all()
            report['zero_primal_max_abs_error']=float(np.abs(h-ordinary).max())
            np.savez_compressed(out/'short_zero_jvp.npz',pooled=h,derivative=d,layers=np.arange(1,29),t=0.)
        report['packed_state_digest_after']=frozen_digest(core)
        assert report['packed_state_digest_before']==report['packed_state_digest_after']
        report['status']='completed' if success else 'short_validation_inconclusive_stopped'
    except (RuntimeError,NotImplementedError,AssertionError,IndexError) as error:
        report.update(status='blocked',error=str(error),traceback=traceback.format_exc())
        (out/'failure_traceback.txt').write_text(report['traceback']);print(report['traceback'],flush=True)
    finally:
        scale(.5)
        for handle in handles:handle.remove()
        buffer.clear();del model,core,loras
        gc.collect();torch.cuda.empty_cache()
        report['elapsed_seconds']=time.perf_counter()-started
        report['peak_host_ram_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['dtype_audit']=[dict(location=x,primal_dtype=p,tangent_dtype=t) for x,p,t in sorted(audit,key=str)]
        report['final_allocated_bytes']=torch.cuda.memory_allocated()
        report['implementation_sha256']=sha(__file__)
        (out/'pilot.json').write_text(json.dumps(report,indent=2,sort_keys=True))
        pd.DataFrame(report['dtype_audit']).to_csv(out/'dtype_audit.csv',index=False)
        print(json.dumps({k:v for k,v in report.items() if k not in ('specification','dtype_audit','traceback')},indent=2),flush=True)


if __name__=='__main__':main()
