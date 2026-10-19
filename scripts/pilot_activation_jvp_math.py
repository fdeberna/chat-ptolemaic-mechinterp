"""Bounded math-SDPA and scoped frozen-4bit forward-AD compatibility pilot."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
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
import bitsandbytes as bnb
import bitsandbytes.autograd._functions as bnb_functions
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


@contextmanager
def frozen_4bit_forward_ad():
    """Process-local replacement, same primal; only packed weights retained in AD context."""
    original=bnb_functions.MatMul4Bit
    class Frozen4BitJVP(original):
        @staticmethod
        def forward(ctx,A,B,out=None,bias=None,quant_state=None):
            if B.requires_grad or (bias is not None and bias.requires_grad):
                raise RuntimeError('This rule requires frozen quantized weights and bias')
            # Absent floating-input tangents are otherwise materialized as zero tensors.
            ctx.set_materialize_grads(False)
            result=original.forward(ctx,A,B,out,bias,quant_state)
            ctx.save_for_forward(B)
            ctx.output_shape=result.shape
            return result

        @staticmethod
        def jvp(ctx,dA,dB,dOut,dBias,dState):
            if dB is not None or dBias is not None:
                raise RuntimeError('Weight and bias tangents must be absent for frozen linear rule')
            B,=ctx.saved_tensors
            if dA is None: return None
            # This is EXACTLY the orientation/cast used by the existing primal forward.
            # Dequantized weights live only during this operation, not in ctx/model state.
            weight=bnb.functional.dequantize_4bit(B,ctx.state).to(ctx.dtype_A).t()
            result=torch.nn.functional.linear(dA.to(ctx.dtype_A),weight,None)
            del weight
            return result
    bnb_functions.MatMul4Bit=Frozen4BitJVP
    try: yield
    finally: bnb_functions.MatMul4Bit=original


def validate_rule(core,token_count,out):
    """Validate all seven actual layer-0 projection shapes with FP16 dual inputs."""
    records=[]
    for name in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                 'mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
        module=core.layers[0].get_submodule(name).base_layer
        x=torch.randn(1,token_count,module.in_features,dtype=torch.float16,device='cuda:0')*.1
        dx=torch.randn_like(x)*.1
        with torch.no_grad():
            ordinary=module(x)
            with frozen_4bit_forward_ad(),torch.autograd.forward_ad.dual_level():
                dual=torch.autograd.forward_ad.make_dual(x,dx)
                computed=module(dual)
                primal,tangent=torch.autograd.forward_ad.unpack_dual(computed)
                weight=bnb.functional.dequantize_4bit(module.weight.data,module.weight.quant_state).to(module.compute_dtype)
                reference=torch.nn.functional.linear(x.to(module.compute_dtype),weight,module.bias).to(x.dtype)
                reference_tangent=torch.nn.functional.linear(dx.to(module.compute_dtype),weight,None).to(x.dtype)
                errors=dict(primal_vs_existing_max_abs=float((primal-ordinary).abs().max()),
                    primal_vs_explicit_max_abs=float((primal-reference).abs().max()),
                    tangent_vs_explicit_max_abs=float((tangent-reference_tangent).abs().max()))
                if any(v!=0. for v in errors.values()):
                    raise RuntimeError(f'Frozen quantized-linear validation failed: {name}, {errors}')
                records.append(dict(module=name,input_shape=list(x.shape),output_shape=list(primal.shape),
                    dequantized_weight_shape=list(weight.shape),input_dtype=str(x.dtype),
                    compute_dtype=str(module.compute_dtype),bias_dtype=None if module.bias is None else str(module.bias.dtype),
                    zero_bias_tangent=True,**errors))
                del weight,reference,reference_tangent,dual,computed,primal,tangent
        del x,dx,ordinary,module
    (out/'quantized_linear_validation.json').write_text(json.dumps(records,indent=2))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default='results/activation_jvp_math_pilot')
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
    loading_peak=dict(peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
    use_rule=False
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
    def call_core(encoded):
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True), \
             (frozen_4bit_forward_ad() if use_rule else nullcontext()):
            return core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
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
                result=call_core(encoded)
                del result
        else:
            scale(t)
            with torch.inference_mode():
                result=call_core(encoded)
                del result
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        primal=np.stack([buffer[layer][0] for layer in range(1,29)])
        if not np.isfinite(primal).all(): raise RuntimeError('Nonfinite pooled primal activations')
        tangent=None
        if jvp:
            if any(buffer[layer][1] is None for layer in range(1,29)):
                raise RuntimeError('Missing forward tangent at a retained layer')
            tangent=np.stack([buffer[layer][1] for layer in range(1,29)])
            if not np.isfinite(tangent).all(): raise RuntimeError('Nonfinite pooled tangents')
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
        backend_change='Flash SDPA -> explicit math SDPA; SDPA API unchanged',loading_peak=loading_peak,
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
            comparison=[]
            for layer in range(28):
                current=ordinary[layer].astype(np.float64);original=old[layer].astype(np.float64)
                comparison.append(dict(layer=layer+1,max_abs_diff=float(np.abs(current-original).max()),
                    relative_difference_denominator_flash=float(np.linalg.norm(current-original)/np.linalg.norm(original)),
                    cosine=float(current@original/(np.linalg.norm(current)*np.linalg.norm(original)))))
            pd.DataFrame(comparison).to_csv(out/'math_vs_flash_short.csv',index=False)
            del old
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
        try:
            primal,derivative,elapsed=measure('short JVP without quantized rule',lambda:run(short,.5,True))
        except NotImplementedError as anticipated:
            initial_trace=traceback.format_exc()
            if 'jvp' not in str(anticipated).lower(): raise
            report['anticipated_bnb_blocker']=str(anticipated)
            (out/'bnb_blocker_traceback.txt').write_text(initial_trace)
            report['stage']='validate frozen quantized-linear JVP rule'
            report['quantized_rule_validation']=measure('quantized rule validation',lambda:validate_rule(core,lengths[short],out))
            use_rule=True
            report['stage']='short full-model JVP with validated quantized rule'
            primal,derivative,elapsed=measure('short full-model JVP with rule',lambda:run(short,.5,True))
        report['scoped_quantized_rule_enabled']=use_rule
        report['jvp_short_seconds']=elapsed
        report['jvp_primal_vs_ordinary_max_abs_diff']=float(np.max(np.abs(primal-ordinary)))
        if report['jvp_primal_vs_ordinary_max_abs_diff']!=0.:
            raise RuntimeError('JVP primal differs from ordinary math-SDPA forward')
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
        if report['longest_primal_max_abs_diff']!=0.:
            raise RuntimeError('Longest JVP primal differs from ordinary math-SDPA forward')
        with np.load('results/full_strength_nf4_fp16/activations/t_0p5000/activations.npz') as a:
            old=a['activations'].reshape(120,29,3584)[longest,1:]
            report['longest_math_vs_flash_max_abs_diff']=float(np.abs(normal-old).max())
            del old
        np.savez_compressed(out/'longest_jvp.npz',pooled=primal,derivative=derivative,layers=np.arange(1,29),t=.5)
        report['stage']='short full-model JVP at t=0'
        normal_zero,_,_=measure('ordinary short zero',lambda:run(short,0.))
        primal,derivative,elapsed=measure('short zero JVP',lambda:run(short,0.,True))
        np.savez_compressed(out/'short_zero_jvp.npz',pooled=primal,derivative=derivative,layers=np.arange(1,29),t=0.)
        report['zero_full_derivative_norm']=float(np.linalg.norm(derivative))
        report['zero_full_primal_vs_ordinary_max_abs_diff']=float(np.abs(primal-normal_zero).max())
        if report['zero_full_derivative_norm']<=0 or report['zero_full_primal_vs_ordinary_max_abs_diff']!=0.:
            raise RuntimeError('Zero-strength full-model derivative/primal check failed')
        report['rough_1680_jvp_seconds_short_based']=1680*report['jvp_short_seconds']
        report['rough_1680_jvp_seconds_longest_based']=1680*report['jvp_longest_seconds']
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
