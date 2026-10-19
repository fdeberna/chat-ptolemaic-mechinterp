"""Resumable full-grid FP32 h/JVP extraction using the validated pilot path."""
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
import shutil
import time

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import psutil
import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from pilot_activation_jvp_fp32 import fp32_path,frozen_digest,sha

GRID=(0.,.01,.02,.04,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)
ROOT='results/instantaneous_nf4_fp32_jvp'


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False));tmp.replace(path)


def array_digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def store_name(t):return f't_{t:.4f}'.replace('.','p')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default=ROOT)
    args=parser.parse_args();root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    stores=root/'activations';stores.mkdir(exist_ok=True)
    started=time.perf_counter();sessions=root/'extraction_sessions';sessions.mkdir(exist_ok=True)
    session_path=sessions/f'{time.time_ns()}.json'
    spec=json.loads(Path('results/full_strength_nf4_fp16/activations/series_config.json').read_text())
    pilot=json.loads(Path('results/activation_jvp_fp32_pilot_v2/pilot.json').read_text())
    assert pilot['status']=='completed' and pilot['short_success']
    assert sha('scripts/pilot_activation_jvp_fp32.py')==pilot['implementation_sha256']
    metadata=pd.read_csv('results/full_strength_nf4_fp16/activations/t_0p5000/metadata.csv')
    prompts=metadata[metadata.layer_index==0].reset_index(drop=True)
    prompts=prompts.drop(columns=['layer_index','model_condition'],errors='ignore')
    assert prompts.prompt_id.tolist()==spec['prompt_ids']
    adapter=Path(spec['adapter_path'])
    assert sha(adapter/'adapter_model.safetensors')==spec['adapter_sha256']
    assert sha(adapter/'adapter_config.json')==spec['adapter_config_sha256']
    assert sha(adapter/'tokenizer.json')==spec['tokenizer_sha256']
    config=dict(schema='instantaneous-fp32-jvp-v1',model_revision=spec['model_revision'],
        model_name=spec['model_name_or_path'],adapter_path=str(adapter),adapter_sha256=spec['adapter_sha256'],
        adapter_config_sha256=spec['adapter_config_sha256'],tokenizer_sha256=spec['tokenizer_sha256'],
        quantization='NF4 double quantized; unchanged FP16 reconstruction promoted FP32 per operation',
        packed_state_sha256=pilot['packed_state_digest_after'],arithmetic='FP32 primal and tangent',
        attention='SDPA, explicit math backend',tf32=False,autocast=False,eval_mode=True,
        pooling='last nonpadding token; block outputs 1-27 and final normalized output 28',
        layers=list(range(1,29)),strengths=GRID,batch_size=1,use_cache=False,
        parameter_grad=False,reverse_graph=False,tangent='scalar dt=1, native PEFT set_scale(t)*intrinsic scaling',
        prompt_metadata_sha256=hashlib.sha256(prompts.to_csv(index=False).encode()).hexdigest(),
        prompt_ids=prompts.prompt_id.tolist(),
        implementation_sha256={p:sha(p) for p in ('scripts/extract_instantaneous_fp32.py',
            'scripts/pilot_activation_jvp_fp32.py','scripts/pilot_activation_jvp_math.py')},
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','numpy')})
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['series_identifier']=identifier
    config_path=root/'series_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==json.loads(json.dumps(config))
    else:atomic_json(config_path,config)
    prompts.to_csv(root/'prompts.csv',index=False)
    reconstruction=root/'reconstruction';reconstruction.mkdir(exist_ok=True)
    for p in config['implementation_sha256']:shutil.copyfile(p,reconstruction/Path(p).name)
    # Compatible completed stores need no model instance or GPU allocation on resume.
    pending=[]
    for t in GRID:
        path=stores/store_name(t)
        if (path/'manifest.json').exists():
            manifest=json.loads((path/'manifest.json').read_text());assert manifest['series_identifier']==identifier
            assert sha(path/'h.npy')==manifest['h_sha256'] and sha(path/'h_dot.npy')==manifest['h_dot_sha256']
        else:pending.append(t)
    if not pending:print('All 1680 prompt-strength pairs already complete',flush=True);return
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    snapshot=Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{spec['model_revision']}"
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,
        quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    model.eval();model.requires_grad_(False);core=model.get_base_model().model
    assert frozen_digest(core)==config['packed_state_sha256']
    loras=[m for m in core.modules() if isinstance(m,LoraLayer)]
    audit=set();handles=[];buffer={};token_index=0
    def capture(layer):
        def hook(module,inputs,output):
            y=output[0] if isinstance(output,tuple) else output
            p,d=torch.autograd.forward_ad.unpack_dual(y[0,token_index,:])
            if d is None or p.dtype!=torch.float32 or d.dtype!=torch.float32:
                raise RuntimeError('Missing or non-FP32 pooled primal/tangent')
            assert p.grad_fn is None and d.grad_fn is None
            buffer[layer]=(p.detach().cpu().numpy().copy(),d.detach().cpu().numpy().copy())
        return hook
    for i in range(27):handles.append(core.layers[i].register_forward_hook(capture(i+1)))
    handles.append(core.norm.register_forward_hook(capture(28)))
    report=dict(series_identifier=identifier,status='running',completed_new_pairs=0,
        device_placement=sorted({str(p.device) for p in model.parameters()}),
        parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
        gpu=torch.cuda.get_device_name(0),forward_seconds=0.,stage='extracting')
    try:
        for t in pending:
            path=stores/store_name(t);path.mkdir(exist_ok=True)
            checkpoints=path/'completed';checkpoints.mkdir(exist_ok=True)
            arrays=[]
            for name in ('h.npy','h_dot.npy'):
                target=path/name
                arrays.append(np.lib.format.open_memmap(target,mode='r+' if target.exists() else 'w+',dtype=np.float32,shape=(120,28,3584)))
            h_store,d_store=arrays
            token_counts=[];times=[]
            for index,prompt in prompts.iterrows():
                checkpoint=checkpoints/f'{index:03d}.json'
                if checkpoint.exists():
                    done=json.loads(checkpoint.read_text())
                    assert done['series_identifier']==identifier and done['prompt_id']==prompt.prompt_id and done['t']==t
                    assert array_digest(h_store[index])==done['h_sha256'] and array_digest(d_store[index])==done['h_dot_sha256']
                    token_counts.append(done['tokens']);times.append(done['seconds']);continue
                encoded=tokenizer(prompt.prompt_text,return_tensors='pt',add_special_tokens=True)
                token_index=int(encoded['attention_mask'].sum())-1
                encoded={k:v.to('cuda:0') for k,v in encoded.items()};buffer.clear()
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.no_grad(),fp32_path(core,audit),torch.autograd.forward_ad.dual_level():
                    scalar=torch.tensor(t,dtype=torch.float32,device='cuda:0')
                    dual=torch.autograd.forward_ad.make_dual(scalar,torch.ones_like(scalar))
                    for module in loras:
                        for name in module.active_adapters:module.set_scale(name,dual)
                    result=core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True);del result
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                pooled=np.stack([buffer[k][0] for k in range(1,29)])
                derivative=np.stack([buffer[k][1] for k in range(1,29)])
                if not np.isfinite(pooled).all() or not np.isfinite(derivative).all():
                    raise RuntimeError(f'Nonfinite activation/JVP t={t}, prompt={prompt.prompt_id}')
                h_store[index]=pooled;d_store[index]=derivative
                h_store.flush();d_store.flush()
                # Commit metadata only after both row arrays are flushed; orphan rows are recomputed.
                atomic_json(checkpoint,dict(series_identifier=identifier,prompt_id=prompt.prompt_id,index=index,t=t,
                    h_sha256=array_digest(pooled),h_dot_sha256=array_digest(derivative),tokens=token_index+1,
                    seconds=seconds,finite=True))
                times.append(seconds);token_counts.append(token_index+1)
                report['completed_new_pairs']+=1;report['forward_seconds']+=seconds
                del encoded,pooled,derivative,dual,scalar
                if (index+1)%10==0:
                    report.update(elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                    atomic_json(session_path,report)
                    print(f't={t:g} {index+1}/120 new_pairs={report["completed_new_pairs"]} mean_s={np.mean(times):.3f} '
                          f'elapsed_min={(time.perf_counter()-started)/60:.1f} GPU_peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
            del h_store,d_store,arrays
            atomic_json(path/'manifest.json',dict(series_identifier=identifier,t=t,shape=[120,28,3584],dtype='float32',
                prompt_ids=prompts.prompt_id.tolist(),layers=list(range(1,29)),h_file='h.npy',h_dot_file='h_dot.npy',
                h_sha256=sha(path/'h.npy'),h_dot_sha256=sha(path/'h_dot.npy'),finite=True,completed_prompts=120,
                mean_jvp_seconds=float(np.mean(times)),forward_seconds=float(np.sum(times)),token_counts=token_counts))
            print(f'COMPLETED strength t={t:g}',flush=True)
        assert frozen_digest(core)==config['packed_state_sha256']
        report['status']='complete';report['packed_state_verified_after']=True
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        for handle in handles:handle.remove()
        buffer.clear();del model,core,loras
        gc.collect();torch.cuda.empty_cache()
        report.update(elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sdpa_audit=[dict(location=n,primal_dtype=p,tangent_dtype=d) for n,p,d in sorted(audit,key=str)])
        atomic_json(session_path,report);atomic_json(root/'extraction_resources.json',report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
