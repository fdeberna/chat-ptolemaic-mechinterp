"""Two-stage, quality-gated own-direction/random steering strength sweep."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from pilot_activation_steering import atomic,sha
from ptolemaic_mechinterp.interventions.steering import residual_steering_hook

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/steering/pilot_L16_stance_L6_style'
PROMPTS=[
 dict(prompt_id='daily_sun_explanation',topic='daily_solar_motion',prompt_text='A child asks why the Sun comes up in the east and goes down in the west each day. The astronomer explains that'),
 dict(prompt_id='lunar_phase_explanation',topic='lunar_phases',prompt_text='A visitor notices that the Moon looks different from one week to the next. The astronomer explains that'),
 dict(prompt_id='seasonal_daylight_explanation',topic='seasonal_daylight',prompt_text='A gardener wonders why summer days are longer than winter days. The astronomer explains that'),
]


def distribution_change(base,steered,tokenizer):
    def logsoftmax(x):
        x=np.asarray(x,dtype=np.float64);x=x-x.max()
        return x-np.log(np.exp(x).sum())
    lp,lq=logsoftmax(base),logsoftmax(steered);p,q=np.exp(lp),np.exp(lq)
    kl=float(p@(lp-lq));assert kl>=-1e-12
    btop=np.argsort(-p,kind='stable')[:5];stop=np.argsort(-q,kind='stable')[:5]
    union=list(dict.fromkeys([int(i) for i in btop]+[int(i) for i in stop]))
    return dict(kl_unsteered_to_steered_nats=max(0.,kl),argmax_changed=int(base.argmax())!=int(steered.argmax()),
        unsteered_argmax_id=int(base.argmax()),steered_argmax_id=int(steered.argmax()),
        unsteered_top5_ids=btop.tolist(),steered_top5_ids=stop.tolist(),
        top_token_probabilities=[dict(token_id=i,token_text=tokenizer.decode([i]),unsteered_probability=float(p[i]),steered_probability=float(q[i])) for i in union])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['low','high'],required=True)
    parser.add_argument('--output-root',default='results/steering/strength_sweep_003_01')
    args=parser.parse_args();out=ROOT/args.output_root
    out.mkdir(parents=True,exist_ok=True)
    for name in ('conditions','prefill_logits','reconstruction'):(out/name).mkdir(exist_ok=True)
    prior=json.loads((OLD/'config.json').read_text())
    hook_path=ROOT/'src/ptolemaic_mechinterp/interventions/steering.py'
    assert sha(hook_path)==prior['sources']['src/ptolemaic_mechinterp/interventions/steering.py']
    config=dict(parent_run_identifier=prior['run_identifier'],prompts=PROMPTS,targets=prior['targets'],scales=prior['scales'],
        directions=['own','random'],strengths=[-.03,.03,-.1,.1],numerical=prior['numerical'],generation=prior['generation'],
        model=prior['model'],revision=prior['revision'],adapter=prior['adapter'],adapter_sha256=prior['adapter_sha256'],adapter_config_sha256=prior['adapter_config_sha256'],tokenizer_sha256=prior['tokenizer_sha256'],versions=prior['versions'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/pilot_activation_steering.py',hook_path,OLD/'config.json',OLD/'directions.npz')},
        quality_gate='Read all ±0.03 continuations against corresponding baselines for coherence, relevance and repetition; per model/target decision recorded before ±0.1. Substantial deterioration blocks escalation. No further strengths.',
        prefix_diagnostic='KL(p_unsteered || p_steered), nats, float64 log-softmax from first-forward logits on identical prompt input IDs; top 5 union; no sampled-generation-prefix comparison')
    import hashlib,importlib.metadata
    assert config['versions']=={p:importlib.metadata.version(p) for p in prior['versions']}
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['run_identifier']=identifier
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'Incompatible resume'
    atomic(out/'config.json',config);pd.DataFrame(PROMPTS).to_csv(out/'prompts.csv',index=False)
    # Commit the development set and code before inspecting any new generations.
    shutil.copy2(__file__,out/'reconstruction'/Path(__file__).name);shutil.copy2(hook_path,out/'reconstruction/steering.py')
    decisions=None;gate_hash=None
    if args.stage=='high':
        gate_path=out/'quality_gate.json';decisions=json.loads(gate_path.read_text());gate_hash=sha(gate_path)
        assert decisions['run_identifier']==identifier and decisions['all_low_conditions_reviewed']
        low=[json.loads(p.read_text()) for p in (out/'conditions').glob('*.json')]
        assert sum(r['k']!=0 and abs(r['k'])==.03 for r in low)==48
        assert {g['model']+'_'+g['target'] for g in decisions['groups']}=={'base_stance','base_style','lora_stance','lora_style'}
    directions=np.load(OLD/'directions.npz')
    tick=time.perf_counter();session=str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert torch.cuda.is_available(),'GPU execution requires approved access outside sandbox'
    torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
    adapter=Path(config['adapter']);snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{config["revision"]}'
    assert sha(adapter/'adapter_model.safetensors')==config['adapter_sha256']
    assert sha(adapter/'adapter_config.json')==config['adapter_config_sha256']
    assert sha(adapter/'tokenizer.json')==config['tokenizer_sha256']
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    base_tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    assert all(tokenizer(p['prompt_text'])['input_ids']==base_tokenizer(p['prompt_text'])['input_ids'] for p in PROMPTS)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'},'No CPU offload permitted'
    core=model.get_base_model().model
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    def resources():return dict(elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def identity(condition,prompt,target,source,k):
        data=dict(run_identifier=identifier,model=condition,prompt_id=prompt['prompt_id'],target=target,direction_source=source,k=k)
        cid=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]
        return data,cid
    @torch.inference_mode()
    def run(condition,prompt,target=None,source=None,k=0.):
        data,cid=identity(condition,prompt,target,source,k);path=out/'conditions'/f'{cid}.json';lp=out/'prefill_logits'/f'{cid}.npz'
        if path.exists():
            record=json.loads(path.read_text());assert record['run_identifier']==identifier and sha(lp)==record['prefill_logits_sha256']
            with np.load(lp) as z:return record,z['logits'].copy()
        encoded={a:b.to('cuda:0') for a,b in tokenizer(prompt['prompt_text'],return_tensors='pt',add_special_tokens=True).items()}
        last=int(encoded['attention_mask'].sum())-1;layer=config['targets'].get(target)
        gamma=k*config['scales'][target] if target else 0.
        ctx=nullcontext();state=None
        if target:
            name=condition if source=='own' else 'random_'+condition
            u=np.asarray(directions[f'{target}__{name}']);assert u.shape==(3584,) and np.isfinite(u).all() and abs(np.linalg.norm(u)-1)<1e-12
            ctx=residual_steering_hook(core.layers[layer-1],torch.tensor(u,device='cuda:0'),gamma,last)
        adapter_context=model.disable_adapter() if condition=='base' else nullcontext()
        torch.cuda.synchronize();start=time.perf_counter();tokens=[];cache=None;ids=encoded['input_ids'];attention=encoded['attention_mask'];first=None
        with adapter_context,ctx as state:
            for step in range(96):
                output=model(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True,output_hidden_states=False,return_dict=True,logits_to_keep=1)
                logits=output.logits[0,-1,:];assert torch.isfinite(logits).all()
                if first is None:first=logits.float().cpu().numpy().copy()
                next_id=int(logits.argmax());tokens.append(next_id);cache=output.past_key_values;del output
                if next_id in eos:break
                ids=torch.tensor([[next_id]],device='cuda:0');attention=torch.cat([attention,torch.ones((1,1),dtype=attention.dtype,device='cuda:0')],dim=1)
        del cache
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        record=dict(**data,condition_id=cid,layer=layer,trained_direction_model=condition if source=='own' else None,prompt_text=prompt['prompt_text'],topic=prompt['topic'],gamma=gamma,activation_scale=config['scales'].get(target),adapter_active=condition=='lora',adapter_strength=1 if condition=='lora' else 0,
            random_seed=(20261026+layer*10+(condition=='lora')) if source=='random' else None,
            generation_config=config['generation'],token_ids=tokens,continuation=tokenizer.decode(tokens,skip_special_tokens=True),generated_tokens=len(tokens),runtime_seconds=elapsed,hook=state,prefix_input_ids=encoded['input_ids'][0].tolist(),quality_gate_sha256=gate_hash if abs(k)==.1 else None,baseline_reused_from_previous_pilot=False,resources=resources())
        if target:
            assert state['calls']==len(tokens) and state['prefill_calls']==1 and state['decode_calls']==len(tokens)-1
            assert np.isfinite(state['ratios']).all()
            baseline_id=identity(condition,prompt,None,None,0.)[1]
            base=json.loads((out/'conditions'/f'{baseline_id}.json').read_text());assert base['prefix_input_ids']==record['prefix_input_ids']
            with np.load(out/'prefill_logits'/f'{baseline_id}.npz') as z:diagnostics=distribution_change(z['logits'],first,tokenizer)
            record.update(baseline_condition_id=baseline_id,identical_to_baseline=tokens==base['token_ids'],actual_perturbation_ratio_min=min(state['ratios']),actual_perturbation_ratio_median=float(np.median(state['ratios'])),actual_perturbation_ratio_max=max(state['ratios']),**diagnostics)
        temp=lp.with_suffix('.tmp.npz');np.savez_compressed(temp,logits=first);temp.replace(lp)
        record['prefill_logits_sha256']=sha(lp);atomic(path,record)
        print(f'COMPLETED {len(list((out/"conditions").glob("*.json")))} {condition} {prompt["prompt_id"]} {target} {source} k={k} seconds={elapsed:.2f}',flush=True)
        return record,first
    try:
        for condition in ('base','lora'):
            for prompt in PROMPTS:run(condition,prompt)
            for target in ('stance','style'):
                if decisions and not next(g for g in decisions['groups'] if g['model']==condition and g['target']==target)['proceed_to_01']:continue
                for source in ('own','random'):
                    for k in ((-.03,.03) if args.stage=='low' else (-.1,.1)):
                        for prompt in PROMPTS:run(condition,prompt,target,source,k)
    except Exception as error:
        atomic(out/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise
    finally:
        atomic(out/f'resources_{args.stage}_{session}.json',dict(stage=args.stage,gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),attention=model.config._attn_implementation,**resources()))
    records=[json.loads(p.read_text()) for p in sorted((out/'conditions').glob('*.json'))]
    with (out/'generations.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(out/'generations.csv',index=False)
    print('STAGE COMPLETE '+args.stage,flush=True)


if __name__=='__main__':main()
