"""Bounded recovered-default greedy/sampling diagnostic; no steering or fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import copy
from contextlib import nullcontext
import hashlib
import importlib.metadata
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/generation_diagnostics/recovered_defaults_comparison'
PROMPTS=[
 dict(prompt_id='daily_sky',topic='daily_rotation',text='The apparent daily motion of the sky can be explained by'),
 dict(prompt_id='annual_sun',topic='annual_motion',text="The Sun's changing position among the stars over the course of a year can be explained by"),
 dict(prompt_id='retrograde',topic='planetary_retrograde',text="The occasional reversal of a planet's motion against the background stars can be explained by"),
 dict(prompt_id='lunar_appearance',topic='astronomical_explanation',text="The regular changes in the Moon's appearance over a month can be explained by"),
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True));tmp.replace(path)

def main():
    start=time.perf_counter();session=str(time.time_ns())
    for name in ('conditions','verification','reconstruction'):(OUT/name).mkdir(parents=True,exist_ok=True)
    prior=json.loads((ROOT/'results/steering/pilot_L16_stance_L6_style/config.json').read_text())
    adapter=Path(prior['adapter']);snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{prior["revision"]}'
    config=dict(prompts=PROMPTS,development_prompts_not_historical=True,model=prior['model'],revision=prior['revision'],adapter=str(adapter),numerical=prior['numerical'],
        seeds=[0,1,2],max_new_tokens=200,sampling=dict(temperature=.8,top_p=.9),repetition_penalty=1.,no_repeat_ngram_size=0,
        top_k_policy='inherited, never overridden',historical_invocation='unknown; supplied script defaults are recovered defaults only',
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')},
        sources={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/recovered_qwen_generation.py',ROOT/'results/steering/pilot_L16_stance_L6_style/config.json',adapter/'adapter_model.safetensors',adapter/'adapter_config.json',adapter/'tokenizer.json',adapter/'tokenizer_config.json',snapshot/'generation_config.json']})
    assert config['sources'][str(adapter/'adapter_model.safetensors')]==prior['adapter_sha256']
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['run_identifier']=identifier
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config,'Incompatible resume'
    atomic(OUT/'config.json',config);pd.DataFrame(PROMPTS).to_csv(OUT/'prompts.csv',index=False)
    shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name);shutil.copy2(ROOT/'scripts/recovered_qwen_generation.py',OUT/'reconstruction/recovered_qwen_generation.py')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);assert torch.cuda.is_available(),'GPU access requires approved execution'
    torch.cuda.reset_peak_memory_stats()
    tokenizers={}
    for condition,source in [('base',snapshot),('lora',adapter)]:
        tok=AutoTokenizer.from_pretrained(source,use_fast=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        assert tok.is_fast
        tokenizers[condition]=tok
    tokenizer_audit={condition:dict(source=str(snapshot if condition=='base' else adapter),tokenizer_class=type(tok).__name__,is_fast=tok.is_fast,padding_side=tok.padding_side,bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,uses_chat_template=False,
        input_ids={p['prompt_id']:tok(p['text'])['input_ids'] for p in PROMPTS}) for condition,tok in tokenizers.items()}
    atomic(OUT/'tokenization.json',tokenizer_audit)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    def assert_no_interventions():
        # No application hooks may remain on any decoder layer or final norm.
        core=model.get_base_model().model
        assert all(not block._forward_hooks and not block._forward_pre_hooks for block in core.layers)
        assert not core.norm._forward_hooks
    assert_no_interventions()
    def context(condition):return model.disable_adapter() if condition=='base' else nullcontext()
    def encode(condition,prompt):return {k:v.to('cuda:0') for k,v in tokenizers[condition](prompt['text'],return_tensors='pt').items()}
    def resources():return dict(elapsed_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def effective(condition,mode):
        tok=tokenizers[condition];kwargs=dict(max_new_tokens=200,do_sample=mode=='sampling',repetition_penalty=1.,no_repeat_ngram_size=0,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
        if mode=='sampling':kwargs.update(temperature=.8,top_p=.9)
        gc=copy.deepcopy(model.generation_config);unused=gc.update(**kwargs);assert not unused
        return gc
    configs={f'{condition}_{mode}':effective(condition,mode).to_dict() for condition in ('base','lora') for mode in ('greedy','sampling')}
    atomic(OUT/'effective_generation_configs.json',configs)
    print('EFFECTIVE top_k '+json.dumps({k:v['top_k'] for k,v in configs.items()}),flush=True)
    @torch.inference_mode()
    def manual(condition,prompt,capture_step=None):
        inp=encode(condition,prompt);ids=inp['input_ids'];attention=inp['attention_mask'];tokens=[];cache=None;captured=None
        with context(condition):
            for step in range(200):
                output=model(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True,output_hidden_states=False,return_dict=True,logits_to_keep=1)
                logits=output.logits[0,-1,:];assert torch.isfinite(logits).all()
                token=int(logits.argmax());tokens.append(token);cache=output.past_key_values
                if step==capture_step:captured=logits.float().cpu().numpy().copy();break
                del output
                if token==tokenizers[condition].eos_token_id:break
                ids=torch.tensor([[token]],device='cuda:0');attention=torch.cat([attention,torch.ones((1,1),dtype=attention.dtype,device='cuda:0')],dim=1)
        return tokens,captured
    def condition_id(condition,prompt,mode,seed):
        data=dict(run_identifier=identifier,model=condition,prompt_id=prompt['prompt_id'],mode=mode,seed=seed)
        return data,hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]
    @torch.inference_mode()
    def generate(condition,prompt,mode,seed,verification=False):
        data,cid=condition_id(condition,prompt,mode,seed);path=OUT/'conditions'/f'{cid}.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['run_identifier']==identifier
            return row,None
        assert_no_interventions();inputs=encode(condition,prompt);gc=effective(condition,mode)
        torch.manual_seed(0 if seed is None else seed);torch.cuda.manual_seed_all(0 if seed is None else seed)
        np.random.seed(0 if seed is None else seed)
        import random
        random.seed(0 if seed is None else seed)
        torch.cuda.synchronize();tick=time.perf_counter()
        with context(condition):
            output=model.generate(**inputs,generation_config=gc,return_dict_in_generate=True,output_logits=verification,output_scores=False,logits_to_keep=1)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        input_ids=inputs['input_ids'][0].tolist();full=output.sequences[0].tolist();assert full[:len(input_ids)]==input_ids
        continuation=full[len(input_ids):];tok=tokenizers[condition]
        row=dict(**data,condition_id=cid,prompt_text=prompt['text'],topic=prompt['topic'],input_ids=input_ids,generated_token_ids=continuation,continuation=tok.decode(continuation,skip_special_tokens=True),adapter_active=condition=='lora',adapter_strength=1 if condition=='lora' else 0,
            generation_config=gc.to_dict(),tokenizer=tokenizer_audit[condition],generated_tokens=len(continuation),stopping_reason='eos' if continuation and continuation[-1]==tok.eos_token_id else 'max_new_tokens',runtime_seconds=seconds,resources=resources(),no_intervention_hooks_verified=True)
        atomic(path,row);print(f'COMPLETED {len(list((OUT/"conditions").glob("*.json")))}/32 {condition} {prompt["prompt_id"]} {mode} seed={seed} seconds={seconds:.2f}',flush=True)
        return row,output
    try:
        for condition in ('base','lora'):
            vp=OUT/'verification'/f'{condition}.json';prompt=PROMPTS[0]
            if not vp.exists():
                torch.cuda.synchronize();tick=time.perf_counter();tokens,_=manual(condition,prompt);torch.cuda.synchronize();seconds=time.perf_counter()-tick
                row,output=generate(condition,prompt,'greedy',None,verification=True)
                generated=row['generated_token_ids'];same=tokens==generated
                check=dict(model=condition,prompt_id=prompt['prompt_id'],manual_tokens=tokens,generate_tokens=generated,token_ids_identical=same,manual_runtime_seconds=seconds,greedy_condition_id=row['condition_id'],settings=configs[f'{condition}_greedy'],same_prefix_divergence=None)
                if not same:
                    index=next((i for i,(a,b) in enumerate(zip(tokens,generated)) if a!=b),min(len(tokens),len(generated)))
                    if output is None:raise RuntimeError('Divergence on resumed verification requires saved logits; stopping, not inventing a diagnosis')
                    assert tokens[:index]==generated[:index]
                    replay,ml=manual(condition,prompt,capture_step=index);assert replay[:index]==generated[:index]
                    gl=output.logits[index][0].float().cpu().numpy()
                    check['same_prefix_divergence']=dict(step=index,shared_prefix_ids=row['input_ids']+generated[:index],manual_argmax=int(ml.argmax()),generate_raw_argmax=int(gl.argmax()),max_abs_logit_difference=float(np.max(np.abs(ml.astype(float)-gl.astype(float)))),manual_top2=np.argsort(-ml)[:2].tolist(),generate_top2=np.argsort(-gl)[:2].tolist())
                    atomic(vp,check);raise RuntimeError('Manual/generate token divergence documented on shared prefix; stop before comparison')
                atomic(vp,check);del output
                print('VERIFIED '+condition+' exact greedy token identity',flush=True)
            else:assert json.loads(vp.read_text())['token_ids_identical']
            for prompt in PROMPTS:
                row,output=generate(condition,prompt,'greedy',None);del output
                for seed in (0,1,2):
                    row,output=generate(condition,prompt,'sampling',seed);del output
    except Exception as error:
        atomic(OUT/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise
    finally:
        atomic(OUT/f'resources_{session}.json',dict(gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),attention=model.config._attn_implementation,**resources()))
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'conditions').glob('*.json'))];assert len(rows)==32
    with (OUT/'generations.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(OUT/'generations.csv',index=False)
    examples=['# All unsteered continuations','', 'Four fixed development prompts, not recovered original prompts. Continuation-only text; no steering.','']
    for condition in ('base','lora'):
        for prompt in PROMPTS:
            examples += [f'## {condition}: {prompt["prompt_id"]}', '',prompt['text'],'']
            for r in sorted([r for r in rows if r['model']==condition and r['prompt_id']==prompt['prompt_id']],key=lambda r:(r['mode'],r['seed'] or 0)):
                examples += [f'### {r["mode"]}, seed={r["seed"]}, stop={r["stopping_reason"]}, ID={r["condition_id"]}', '',r['continuation'],'']
    (OUT/'examples.md').write_text('\n'.join(examples));print('COMPLETE '+str(OUT),flush=True)

if __name__=='__main__':main()
