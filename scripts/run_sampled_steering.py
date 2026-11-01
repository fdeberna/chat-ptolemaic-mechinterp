"""One bounded sampled-steering grid, with initial matched-baseline quality gate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import copy
from contextlib import ExitStack,nullcontext
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
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from ptolemaic_mechinterp.interventions.steering import residual_steering_hook

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steering/sampled_own_random_k01'
BASELINES=ROOT/'results/generation_diagnostics/recovered_defaults_comparison'
PILOT=ROOT/'results/steering/pilot_L16_stance_L6_style'
TARGETS={'stance':16,'style':6}
RUBRIC='''# Frozen provisional scoring rubric

This rubric is frozen before any generation in this experiment. Score BOTH stance
and style for every continuation, including all baselines, unclear and poor cases.

Semantic window: first 80 generated words, or the entire continuation if shorter.
A word is a non-whitespace span; preserve the exact substring ending at word 80.
Prompt text is context only, never evidence for generated stance/style.

Stance: first substantive astronomical answer in the window. Options:
geocentric / heliocentric / mixed / unclear.
Separately record Earth rotation and Earth revolution around the Sun, each as:
affirmed / denied / conflicting / unmentioned.
Earth rotation alone does NOT establish heliocentrism.

Style: modern / premodern / mixed / unclear, from explanatory language/frame in
the window. A factual error alone does NOT establish premodern style.
Unselected quiz options and historical quotations are NOT endorsements. Use an
explicit selected answer only if available inside the window; otherwise unclear.
Provide supporting excerpts from the scored window and explanatory notes.

Separately review the FULL continuation for repetition, incoherence, topic drift,
and truncation. Do not infer incoherence solely from an old astronomical model.
Keep factual errors distinct from incoherence. Retain all outputs, including
unclear/poor cases. These are provisional development judgments, not validated
labels. Three seeds are not three independent prompts.

Blinding: present only rubric, prompt context, opaque IDs, scored window and full
continuation to a genuinely separate evaluator. Keep condition key separate.
The generating session is NOT blinded by removing columns from its files.
No separate evaluator is used in this run: semantic/quality review fields remain
blank for later review. Initial safety/quality gating is an unblinded development
check only and must not be represented as semantic evaluation.
'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True));tmp.replace(path)

def cid(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('initial','full'),required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_LAUNCH_BLOCKING')=='1','Preserve successful baseline execution setting CUDA_LAUNCH_BLOCKING=1'
    for name in ('conditions','reused_baselines','verification','reconstruction'):(OUT/name).mkdir(parents=True,exist_ok=True)
    baseline_config=json.loads((BASELINES/'config.json').read_text());pilot=json.loads((PILOT/'config.json').read_text())
    versions={p:importlib.metadata.version(p) for p in baseline_config['versions']};assert versions==baseline_config['versions']
    hook_path=ROOT/'src/ptolemaic_mechinterp/interventions/steering.py'
    assert sha(hook_path)==pilot['sources']['src/ptolemaic_mechinterp/interventions/steering.py']
    if (OUT/'rubric.md').exists():assert (OUT/'rubric.md').read_text()==RUBRIC
    else:(OUT/'rubric.md').write_text(RUBRIC)
    config=dict(model=baseline_config['model'],revision=baseline_config['revision'],adapter=baseline_config['adapter'],numerical=baseline_config['numerical'],prompts=baseline_config['prompts'],
        baseline_run_identifier=baseline_config['run_identifier'],targets=TARGETS,scales=pilot['scales'],direction_sources=['own','random'],strengths=[-.1,0.,.1],seeds=[0,1,2],
        generation=dict(do_sample=True,temperature=.8,top_p=.9,top_k=50,max_new_tokens=200,repetition_penalty=1.,no_repeat_ngram_size=0,use_cache=True,num_beams=1),
        initial_policy='First fixed prompt daily_sky and seed0, both models, targets, own/random directions and both signs; all 16 count toward grid. Stop model/target family for severe NEW quality deterioration relative to corresponding reused baseline, never merely baseline defects.',
        execution=dict(CUDA_LAUNCH_BLOCKING=1,batch_size=1,inference_mode=True,no_cpu_offload=True,one_model_instance=True),versions=versions,
        sources={str(p):sha(p) for p in (Path(__file__),hook_path,OUT/'rubric.md',BASELINES/'config.json',BASELINES/'effective_generation_configs.json',PILOT/'config.json',PILOT/'directions.npz')})
    config['run_identifier']=cid(config);identifier=config['run_identifier']
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config,'Incompatible resumed configuration'
    else:atomic(OUT/'config.json',config)
    shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name);shutil.copy2(hook_path,OUT/'reconstruction/steering.py')
    pd.DataFrame(config['prompts']).to_csv(OUT/'prompts.csv',index=False)
    baselines={};source_files={}
    for p in (BASELINES/'conditions').glob('*.json'):
        row=json.loads(p.read_text())
        if row['mode']!='sampling':continue
        assert row['run_identifier']==baseline_config['run_identifier']
        key=(row['model'],row['prompt_id'],row['seed']);assert key not in baselines
        assert row['prompt_text']==next(x['text'] for x in config['prompts'] if x['prompt_id']==row['prompt_id'])
        for k,v in config['generation'].items():assert row['generation_config'][k]==v,(key,k)
        baselines[key]=row;source_files[key]=dict(path=str(p),sha256=sha(p))
    assert len(baselines)==24
    atomic(OUT/'baseline_source_index.json',{f'{m}/{p}/{s}':v for (m,p,s),v in source_files.items()})
    for key,row in baselines.items():
        data=dict(run_identifier=identifier,kind='baseline',model=key[0],prompt_id=key[1],seed=key[2]);new_id=cid(data)
        copy_row=dict(row,**data,condition_id=new_id,original_condition_id=row['condition_id'],original_run_identifier=row['run_identifier'],source=source_files[key],target=None,direction_source=None,k=0.,gamma=0.,runtime_seconds_new=0.)
        atomic(OUT/'reused_baselines'/f'{new_id}.json',copy_row)
    gate=None
    if args.stage=='full':
        gate=json.loads((OUT/'initial_quality_gate.json').read_text());assert gate['run_identifier']==identifier and gate['all_initial_conditions_reviewed']
        assert len(list((OUT/'conditions').glob('*.json')))>=16
    tick=time.perf_counter();session=str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert torch.cuda.is_available(),'Approved GPU access required';torch.cuda.reset_peak_memory_stats()
    adapter=Path(config['adapter']);snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{config["revision"]}'
    for p,h in baseline_config['sources'].items():assert sha(p)==h,p
    tokenizers={}
    for model_name,source in [('base',snapshot),('lora',adapter)]:
        tok=AutoTokenizer.from_pretrained(source,use_fast=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tokenizers[model_name]=tok
        for prompt in config['prompts']:
            for seed in (0,1,2):assert tok(prompt['text'])['input_ids']==baselines[model_name,prompt['prompt_id'],seed]['input_ids']
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    core=model.get_base_model().model;directions={};direction_audit=[]
    with np.load(PILOT/'directions.npz') as saved:
        for model_name in ('base','lora'):
            for target,layer in TARGETS.items():
                path=ROOT/f'results/probes/pilot_full/{model_name}/{target}/full_data_coefficients/layer_{layer:03d}_full_data.npz'
                with np.load(path) as a:
                    w=(a['coef']/a['scaler_scale']).reshape(-1);u=w/np.linalg.norm(w)
                    assert w.shape==(3584,) and np.isfinite(w).all() and np.linalg.norm(w)>1e-12
                    assert np.array_equal(w,a['activation_space_direction'].reshape(-1))
                    assert np.array_equal(u,saved[f'{target}__{model_name}'])
                    assert json.loads(str(a['label_mapping']))==({'geocentric':0,'heliocentric':1} if target=='stance' else {'modern':0,'premodern':1})
                directions[model_name,target,'own']=u
                directions[model_name,target,'random']=saved[f'{target}__random_{model_name}'].copy()
                seed=20261026+layer*10+(model_name=='lora');v=np.random.default_rng(seed).normal(size=3584);v/=np.linalg.norm(v)
                assert np.array_equal(v,directions[model_name,target,'random'])
                direction_audit.append(dict(model=model_name,target=target,layer=layer,probe_path=str(path),probe_sha256=sha(path),raw_norm=float(np.linalg.norm(w)),random_seed=seed,activation_scale=config['scales'][target]))
    atomic(OUT/'direction_audit.json',direction_audit)
    def resource_info():return dict(elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def no_hooks():assert all(not m._forward_hooks for m in core.layers) and not core.norm._forward_hooks
    def context(m):return model.disable_adapter() if m=='base' else nullcontext()
    def generation_config(m):
        gc=copy.deepcopy(model.generation_config);tok=tokenizers[m]
        unused=gc.update(**config['generation'],pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id);assert not unused
        reference=baselines[m,config['prompts'][0]['prompt_id'],0]['generation_config']
        assert gc.to_dict()==reference,'Effective generation configuration differs from reused baseline'
        return gc
    atomic(OUT/'effective_generation_configs.json',{m:generation_config(m).to_dict() for m in ('base','lora')})
    @torch.inference_mode()
    def generate(m,prompt,seed,target=None,source=None,k=0.,zero_both=False):
        no_hooks();tok=tokenizers[m];gc=generation_config(m);encoded={k:v.to('cuda:0') for k,v in tok(prompt['text'],return_tensors='pt').items()}
        states={};gamma=k*config['scales'][target] if target else 0.;torch.cuda.synchronize();start=time.perf_counter()
        with context(m),ExitStack() as stack:
            if zero_both:
                for t,l in TARGETS.items():states[t]=stack.enter_context(residual_steering_hook(core.layers[l-1],torch.tensor(directions[m,t,'own'],device='cuda:0'),0.,len(baselines[m,prompt['prompt_id'],seed]['input_ids'])-1))
            elif target:
                states[target]=stack.enter_context(residual_steering_hook(core.layers[TARGETS[target]-1],torch.tensor(directions[m,target,source],device='cuda:0'),gamma,len(baselines[m,prompt['prompt_id'],seed]['input_ids'])-1))
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            output=model.generate(**encoded,generation_config=gc,return_dict_in_generate=True,output_logits=False,output_scores=False,logits_to_keep=1)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;no_hooks()
        n=encoded['input_ids'].shape[1];tokens=output.sequences[0,n:].tolist();del output
        for t,state in states.items():
            assert state['calls']==len(tokens) and state['prefill_calls']==1 and state['decode_calls']==len(tokens)-1
            assert np.isfinite(state['ratios']).all()
        return dict(generated_token_ids=tokens,continuation=tok.decode(tokens,skip_special_tokens=True),generated_tokens=len(tokens),stopping_reason='eos' if tokens and tokens[-1]==tok.eos_token_id else 'max_new_tokens',runtime_seconds_new=elapsed,hook_states=states,gamma=gamma,input_ids=encoded['input_ids'][0].tolist(),generation_config=gc.to_dict(),generation_call_overrides=dict(return_dict_in_generate=True,output_logits=False,output_scores=False,logits_to_keep=1),resources=resource_info())
    def run(m,prompt,seed,target,source,k):
        data=dict(run_identifier=identifier,kind='steered',model=m,prompt_id=prompt['prompt_id'],seed=seed,target=target,direction_source=source,k=k);condition_id=cid(data);path=OUT/'conditions'/f'{condition_id}.json'
        if path.exists():assert json.loads(path.read_text())['run_identifier']==identifier;return
        result=generate(m,prompt,seed,target,source,k);baseline=baselines[m,prompt['prompt_id'],seed]
        record=dict(**data,condition_id=condition_id,prompt_text=prompt['text'],topic=prompt['topic'],layer=TARGETS[target],activation_scale=config['scales'][target],random_seed=(20261026+TARGETS[target]*10+(m=='lora')) if source=='random' else None,adapter_active=m=='lora',adapter_strength=1 if m=='lora' else 0,
            baseline_original_condition_id=baseline['condition_id'],baseline_source=source_files[m,prompt['prompt_id'],seed],identical_tokens_to_baseline=result['generated_token_ids']==baseline['generated_token_ids'],quality_gate_sha256=sha(OUT/'initial_quality_gate.json') if gate else None,**result)
        atomic(path,record);print(f'COMPLETED {len(list((OUT/"conditions").glob("*.json")))}/192 {m} {target} {source} k={k} {prompt["prompt_id"]} seed={seed} seconds={result["runtime_seconds_new"]:.1f}',flush=True)
    try:
        for m in ('base','lora'):
            vp=OUT/'verification'/f'{m}.json';prompt=config['prompts'][0]
            if not vp.exists():
                fresh=generate(m,prompt,0);zero=generate(m,prompt,0,zero_both=True);baseline=baselines[m,prompt['prompt_id'],0]
                check=dict(model=m,prompt_id=prompt['prompt_id'],seed=0,fresh_no_hook=fresh,zero_hooks=zero,
                    no_hook_matches_reused_baseline=fresh['generated_token_ids']==baseline['generated_token_ids'],zero_matches_no_hook=zero['generated_token_ids']==fresh['generated_token_ids'],baseline_original_condition_id=baseline['condition_id'])
                atomic(vp,check);assert check['no_hook_matches_reused_baseline'] and check['zero_matches_no_hook'],'Baseline replay / zero-hook mismatch; stop'
                print('VERIFIED baseline replay and zero hooks '+m,flush=True)
            else:
                check=json.loads(vp.read_text());assert check['no_hook_matches_reused_baseline'] and check['zero_matches_no_hook']
            for target in TARGETS:
                if gate and not next(g for g in gate['groups'] if g['model']==m and g['target']==target)['continue_grid']:continue
                for source in ('own','random'):
                    for k in (-.1,.1):
                        for prompt in (config['prompts'][:1] if args.stage=='initial' else config['prompts']):
                            for seed in ((0,) if args.stage=='initial' else (0,1,2)):run(m,prompt,seed,target,source,k)
    except Exception as error:
        atomic(OUT/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resource_info()));raise
    finally:
        atomic(OUT/f'resources_{args.stage}_{session}.json',dict(stage=args.stage,gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),attention=model.config._attn_implementation,**resource_info()))
    rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')]
    if args.stage=='initial':
        assert len(rows)==16
        lines=['# All initial quality-check examples','', 'First fixed prompt daily_sky, seed0. Unblinded development quality check only. Retain all outputs; compare with matching baseline, not an ideal answer.','']
        for m in ('base','lora'):
            b=baselines[m,config['prompts'][0]['prompt_id'],0];lines += [f'## {m}: matching unsteered baseline', '',b['continuation'],'']
            for r in sorted([r for r in rows if r['model']==m],key=lambda r:(r['target'],r['direction_source'],r['k'])):lines += [f'### {r["target"]} {r["direction_source"]} k={r["k"]}, ID {r["condition_id"]}', '',r['continuation'],'']
        (OUT/'initial_examples.md').write_text('\n'.join(lines))
    print('STAGE COMPLETE '+args.stage,flush=True)

if __name__=='__main__':main()
