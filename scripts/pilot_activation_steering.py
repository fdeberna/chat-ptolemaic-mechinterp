"""Bounded 300-condition astronomy steering development pilot; no fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shutil
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ptolemaic_mechinterp.interventions.steering import residual_steering_hook

ROOT=Path(__file__).resolve().parents[1]
KS=(-.01,-.003,.003,.01)
TARGETS={'stance':16,'style':6}
PROMPTS=[
 ('crescent_dusk','lunar_appearance','At dusk, a narrow crescent appears near the western horizon. Over the following evenings,'),
 ('wanderer_notebook','apparent_planet_motion','An observer keeps a notebook of a bright wandering light against the fixed stars. After several weeks,'),
 ('voyage_pole','sky_geometry','During a long sea voyage, the height of the northern pole above the horizon changes. This observation'),
 ('dark_edge','eclipse_observation','A dark circular edge moves across the bright face of the Moon. Two observers in distant towns compare their notes and conclude that'),
 ('winter_hours','seasonal_sky','A village timekeeper finds that the interval from sunrise to sunset is shorter in winter. To account for this pattern,'),
 ('comet_visitors','comet_observation','A visitor asks why a comet can remain visible for many nights while a shooting star disappears almost at once. The astronomer replies that'),
]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True));tmp.replace(path)

def prepare(out):
    sources={};directions={};audits=[];scales={}
    def provenance(p):sources[str(p.relative_to(ROOT))]=sha(p)
    metadata_path=ROOT/'results/activations/pilot_full/base/metadata.csv';provenance(metadata_path)
    meta=pd.read_csv(metadata_path)
    ap=metadata_path.parent/'activations.npz';provenance(ap)
    with np.load(ap) as x:
        for target,layer in TARGETS.items():
            h=x['activations'][meta.layer_index.to_numpy()==layer].astype(np.float64)
            assert h.shape==(120,3584) and np.isfinite(h).all()
            scales[target]=float(np.median(np.linalg.norm(h,axis=1)))
    for target,layer in TARGETS.items():
        for model in ('base','lora'):
            p=ROOT/f'results/probes/pilot_full/{model}/{target}/full_data_coefficients/layer_{layer:03d}_full_data.npz';provenance(p)
            with np.load(p) as a:
                mapping=json.loads(str(a['label_mapping']));positive='heliocentric' if target=='stance' else 'premodern'
                assert mapping[positive]==1 and str(a['positive_class'])==positive
                scale=a['scaler_scale'];v=a['coef'].reshape(-1);w=v/scale
                assert w.shape==(3584,) and np.isfinite(w).all() and (scale>0).all() and np.linalg.norm(w)>1e-12
                raw=a['activation_space_direction'].reshape(-1)
                assert np.allclose(w,raw,rtol=1e-12,atol=1e-12)
                directions[target,model]=w/np.linalg.norm(w)
                audits.append(dict(target=target,layer=layer,source=model,positive_class=positive,raw_norm=float(np.linalg.norm(w)),raw_max_difference=float(np.max(np.abs(w-raw))),artifact=str(p.relative_to(ROOT)),activation_scale=scales[target]))
        # Different generating models have separate, reproducibly seeded controls.
        for model,offset in [('base',0),('lora',1)]:
            seed=20261026+layer*10+offset
            w=np.random.default_rng(seed).normal(size=3584);directions[target,'random_'+model]=w/np.linalg.norm(w)
            audits.append(dict(target=target,layer=layer,source='random_'+model,seed=seed,activation_scale=scales[target]))
    words=lambda s:re.findall(r'[a-z]+',s.lower())
    train=meta.loc[meta.layer_index==0,['prompt_id','prompt_text']]
    overlap=[]
    for pid,topic,text in PROMPTS:
        a=words(text);best=0;matched=None
        for item in train.itertuples(index=False):
            b=words(item.prompt_text);length=0
            for i in range(len(a)):
                for j in range(len(b)):
                    k=0
                    while i+k<len(a) and j+k<len(b) and a[i+k]==b[j+k]:k+=1
                    length=max(length,k)
            if length>best:best=length;matched=item.prompt_id
        overlap.append(dict(prompt_id=pid,topic=topic,family=pid,prompt_text=text,max_contiguous_shared_words=best,closest_training_prompt=matched))
    assert max(x['max_contiguous_shared_words'] for x in overlap)<8,overlap
    pd.DataFrame(overlap).to_csv(out/'development_prompts.csv',index=False)
    np.savez_compressed(out/'directions.npz',**{f'{a}__{b}':v for (a,b),v in directions.items()})
    atomic(out/'direction_audit.json',audits)
    return directions,scales,sources,overlap

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',default='results/steering/pilot_L16_stance_L6_style');parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    out=ROOT/args.output_root;out.mkdir(parents=True,exist_ok=True);(out/'conditions').mkdir(exist_ok=True)
    tick=time.perf_counter();session=str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2)
    directions,scales,sources,prompts=prepare(out)
    previous=json.loads((ROOT/'results/full_strength_nf4_fp16/activations/series_config.json').read_text());adapter=Path(previous['adapter_path'])
    snapshot=Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{previous['model_revision']}"
    assert sha(adapter/'adapter_model.safetensors')==previous['adapter_sha256']
    assert sha(adapter/'adapter_config.json')==previous['adapter_config_sha256']
    sources['scripts/pilot_activation_steering.py']=sha(Path(__file__));sources['src/ptolemaic_mechinterp/interventions/steering.py']=sha(ROOT/'src/ptolemaic_mechinterp/interventions/steering.py')
    config=dict(model='Qwen/Qwen2.5-7B',revision=previous['model_revision'],adapter=str(adapter),adapter_sha256=previous['adapter_sha256'],adapter_config_sha256=previous['adapter_config_sha256'],tokenizer_sha256=sha(adapter/'tokenizer.json'),sources=sources,scales=scales,targets=TARGETS,strengths=list(KS),random_seed_rule='20261026 + layer*10 + (0 base / 1 lora)',
        numerical=dict(dtype='float16',quantization='NF4',double_quant=True,compute_dtype='float16',attention='sdpa',device_map={'':0},fp32_jvp_wrapper=False),
        generation=dict(do_sample=False,max_new_tokens=96,use_cache=True,batch_size=1,template='plain continuation; add_special_tokens=True',eos_policy='stop at configured model eos IDs',logits_to_keep=1),
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')},prompts=prompts)
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['run_identifier']=identifier
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'Incompatible resume'
    atomic(out/'config.json',config)
    (out/'reconstruction').mkdir(exist_ok=True)
    shutil.copy2(__file__,out/'reconstruction'/Path(__file__).name);shutil.copy2(ROOT/'src/ptolemaic_mechinterp/interventions/steering.py',out/'reconstruction/steering.py')
    assert torch.cuda.is_available(),'CUDA unavailable; use approved GPU execution outside sandbox'
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    base_tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    assert all(tokenizer(x[2])['input_ids']==base_tokenizer(x[2])['input_ids'] for x in PROMPTS)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'},'Offload forbidden'
    core=model.get_base_model().model
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    def adapter_context(condition):return model.disable_adapter() if condition=='base' else nullcontext()
    def resources():return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.perf_counter()-tick)
    def encode(text):return {k:v.to('cuda:0') for k,v in tokenizer(text,return_tensors='pt',add_special_tokens=True).items()}
    @torch.inference_mode()
    def decode(encoded,limit=96):
        tokens=[];cache=None;ids=encoded['input_ids'];attention=encoded['attention_mask'];first=None
        for i in range(limit):
            result=model(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True,output_hidden_states=False,return_dict=True,logits_to_keep=1)
            logits=result.logits[0,-1,:]
            assert torch.isfinite(logits).all(),'Nonfinite logits'
            if first is None:first=logits.float().cpu().numpy()
            next_id=int(logits.argmax());tokens.append(next_id);cache=result.past_key_values
            del result
            if next_id in eos:break
            ids=torch.tensor([[next_id]],device='cuda:0');attention=torch.cat([attention,torch.ones((1,1),dtype=attention.dtype,device='cuda:0')],dim=1)
        del cache
        return dict(token_ids=tokens,continuation=tokenizer.decode(tokens,skip_special_tokens=True),generated_tokens=len(tokens)),first
    checks=[]
    def hook_checks(condition):
        encoded=encode(PROMPTS[0][2]);last=int(encoded['attention_mask'].sum())-1
        with adapter_context(condition),torch.inference_mode():
            reference=model(**encoded,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
            hidden=reference.hidden_states
            for target,layer in TARGETS.items():
                module=core.layers[layer-1];captured=[]
                handle=module.register_forward_hook(lambda m,i,o:captured.append((o[0] if isinstance(o,tuple) else o).detach().clone()))
                try:model(**encoded,use_cache=False,output_hidden_states=False,return_dict=True,logits_to_keep=1)
                finally:handle.remove()
                location_diff=float((captured[0]-hidden[layer]).abs().max());assert location_diff==0
                u=torch.tensor(directions[target,'base'],device='cuda:0');gamma=.01*scales[target]
                probe_path=ROOT/f'results/probes/pilot_full/base/{target}/full_data_coefficients/layer_{layer:03d}_full_data.npz'
                with np.load(probe_path) as a:w=torch.tensor((a['coef']/a['scaler_scale']).reshape(-1),device='cuda:0')
                observed={}
                def observer(before,after,index,delta):
                    diff=after.double()-before.double();other=diff.clone();other[0,index,:]=0
                    actual=diff[0,index,:];wanted=u.double()*gamma
                    absolute=float((actual-wanted).abs().max())
                    rounding_bound=float((torch.finfo(before.dtype).eps*(before[0,index,:].abs().float()+delta.abs())+torch.finfo(before.dtype).tiny).max())
                    assert torch.count_nonzero(other)==0 and absolute<=rounding_bound
                    score=float(w@actual);expected=float(gamma*w.norm());score_error=abs(score-expected)
                    score_bound=float(w.norm()*(actual-wanted).norm())+1e-10
                    assert score_error<=score_bound
                    observed.update(actual_delta_max_error=absolute,rounding_bound=rounding_bound,delta_relative_error=float((actual-wanted).norm()/wanted.norm()),raw_probe_score_change=score,expected_raw_probe_score_change=expected,score_absolute_error=score_error,score_error_bound=score_bound,unintended_positions_nonzero=int(torch.count_nonzero(other)))
                before_hooks=len(module._forward_hooks)
                with residual_steering_hook(module,u,gamma,last,observer) as state:
                    model(**encoded,use_cache=False,output_hidden_states=False,return_dict=True,logits_to_keep=1)
                assert len(module._forward_hooks)==before_hooks
                try:
                    with residual_steering_hook(module,u,gamma,last):raise RuntimeError('deliberate hook cleanup check')
                except RuntimeError:pass
                assert len(module._forward_hooks)==before_hooks
                after=model(**encoded,use_cache=False,output_hidden_states=False,return_dict=True,logits_to_keep=1)
                cleanup=float((after.logits-reference.logits).abs().max());assert cleanup==0
                checks.append(dict(model=condition,target=target,layer=layer,module=f'model.layers.{layer-1}',hook_vs_saved_hidden_location_max_difference=location_diff,cleanup_logits_max_difference=cleanup,exception_cleanup=True,**observed))
            del reference,hidden
        atomic(out/'numerical_checks.json',checks)
    def condition_id(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]
    def run(condition,prompt,target=None,source=None,k=0.,limit=96,save=True):
        pid,topic,text=prompt;layer=TARGETS.get(target);gamma=k*scales[target] if target else 0.
        identity=dict(run_identifier=identifier,model=condition,prompt_id=pid,target=target,direction_source=source,k=k)
        cid=condition_id(identity);path=out/'conditions'/f'{cid}.json'
        if save and path.exists():
            record=json.loads(path.read_text());assert record['run_identifier']==identifier;return record,None
        encoded=encode(text);last=int(encoded['attention_mask'].sum())-1
        state=None;context=nullcontext()
        if target:
            direction_key='random_'+condition if source=='random' else source
            u=torch.tensor(directions[target,direction_key],device='cuda:0')
            context=residual_steering_hook(core.layers[layer-1],u,gamma,last)
        torch.cuda.synchronize();start=time.perf_counter()
        with adapter_context(condition),context as state:
            result,logits=decode(encoded,limit)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if target:
            assert state['prefill_calls']==1 and state['decode_calls']==result['generated_tokens']-1
        record=dict(**identity,condition_id=cid,prompt_text=text,topic=topic,family=pid,layer=layer,gamma=gamma,activation_scale=scales.get(target),
            adapter_active=condition=='lora',adapter_strength=1 if condition=='lora' else 0,random_seed=(20261026+layer*10+(condition=='lora')) if source=='random' else None,
            generation_config=config['generation'],prompt_tokens=int(encoded['input_ids'].shape[1]),runtime_seconds=elapsed,hook=state,**result,resources=resources())
        if state:
            record['actual_perturbation_ratio_min']=min(state['ratios']);record['actual_perturbation_ratio_median']=float(np.median(state['ratios']));record['actual_perturbation_ratio_max']=max(state['ratios'])
        if save:atomic(path,record)
        return record,logits
    try:
        # Exactly two complete smoke decodes: baseline plus zero-hook check.
        # The baseline is reused in the 300-condition grid; the zero-hook run is
        # validation only, not an extra experimental condition.
        if not (out/'smoke.json').exists():
            baseline,bl=run('base',PROMPTS[0])
            if bl is None:
                with adapter_context('base'),torch.inference_mode():
                    bl=model(**encode(PROMPTS[0][2]),use_cache=False,output_hidden_states=False,logits_to_keep=1).logits[0,-1].float().cpu().numpy()
            zero,zl=run('base',PROMPTS[0],target='stance',source='base',k=0.,save=False)
            difference=float(np.max(np.abs(bl-zl)));assert difference<=1e-6 and baseline['token_ids']==zero['token_ids']
            smoke=dict(no_hook_vs_zero_max_logits_difference=difference,tolerance=1e-6,identical_greedy_tokens=True,baseline_seconds=baseline['runtime_seconds'],zero_hook_seconds=zero['runtime_seconds'],tokens=[baseline['generated_tokens'],zero['generated_tokens']],estimated_300_generation_minutes=300*np.mean([baseline['runtime_seconds'],zero['runtime_seconds']])/60,resources=resources(),gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),attention=model.config._attn_implementation,sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled(),memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled()))
            atomic(out/'smoke.json',smoke);print('SMOKE '+json.dumps(smoke),flush=True)
        if args.smoke_only:return
        for condition in ('base','lora'):
            hook_checks(condition)
            for prompt in PROMPTS:run(condition,prompt)
            for target in TARGETS:
                for source in ('base','lora','random'):
                    for k in KS:
                        for prompt in PROMPTS:run(condition,prompt,target,source,k)
                        completed=len(list((out/'conditions').glob('*.json')))
                        atomic(out/'progress.json',dict(completed=completed,expected=300,session=session,resources=resources()))
                        print(f'PROGRESS {completed}/300 {condition} {target} {source} k={k} elapsed_min={(time.perf_counter()-tick)/60:.2f}',flush=True)
        assert len(list((out/'conditions').glob('*.json')))==300
    except Exception as error:
        atomic(out/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()))
        raise
    finally:
        atomic(out/f'resources_{session}.json',resources())
    records=[json.loads(p.read_text()) for p in sorted((out/'conditions').glob('*.json'))]
    with (out/'generations.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(out/'generations.csv',index=False)
    # Stable randomized blinding with no condition columns in the review table.
    order=np.random.default_rng(20261026).permutation(len(records));review=[];key=[]
    for j,i in enumerate(order):
        record=records[i];rid=f'R{j+1:03d}'
        review.append(dict(review_id=rid,continuation=record['continuation'],generated_stance='',generated_style='',coherence='',relevance='',repetition='',notes=''))
        key.append(dict(review_id=rid,condition_id=record['condition_id'],model=record['model'],target=record['target'],layer=record['layer'],direction_source=record['direction_source'],k=record['k'],gamma=record['gamma'],prompt_id=record['prompt_id'],prompt_text=record['prompt_text']))
    pd.DataFrame(review).to_csv(out/'blinded_review.csv',index=False);pd.DataFrame(key).to_csv(out/'condition_key.csv',index=False)
    (out/'review_instructions.md').write_text('Score only the continuation. Stance: geocentric / heliocentric / mixed / unclear. Style: modern / premodern / mixed / unclear. Leave unclear when evidence is absent. Coherence, relevance, repetition: record descriptive notes; relevance may require the prompt from the separately held condition key. No suitable pre-existing independent text-labeling rubric/evaluator was found. These are pilot review instructions, not validated automatic labels. Condition key should be held separately by the review organizer. No probe scores supply behavioral labels.\n')
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
