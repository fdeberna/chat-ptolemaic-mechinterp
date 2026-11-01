"""Small stance experiment consuming the frozen reproduced generation preset."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import copy
import argparse
from contextlib import nullcontext
import hashlib,importlib.metadata,json,random,resource,shutil,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig
from ptolemaic_mechinterp.interventions.steering import residual_steering_hook
from run_sampled_steering import RUBRIC

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'results/steering/reproduced_protocol_stance_development';OUT=ROOT/'results/steering/reproduced_protocol_stance_strength_sweep';PILOT=ROOT/'results/steering/pilot_L16_stance_L6_style'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True));t.replace(p)

def main():
 args=argparse.ArgumentParser();args.add_argument('--stage',choices=['low','high'],required=True);stage=args.parse_args().stage
 OUT.mkdir(parents=True,exist_ok=True)
 assert os.environ.get('CUDA_LAUNCH_BLOCKING')=='1'
 for name in ['conditions','verification','reconstruction']:(OUT/name).mkdir(exist_ok=True)
 preset_path=ROOT/'configs/generation_reproduced_v1.json';preset=json.loads(preset_path.read_text());selection=json.loads((SOURCE/'development_selection.json').read_text());old=json.loads((PILOT/'config.json').read_text())
 assert {p:importlib.metadata.version(p) for p in preset['versions']}==preset['versions']
 sources={str(p):sha(p) for p in [Path(__file__),preset_path,SOURCE/'development_selection.json',ROOT/'src/ptolemaic_mechinterp/interventions/steering.py',PILOT/'directions.npz']}
 historical={}
 for path,digest in selection['source_sha256'].items():
  assert sha(path)==digest
  model='base' if Path(path).name=='qwen25_base_eval.jsonl' else 'lora' if Path(path).name=='qwen25_lora1000_eval.jsonl' else None
  if model:
   for r in [json.loads(x) for x in Path(path).read_text().splitlines()]:
    if r['prompt_id'] in selection['selected_prompt_ids'] and r['sample_id'] in selection['steering_sample_ids']:historical[model,r['prompt_id'],r['sample_id']]=r
 assert len(historical)==12
 for (m,p,s),r in historical.items():
  seed=int.from_bytes(hashlib.sha256(f'{r["seed"]}|{r["prompt_file"]}|{p}|{s}'.encode()).digest()[:8],'big')%(2**31);assert seed==r['sample_seed']
  for field in ['max_new_tokens','temperature','top_p','do_sample','repetition_penalty','no_repeat_ngram_size']:assert r[field]==preset['generation_config'][field]
  assert r['prompt']==historical['base',p,s]['prompt']
 if (OUT/'rubric.md').exists():assert (OUT/'rubric.md').read_text()==RUBRIC
 else:(OUT/'rubric.md').write_text(RUBRIC)
 config=dict(preset_path=str(preset_path),preset_sha256=sha(preset_path),selection=selection,selected_prompt_ids=selection['selected_prompt_ids'],sample_ids=[0,1,2],models=['base','lora'],target='stance',saved_layer=16,module='model.layers.15 residual output',direction_sources=['own','random'],strengths=[-.5,-.3,-.1,0.,.1,.3,.5],reused_source=str(SOURCE),activation_scale=old['scales']['stance'],token_policy='final non-padding prompt token, then every cached newly processed token',sources=sources,rubric_sha256=sha(OUT/'rubric.md'),no_independent_evaluator=True)
 config['run_identifier']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();identifier=config['run_identifier']
 if (OUT/'experiment_config.json').exists():assert json.loads((OUT/'experiment_config.json').read_text())==config,'Incompatible resume'
 atomic(OUT/'experiment_config.json',config);shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name);shutil.copy2(preset_path,OUT/'reconstruction'/preset_path.name)
 assert sha(ROOT/'src/ptolemaic_mechinterp/interventions/steering.py')==old['sources']['src/ptolemaic_mechinterp/interventions/steering.py']
 reused=[json.loads(p.read_text()) for p in sorted((SOURCE/'conditions').glob('*.json'))]
 assert len(reused)==60
 atomic(OUT/'reused_records.json',reused)
 models=['base','lora']
 if stage=='high':
  gate=json.loads((OUT/'quality_gate.json').read_text());assert gate['reviewed_low_count']==48
  assert gate['low_condition_sha256']=={p.name:sha(p) for p in sorted((OUT/'conditions').glob('*.json')) if abs(json.loads(p.read_text())['k'])==.3}
  models=[m for m in models if gate['models'][m]['proceed']]
  if not models:print('No model passed escalation gate');return
 tick=time.perf_counter();session=stage+'_'+str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2);assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
 adapter=Path(preset['adapter']);snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{preset["revision"]}'
 for p in [adapter/'adapter_model.safetensors',adapter/'adapter_config.json',adapter/'tokenizer.json']:assert sha(p)==preset['source_hashes'][str(p)]
 tokenizers={}
 for m,path in [('base',snapshot),('lora',adapter)]:
  tok=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True)
  if tok.pad_token is None:tok.pad_token=tok.eos_token
  tokenizers[m]=tok;assert tok.eos_token_id==tok.pad_token_id==151643
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
 model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
 core=model.get_base_model().model;vectors={}
 with np.load(PILOT/'directions.npz') as a:
  for m in ['base','lora']:
   vectors[m,'own']=a[f'stance__{m}'].copy();vectors[m,'random']=a[f'stance__random_{m}'].copy()
   path=ROOT/f'results/probes/pilot_full/{m}/stance/full_data_coefficients/layer_016_full_data.npz'
   with np.load(path) as probe:
    w=(probe['coef']/probe['scaler_scale']).reshape(-1);assert np.array_equal(w/np.linalg.norm(w),vectors[m,'own']) and str(probe['positive_class'])=='heliocentric'
   for source in ['own','random']:assert np.isfinite(vectors[m,source]).all() and abs(np.linalg.norm(vectors[m,source])-1)<1e-12
 gc=GenerationConfig.from_dict(copy.deepcopy(preset['generation_config']));assert gc.num_return_sequences==1 and gc.top_k==50 and gc.max_new_tokens==150
 atomic(OUT/'effective_generation_config.json',gc.to_dict())
 def resources():return dict(elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 def no_hooks():assert all(not x._forward_hooks for x in core.layers) and not core.norm._forward_hooks
 @torch.inference_mode()
 def run(m,pid,s,source=None,k=0.,verification=False):
  identity=dict(run_identifier=identifier,model=m,prompt_id=pid,sample_id=s,direction_source=source,k=k);condition_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
  path=OUT/('verification' if verification else 'conditions')/f'{condition_id}.json'
  if path.exists():return json.loads(path.read_text())
  baseline=next(x for x in reused if x['model']==m and x['prompt_id']==pid and x['sample_id']==s and x['kind']=='baseline')
  r=historical[m,pid,s];tok=tokenizers[m];encoded={k:v.to('cuda:0') for k,v in tok(r['prompt'],return_tensors='pt').items()};assert encoded['input_ids'][0].tolist()==baseline['input_ids'];no_hooks();state=None
  ctx=nullcontext() if source is None else residual_steering_hook(core.layers[15],torch.tensor(vectors[m,source],device='cuda:0'),k*config['activation_scale'],int(encoded['attention_mask'].sum())-1)
  torch.cuda.synchronize();start=time.perf_counter()
  with (model.disable_adapter() if m=='base' else nullcontext()),ctx as state:
   random.seed(r['sample_seed']);np.random.seed(r['sample_seed']);torch.manual_seed(r['sample_seed']);torch.cuda.manual_seed_all(r['sample_seed'])
   output=model.generate(**encoded,generation_config=copy.deepcopy(gc),return_dict_in_generate=True,output_logits=False,output_scores=False,logits_to_keep=1)
  torch.cuda.synchronize();seconds=time.perf_counter()-start;no_hooks();tokens=output.sequences[0,encoded['input_ids'].shape[1]:].tolist();del output;text=tok.decode(tokens,skip_special_tokens=True)
  if state:assert state['prefill_calls']==1 and state['calls']==len(tokens) and state['decode_calls']==len(tokens)-1
  result=dict(**identity,condition_id=condition_id,kind='zero_hook_verification' if verification else 'baseline' if source is None else 'steered',prompt_text=r['prompt'],sample_seed=r['sample_seed'],historical_record=r,input_ids=encoded['input_ids'][0].tolist(),generated_token_ids=tokens,continuation=text,generated_tokens=len(tokens),stopping_reason='eos' if tokens and tokens[-1]==tok.eos_token_id else 'max_new_tokens',runtime_seconds=seconds,adapter_active=m=='lora',layer=16 if source else None,gamma=k*config['activation_scale'],hook_state=state,generation_config=gc.to_dict(),historical_exact_match=text==r['completion'],baseline_condition_id=baseline['condition_id'],baseline_source=str(SOURCE),stage=stage,random_control_seed=(20261026+160+(m=='lora')) if source=='random' else None,resources=resources())
  atomic(path,result);print(f'COMPLETED {len(list((OUT/"conditions").glob("*.json")))}/96 {m} {pid} sample{s} {source} k={k} historical_exact={result["historical_exact_match"]} seconds={seconds:.1f}',flush=True);return result
 try:
  for m in models:
   for pid in selection['selected_prompt_ids']:
    for source in ['own','random']:
     for k in ([-.3,.3] if stage=='low' else [-.5,.5]):
      for s in [0,1,2]:run(m,pid,s,source,k)
 except Exception as error:
  atomic(OUT/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise
 finally:atomic(OUT/f'resources_{session}.json',dict(gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),**resources()))
 print('GENERATION COMPLETE',flush=True)
if __name__=='__main__':main()
