"""Small stance experiment consuming the frozen reproduced generation preset."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import copy
from contextlib import nullcontext
import hashlib,importlib.metadata,json,random,resource,shutil,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig
from ptolemaic_mechinterp.interventions.lora_ablation import ablate_lora_blocks, adapter_state, validate_ablation_inventory
import csv
from run_sampled_steering import RUBRIC

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/adapter_ablation/stage3a';PILOT=ROOT/'results/steering/pilot_L16_stance_L6_style'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True));t.replace(p)

def main():
 assert os.environ.get('CUDA_LAUNCH_BLOCKING')=='1'
 for name in ['conditions','verification','reconstruction']:(OUT/name).mkdir(exist_ok=True)
 preset_path=ROOT/'configs/generation_reproduced_v1.json';preset=json.loads(preset_path.read_text());selection=json.loads((OUT/'development_selection.json').read_text());old=json.loads((PILOT/'config.json').read_text())
 assert {p:importlib.metadata.version(p) for p in preset['versions']}==preset['versions']
 sources={str(p):sha(p) for p in [Path(__file__),preset_path,OUT/'development_selection.json',ROOT/'src/ptolemaic_mechinterp/interventions/lora_ablation.py',ROOT/'results/adapter_ablation/step1_inspection/module_inventory.csv']}
 historical={}
 for path,digest in selection['source_sha256'].items():
  assert sha(path)==digest
  model='base' if Path(path).name=='qwen25_base_eval.jsonl' else 'lora' if Path(path).name=='qwen25_lora1000_eval.jsonl' else None
  if model:
   for r in [json.loads(x) for x in Path(path).read_text().splitlines()]:
    if r['prompt_id'] in selection['selected_prompt_ids'] and r['sample_id'] in selection['steering_sample_ids']:historical[model,r['prompt_id'],r['sample_id']]=r
 assert len(historical)==24
 for (m,p,s),r in historical.items():
  seed=int.from_bytes(hashlib.sha256(f'{r["seed"]}|{r["prompt_file"]}|{p}|{s}'.encode()).digest()[:8],'big')%(2**31);assert seed==r['sample_seed']
  for field in ['max_new_tokens','temperature','top_p','do_sample','repetition_penalty','no_repeat_ngram_size']:assert r[field]==preset['generation_config'][field]
  assert r['prompt']==historical['base',p,s]['prompt']
 if (OUT/'rubric.md').exists():assert (OUT/'rubric.md').read_text()==RUBRIC
 else:(OUT/'rubric.md').write_text(RUBRIC)
 config=dict(preset_path=str(preset_path),preset_sha256=sha(preset_path),selection=selection,conditions={'base':list(range(1,29)),'intact_lora':[],'ablate_L6':[6],'ablate_L16':[16],'ablate_L28':[28]},token_policy='all prompt and cached tokens; fresh KV cache',sources=sources,rubric_sha256=sha(OUT/'rubric.md'),no_independent_evaluator=True)
 config['run_identifier']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();identifier=config['run_identifier']
 if (OUT/'experiment_config.json').exists():assert json.loads((OUT/'experiment_config.json').read_text())==config,'Incompatible resume'
 atomic(OUT/'experiment_config.json',config);shutil.copy2(__file__,OUT/'reconstruction'/Path(__file__).name);shutil.copy2(preset_path,OUT/'reconstruction'/preset_path.name)
 tick=time.perf_counter();session=str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2);assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
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
 core=model.get_base_model().model
 inventory=list(csv.DictReader((ROOT/'results/adapter_ablation/step1_inspection/module_inventory.csv').open()))
 modules=validate_ablation_inventory(model,inventory);initial=adapter_state(model,modules)
 reuse=[json.loads(x) for x in (ROOT/'results/steering/reproduced_protocol_stance_development/generations.jsonl').read_text().splitlines()]
 gc=GenerationConfig.from_dict(copy.deepcopy(preset['generation_config']));assert gc.num_return_sequences==1 and gc.top_k==50 and gc.max_new_tokens==150
 atomic(OUT/'effective_generation_config.json',gc.to_dict())
 def resources():return dict(elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 def no_hooks():assert all(not x._forward_hooks for x in core.layers) and not core.norm._forward_hooks
 @torch.inference_mode()
 def run(condition,pid,s):
  m='base' if condition=='base' else 'lora';r=historical[m,pid,s];tok=tokenizers[m]
  identity=dict(run_identifier=identifier,condition=condition,model=m,prompt_id=pid,sample_id=s)
  cid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24];path=OUT/'conditions'/f'{cid}.json'
  if path.exists():return json.loads(path.read_text())
  encoded={k:v.to('cuda:0') for k,v in tok(r['prompt'],return_tensors='pt').items()}
  previous=next((x for x in reuse if condition in ['base','intact_lora'] and x['kind']=='baseline' and x['model']==m and x['prompt_id']==pid and x['sample_id']==s),None)
  if previous:
   assert previous['generation_config']==gc.to_dict() and previous['sample_seed']==r['sample_seed'] and previous['input_ids']==encoded['input_ids'][0].tolist()
   prior=json.loads((ROOT/'results/steering/reproduced_protocol_stance_development/experiment_config.json').read_text());assert prior['preset_sha256']==config['preset_sha256']
   tokens=previous['generated_token_ids'];seconds=0.;reused=previous['condition_id']
  else:
   no_hooks();assert adapter_state(model,modules)==initial
   torch.cuda.synchronize();start=time.perf_counter()
   with ablate_lora_blocks(model,inventory,config['conditions'][condition]) as state:
    random.seed(r['sample_seed']);np.random.seed(r['sample_seed']);torch.manual_seed(r['sample_seed']);torch.cuda.manual_seed_all(r['sample_seed'])
    output=model.generate(**encoded,generation_config=copy.deepcopy(gc),return_dict_in_generate=True,output_logits=False,output_scores=False,logits_to_keep=1)
   torch.cuda.synchronize();seconds=time.perf_counter()-start;tokens=output.sequences[0,encoded['input_ids'].shape[1]:].tolist();del output
   assert adapter_state(model,modules)==initial;no_hooks();reused=None
  result={**identity,'condition_id':cid,'prompt_text':r['prompt'],'sample_seed':r['sample_seed'],'input_ids':encoded['input_ids'][0].tolist(),'generated_token_ids':tokens,'continuation':tok.decode(tokens,skip_special_tokens=True),'generated_tokens':len(tokens),'stopping_reason':'eos' if tokens and tokens[-1]==tok.eos_token_id else 'max_new_tokens','runtime_seconds':seconds,'disabled_saved_layers':config['conditions'][condition],'disabled_modules':[x['expected_peft_module_path'] for x in inventory if int(x['saved_layer']) in config['conditions'][condition]],'reused_condition_id':reused,'generation_config':gc.to_dict(),'resources':resources()}
  atomic(path,result);print(f"COMPLETED {len(list((OUT/'conditions').glob('*.json')))}/60 {condition} {pid} sample{s} reused={bool(reused)} seconds={seconds:.1f}",flush=True)
  return result
 try:
  for condition in config['conditions']:
   for pid in selection['selected_prompt_ids']:
    for s in [0,1,2]:run(condition,pid,s)
 except Exception as error:
  atomic(OUT/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise
 finally:atomic(OUT/f'resources_{session}.json',dict(gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),**resources()))
 print('GENERATION COMPLETE',flush=True)
if __name__=='__main__':main()
