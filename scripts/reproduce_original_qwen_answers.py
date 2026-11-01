"""Twenty matched historical-sample replays and two original-helper parity checks."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'chat-ptolemaic'
OUT=ROOT/'results/generation_diagnostics/original_answers_10_prompt_replay'
PROMPT_IDS=['astro_000','astro_005','astro_010','astro_015','astro_022','declarative_000','questions_002','questions_004','questions_009','questions_012']
PARAMS=['max_new_tokens','temperature','top_p','do_sample','repetition_penalty','no_repeat_ngram_size']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,sort_keys=True));temp.replace(path)

def main():
    for name in ['conditions','parity','source_snapshots']:(OUT/name).mkdir(parents=True,exist_ok=True)
    historical_paths={m:ORIGINAL/f'evaluation/generation_outputs_qwen/{filename}' for m,filename in [('base','qwen25_base_eval.jsonl'),('lora','qwen25_lora1000_eval.jsonl')]}
    script=ORIGINAL/'scripts/qlora/generate_eval_qwen.py'
    spec=importlib.util.spec_from_file_location('original_qwen_eval',script);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    selected={}
    for m,path in historical_paths.items():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for pid in PROMPT_IDS:
            matches=[r for r in rows if r['prompt_id']==pid and r['sample_id']==0];assert len(matches)==1
            r=matches[0];assert r['model_name']=='Qwen/Qwen2.5-7B'
            assert original.make_sample_seed(r['seed'],r['prompt_file'],r['prompt_id'],r['sample_id'])==r['sample_seed']
            prompt_file=ORIGINAL/r['prompt_file'].removeprefix('./')
            prompt_specs=original.load_prompts([prompt_file],ORIGINAL,None)
            assert next(x['prompt'] for x in prompt_specs if x['prompt_id']==pid)==r['prompt']
            selected[m,pid]=r
    for pid in PROMPT_IDS:
        b,l=selected['base',pid],selected['lora',pid]
        for field in ['prompt','sample_seed','sample_id',*PARAMS]:assert b[field]==l[field],(pid,field)
    prior=json.loads((ROOT/'results/generation_diagnostics/recovered_defaults_comparison/config.json').read_text())
    adapter=Path(prior['adapter']);snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{prior["revision"]}'
    assert all(r['adapter_dir'] is None for (m,_),r in selected.items() if m=='base')
    assert all((ORIGINAL/r['adapter_dir'].removeprefix('./')).resolve()==adapter.resolve() for (m,_),r in selected.items() if m=='lora')
    sources=[Path(__file__),script,ORIGINAL/'README.md',*historical_paths.values(),adapter/'adapter_config.json',adapter/'adapter_model.safetensors',adapter/'tokenizer.json',snapshot/'generation_config.json',ROOT/'scripts/diagnose_unsteered_decoding.py']
    config=dict(prompt_ids=PROMPT_IDS,sample_id=0,selection='Ten prompts fixed by topic coverage without screening response quality; recorded sample0 per model, not all five samples. Original prompt wording is used after source numbering cleanup.',
        original_repo=str(ORIGINAL),model=prior['model'],revision=prior['revision'],adapter=str(adapter),numerical=prior['numerical'],
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes']},execution=dict(CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'),inference_mode=True,batch_size=1,no_offload=True),
        sources={str(p):sha(p) for p in sources},historical_settings={f'{m}/{pid}':{k:r[k] for k in ['prompt_file','prompt_id','prompt','sample_id','sample_seed','seed','num_return_sequences',*PARAMS]} for (m,pid),r in selected.items()})
    assert config['sources'][str(adapter/'adapter_model.safetensors')]==prior['sources'][str(adapter/'adapter_model.safetensors')]
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['run_identifier']=identifier
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config,'Incompatible resume'
    atomic(OUT/'config.json',config)
    pd.DataFrame([dict(prompt_id=pid,prompt=selected['base',pid]['prompt'],prompt_file=selected['base',pid]['prompt_file'],sample_id=0,sample_seed=selected['base',pid]['sample_seed']) for pid in PROMPT_IDS]).to_csv(OUT/'selected_prompts.csv',index=False)
    for p in [script,Path(__file__)]:shutil.copy2(p,OUT/'source_snapshots'/p.name)
    for m,path in historical_paths.items():shutil.copy2(path,OUT/'source_snapshots'/path.name)
    tick=time.perf_counter();session=str(time.time_ns());torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert torch.cuda.is_available(),'Approved GPU execution required';torch.cuda.reset_peak_memory_stats()
    tokenizers={}
    for m,path in [('base',snapshot),('lora',adapter)]:
        tok=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tokenizers[m]=tok
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,quantization_config=quant,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    core=model.get_base_model().model
    assert all(not block._forward_hooks for block in core.layers) and not core.norm._forward_hooks
    def context(m):return model.disable_adapter() if m=='base' else nullcontext()
    def resources():return dict(elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def current_config(m,record):
        gc=copy.deepcopy(model.generation_config);kwargs={k:record[k] for k in PARAMS};kwargs.update(pad_token_id=tokenizers[m].pad_token_id,eos_token_id=tokenizers[m].eos_token_id,use_cache=True,num_beams=1)
        unused=gc.update(**kwargs);assert not unused
        return gc
    atomic(OUT/'effective_settings.json',{m:dict(generation_config=current_config(m,selected[m,PROMPT_IDS[0]]).to_dict(),tokenizer_source=str(snapshot if m=='base' else adapter),tokenizer_class=type(tokenizers[m]).__name__,eos=tokenizers[m].eos_token_id,pad=tokenizers[m].pad_token_id,format='tokenizer(prompt, return_tensors=pt); no chat template',num_return_sequences_explanation='Historical record5 is outer sample-loop count. Each generate() call returns one sample; this replay chooses recorded sample_id0 only.') for m in ['base','lora']})
    @torch.inference_mode()
    def run(m,pid):
        r=selected[m,pid];path=OUT/'conditions'/f'{m}_{pid}_sample0.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['run_identifier']==identifier;return row
        tok=tokenizers[m];inputs={k:v.to('cuda:0') for k,v in tok(r['prompt'],return_tensors='pt').items()};original.set_seed(r['sample_seed'])
        gc=current_config(m,r);torch.cuda.synchronize();start=time.perf_counter()
        with context(m):out=model.generate(**inputs,generation_config=gc,return_dict_in_generate=True,output_scores=False,output_logits=False,logits_to_keep=1)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        n=inputs['input_ids'].shape[1];ids=out.sequences[0,n:].tolist();completion=tok.decode(ids,skip_special_tokens=True);del out
        old=r['completion'];prefix=0
        for a,b in zip(old,completion):
            if a!=b:break
            prefix+=1
        row=dict(run_identifier=identifier,model=m,prompt_id=pid,prompt=r['prompt'],input_ids=inputs['input_ids'][0].tolist(),sample_id=0,sample_seed=r['sample_seed'],historical_record=r,historical_source=str(historical_paths[m]),historical_source_sha256=config['sources'][str(historical_paths[m])],completion=completion,generated_token_ids=ids,generated_tokens=len(ids),stopping_reason='eos' if ids and ids[-1]==tok.eos_token_id else 'max_new_tokens',exact_text_match=completion==old,common_prefix_characters=prefix,historical_characters=len(old),current_characters=len(completion),effective_generation_config=gc.to_dict(),runtime_seconds=elapsed,resources=resources())
        atomic(path,row);print(f'COMPLETED {len(list((OUT/"conditions").glob("*.json")))}/20 {m} {pid} exact={row["exact_text_match"]} prefix_chars={prefix} seconds={elapsed:.1f}',flush=True);return row
    try:
        for m in ['base','lora']:
            for i,pid in enumerate(PROMPT_IDS):
                row=run(m,pid)
                if i==0:
                    path=OUT/'parity'/f'{m}.json'
                    if path.exists():assert json.loads(path.read_text())['token_ids_identical'];continue
                    r=selected[m,pid];captured={}
                    class Proxy:
                        def generate(self,**kwargs):
                            output=model.generate(**kwargs);captured['full_ids']=output[0].tolist();return output
                    start=time.perf_counter()
                    with context(m),torch.inference_mode():text=original.generate_completion(Proxy(),tokenizers[m],r['prompt'],SimpleNamespace(**{k:r[k] for k in PARAMS}),r['sample_seed'])
                    torch.cuda.synchronize();duration=time.perf_counter()-start
                    ids=captured['full_ids'][len(row['input_ids']):]
                    result=dict(model=m,prompt_id=pid,current_runner_text=row['completion'],original_helper_text=text,original_helper_token_ids=ids,token_ids_identical=ids==row['generated_token_ids'],text_identical=text==row['completion'],runtime_seconds=duration)
                    atomic(path,result);assert result['token_ids_identical'] and result['text_identical'],'Original helper/current wrapper mismatch; stop for diagnosis'
                    print('PARITY PASS '+m,flush=True)
    except Exception as error:
        atomic(OUT/f'failure_{session}.json',dict(error=repr(error),traceback=traceback.format_exc(),resources=resources()));raise
    finally:atomic(OUT/f'resources_{session}.json',dict(gpu=torch.cuda.get_device_name(0),placement=sorted({str(p.device) for p in model.parameters()}),dtypes=sorted({str(p.dtype) for p in model.parameters()}),**resources()))
    rows=[json.loads(p.read_text()) for p in (OUT/'conditions').glob('*.json')];assert len(rows)==20
    with (OUT/'comparisons.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    pd.DataFrame([{k:r[k] for k in ['model','prompt_id','sample_seed','exact_text_match','common_prefix_characters','historical_characters','current_characters','generated_tokens','stopping_reason','runtime_seconds']} for r in rows]).to_csv(OUT/'comparison_metrics.csv',index=False)
    examples=['# Ten original prompts: all historical/current answer pairs','', 'One fixed recorded sample_id0 per prompt per model. Exact original text, saved sample seed and recorded generation arguments are reused.','']
    for m in ['base','lora']:
        for pid in PROMPT_IDS:
            r=next(r for r in rows if r['model']==m and r['prompt_id']==pid)
            examples += [f'## {m} / {pid} / sample_seed {r["sample_seed"]}', '',r['prompt'],'',f'Exact decoded-text match: {r["exact_text_match"]}','', '### Historical saved answer','',r['historical_record']['completion'],'','### Current runner','',r['completion'],'']
    (OUT/'examples.md').write_text('\n'.join(examples));print('COMPLETE '+str(OUT),flush=True)

if __name__=='__main__':main()
