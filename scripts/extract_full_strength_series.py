"""Resumable GPU extraction of a separately labeled, consistent NF4/FP16 series."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from validate_logistic_probe_sensitivity import set_lora_strength, strength_dir

GRID = (0., .01, .02, .04, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/full_strength_nf4_fp16/activations")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    old_root = Path("results/logistic_probe_sensitivity/activations")
    old_meta = pd.read_csv(old_root/"t_0p0000/metadata.csv")
    prompts = old_meta[old_meta.layer_index == 0].reset_index(drop=True)
    adapter = Path("../chat-ptolemaic/outputs/qwen25-7b-astronomy-qlora-run2_04212026_FINAL").resolve()
    snapshot = Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{REVISION}"
    assert torch.cuda.is_available(), "GPU unavailable: run outside sandbox"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    specification = dict(model_name_or_path="Qwen/Qwen2.5-7B", model_revision=REVISION,
        adapter_path=str(adapter), adapter_sha256=sha(adapter/"adapter_model.safetensors"),
        adapter_config_sha256=sha(adapter/"adapter_config.json"), dtype="float16",
        load_in_4bit=True, quant_type="nf4", double_quant=True, compute_dtype="float16",
        device_map={"":0}, attention="sdpa", batch_size=1, max_length=None,
        pooling="last non-padding token", layers=list(range(29)), eval_mode=True,
        use_cache=False, output_hidden_states=False, seed=0,
        multiplier="PEFT set_scale(adapter,t): t times configured alpha/r, no merging",
        prompt_ids=prompts.prompt_id.astype(str).tolist(),
        tokenizer_sha256=sha(adapter/"tokenizer.json"),
        versions={p:importlib.metadata.version(p) for p in ("torch","transformers","peft","bitsandbytes")})
    identifier = hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    specification["series_identifier"] = identifier
    specpath = root/"series_config.json"
    if specpath.exists() and json.loads(specpath.read_text()) != specification:
        raise RuntimeError("Existing series has different extraction configuration")
    write_json(specpath,specification)
    tokenizer = AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    base_tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    assert all(tokenizer(p)["input_ids"] == base_tokenizer(p)["input_ids"] for p in prompts.prompt_text)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
        torch_dtype=torch.float16,quantization_config=quant,device_map={"":0},
        attn_implementation="sdpa",low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    core = model.get_base_model().model
    buffer = {}
    token_index = 0
    handles = []
    def hook(layer):
        def capture(module, inputs, output):
            tensor = output[0] if isinstance(output,tuple) else output
            buffer[layer] = tensor[0,token_index,:].detach().float().cpu().numpy().copy()
        return capture
    handles.append(core.embed_tokens.register_forward_hook(hook(0)))
    for i in range(27): handles.append(core.layers[i].register_forward_hook(hook(i+1)))
    handles.append(core.norm.register_forward_hook(hook(28)))
    def forward(index):
        nonlocal token_index
        buffer.clear()
        encoded = tokenizer(prompts.iloc[index].prompt_text,return_tensors="pt",add_special_tokens=True)
        token_index = int(encoded['attention_mask'].sum())-1
        encoded = {k:v.to("cuda:0") for k,v in encoded.items()}
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            result = core(**encoded,use_cache=False,output_hidden_states=False,return_dict=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-tick
        vectors = np.stack([buffer[k] for k in range(29)])
        del result,encoded
        return vectors,elapsed,token_index+1
    pilot = []
    # Verify hook pooling against the standard hidden-state API on one short prompt.
    set_lora_strength(model,0.)
    pooled_check,_,_ = forward(0)
    encoded = {k:v.to('cuda:0') for k,v in tokenizer(prompts.iloc[0].prompt_text,
               return_tensors='pt',add_special_tokens=True).items()}
    with torch.inference_mode():
        check_outputs = core(**encoded,use_cache=False,output_hidden_states=True,return_dict=True)
    conventional = np.stack([h[0,token_index,:].float().cpu().numpy() for h in check_outputs.hidden_states])
    pooling_max_diff = float(np.abs(pooled_check-conventional).max())
    assert pooling_max_diff == 0., 'Hook pooling differs from validated hidden-state convention'
    del encoded,check_outputs,conventional,pooled_check
    pilot_ids = [0, len(prompts)//2, len(prompts)-1]
    for t in (0.,1.):
        set_lora_strength(model,t)
        with np.load(strength_dir(old_root,t)/"activations.npz") as saved:
            old = saved['activations'].reshape(len(prompts),29,3584)
            for index in pilot_ids:
                new,elapsed,length = forward(index)
                diff = new.astype(np.float64)-old[index].astype(np.float64)
                pilot.append(dict(t=t,prompt_id=str(prompts.iloc[index].prompt_id),tokens=length,
                    seconds=elapsed,max_abs_diff=float(np.abs(diff).max()),
                    rms_diff=float(np.sqrt(np.mean(diff**2))),
                    relative_frobenius_diff=float(np.linalg.norm(diff)/np.linalg.norm(old[index]))))
        del old
    placement = {str(p.device) for p in model.parameters()}
    report = dict(series_identifier=identifier,device_placement=sorted(placement),
        hook_vs_hidden_state_max_diff=pooling_max_diff,
        gpu=torch.cuda.get_device_name(0),allocated_vram_bytes=torch.cuda.memory_allocated(),
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
        peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        current_ram_bytes=psutil.Process().memory_info().rss,
        elapsed_seconds=time.perf_counter()-started,prompt_checks=pilot,
        compatible_with_old_series=False,
        reason="Different NF4 quantization and explicit FP16 setup; all strengths regenerated")
    write_json(root.parent/"extraction_pilot.json",report)
    print("PILOT "+json.dumps(report),flush=True)
    if not args.pilot_only:
        for t in GRID:
            dest = strength_dir(root,t)
            dest.mkdir(exist_ok=True)
            if (dest/"manifest.json").exists():
                manifest = json.loads((dest/"manifest.json").read_text())
                assert manifest['series_identifier'] == identifier
                print(f"Reusing complete t={t}",flush=True)
                continue
            set_lora_strength(model,t)
            checkpoint = dest/"checkpoint.npz"
            chunks,timings,lengths = [],[],[]
            if checkpoint.exists():
                with np.load(checkpoint) as a:
                    assert str(a['series_identifier']) == identifier
                    chunks=list(a['pooled']); timings=list(a['seconds']); lengths=list(a['tokens'])
            for index in range(len(chunks),len(prompts)):
                new,elapsed,length=forward(index)
                chunks.append(new);timings.append(elapsed);lengths.append(length)
                if (index+1)%10 == 0 or index+1 == len(prompts):
                    temp=dest/"checkpoint.tmp.npz"
                    np.savez_compressed(temp,pooled=np.stack(chunks),seconds=timings,tokens=lengths,
                                        series_identifier=np.asarray(identifier))
                    temp.replace(checkpoint)
                    print(f"t={t:g} prompts={index+1}/120 mean_s={np.mean(timings):.3f}",flush=True)
            pooled=np.stack(chunks).reshape(-1,3584)
            np.savez_compressed(dest/"activations.npz",activations=pooled)
            meta=old_meta.copy()
            meta['model_condition']=f'nf4_fp16_lora_t_{t:g}'
            meta['adapter_path']=str(adapter)
            meta.to_csv(dest/"metadata.csv",index=False)
            write_json(dest/"manifest.json",dict(activation_file="activations.npz",metadata_file="metadata.csv",
                num_rows=len(pooled),hidden_size=3584,layer_indices=list(range(29)),
                layer_zero="embedding_output",series_identifier=identifier,
                extraction_config=dict(specification,adapter_strength_t=t),
                activation_sha256=sha(dest/"activations.npz"),
                mean_seconds_per_prompt=float(np.mean(timings)),total_forward_seconds=float(np.sum(timings))))
            # Completed stores replace pooled checkpoints; do not duplicate activation arrays.
            checkpoint.unlink()
            del chunks,pooled,meta
    for handle in handles: handle.remove()
    del model,core,p
    gc.collect();torch.cuda.empty_cache()
    report.update(elapsed_seconds=time.perf_counter()-started,
        peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
        final_allocated_vram_bytes=torch.cuda.memory_allocated())
    write_json(root.parent/"extraction_resources.json",report)
    print("Extraction complete; model memory released",flush=True)


if __name__ == "__main__": main()
