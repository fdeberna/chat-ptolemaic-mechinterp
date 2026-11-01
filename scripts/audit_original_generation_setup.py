"""Read-only provenance audit; stop without inference if Qwen evaluation is absent."""
import csv
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'chat-ptolemaic'
OUT=ROOT/'results/generation_diagnostics/original_setup_audit'


def git(*args):
    return subprocess.check_output(['git','-C',str(ORIGINAL),*args],text=True)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    current=json.loads((ROOT/'results/steering/pilot_L16_stance_L6_style/config.json').read_text())
    adapter=Path(current['adapter'])
    snapshot=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/{current["revision"]}'
    sources=[ORIGINAL/'README.md',ORIGINAL/'evaluation/prompts.txt',ORIGINAL/'nanogpt/generate.py',ORIGINAL/'src/translate.py',ORIGINAL/'.gitignore',
        adapter/'adapter_config.json',adapter/'README.md',adapter/'tokenizer_config.json',adapter/'special_tokens_map.json',adapter/'chat_template.jinja',adapter/'checkpoint-1000/trainer_state.json',
        snapshot/'generation_config.json',ROOT/'scripts/pilot_activation_steering.py',ROOT/'results/steering/pilot_L16_stance_L6_style/config.json',ROOT/'results/steering/pilot_L16_stance_L6_style/smoke.json',Path(__file__).resolve()]
    hashes={str(p):digest(p) for p in sources}
    hashes[str(adapter/'adapter_model.safetensors')]=digest(adapter/'adapter_model.safetensors')
    hashes[str(adapter/'tokenizer.json')]=digest(adapter/'tokenizer.json')
    assert hashes[str(adapter/'adapter_model.safetensors')]==current['adapter_sha256']
    assert hashes[str(adapter/'tokenizer.json')]==current['tokenizer_sha256']
    source_dir=OUT/'source_snapshots';source_dir.mkdir(exist_ok=True)
    for i,p in enumerate(sources):shutil.copy2(p,source_dir/f'{i:02d}_{p.name}')
    history={name:git('show',f'd8e3ef6:scripts/inspect/{name}') for name in ['sample.py','sample2.py','sample3.py','sample4.py']}
    for name,text in history.items():(source_dir/f'historical_{name}').write_text(text)
    lines=(ORIGINAL/'evaluation/prompts.txt').read_text().splitlines()
    roles={6:'Candidate for daily/yearly cycles; does not explicitly ask about Earth rotation',5:'Candidate for Earth–Sun model relation',8:'Candidate explanatory framing: apparent planetary loops',10:'Candidate explanatory framing: lunar phases'}
    candidates=[dict(candidate_id=f'evaluation_prompts_line_{line:03d}',source_line=line,prompt_text=lines[line-1],intended_role=role,
        original_run_prompt_id='UNKNOWN; source file has no IDs',verified_Qwen_behavioral_prompt=False,selected_by='topic coverage only; no responses inspected or generated') for line,role in roles.items()]
    with (OUT/'candidate_prompts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    table=[
      ('Base model','Adapter export names Qwen/Qwen2.5-7B; verified export metadata, not an original generation log','Qwen/Qwen2.5-7B; verified saved pilot config',str(adapter/'adapter_config.json')+':6'),
      ('Base revision','Unknown: adapter revision=null; no original run pin recovered',current['revision'],'adapter_config.json:31; pilot config revision'),
      ('Adapter','Final export is accessible and hashes match current pilot; which checkpoint original behavioral run used is unknown',str(adapter)+'; disabled for base, active for LoRA','adapter_config.json; SHA256 in manifest'),
      ('Tokenizer','Adapter-local Qwen2 tokenizer files exist; actual original generation tokenizer/formatting call unknown','Adapter-local tokenizer, checked against base on pilot prompts','tokenizer_config.json; pilot_activation_steering.py'),
      ('Prompt format','Unknown: no Qwen evaluation runner or rendered prompts recovered','Plain text continuation, add_special_tokens=True; no chat template','pilot_activation_steering.py encode/decode'),
      ('Special tokens','Export declares add_bos_token=false; EOS/PAD <|endoftext|> ID151643. Chat-template file also exists; its use in original run is unknown','No chat rendering; tokenizer special-token behavior preserved; EOS151643; PAD=EOS','tokenizer_config.json:2,5; special_tokens_map.json; cached generation_config.json'),
      ('Quantization / dtype','Original behavioral inference settings unknown. QLoRA export name does not establish inference quantization','NF4 double quantization; FP16 base compute/storage, FP32 adapter parameters','pilot config numerical; smoke.json actual parameter dtypes'),
      ('Attention / device','Unknown for original Qwen behavioral run','SDPA, single GPU cuda:0, no offload','pilot config and smoke.json'),
      ('Sampling / temperature / top-p / top-k','Unknown; no original Qwen invocation or saved effective generation config','Greedy raw-logit argmax; no sampling, temperature/top-p/top-k warpers','pilot_activation_steering.py decode'),
      ('Repetition / n-gram / length penalties','Unknown for original Qwen run','No logit penalties or no-repeat filter; manual argmax loop','pilot_activation_steering.py decode'),
      ('Stopping / output length','Unknown for original Qwen run','EOS151643 or 96 new tokens; KV cache, batch1','pilot config; cached generation_config.json'),
      ('Seeds','Unknown original run seeds','Greedy, torch seed0; no stochastic decoding','pilot_activation_steering.py'),
    ]
    with (OUT/'setup_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['setting','original_evidence','current_pilot','source']);w.writerows(table)
    manifest=dict(status='stopped_missing_original_Qwen_generation_configuration',generations=0,model_loads=0,GPU_used=False,dependencies_changed=False,steering_run=False,
        original_repo=str(ORIGINAL),original_HEAD=git('rev-parse','HEAD').strip(),branches=git('branch','-a').splitlines(),
        available_history_generation_related_paths=git('log','--all','--format=','--name-only','--','*qwen*','*lora*','*eval*','*generat*','*sample*').splitlines(),
        current_generation=current['generation'],source_sha256=hashes,historical_sample_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in history.items()},
        elapsed_seconds=time.perf_counter()-start,peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        search_scope='Original checkout including ignored output/evaluation files, relevant source/config files, data directory names and likely generation metadata filenames, and available local/remote-tracking branch histories. Python/Windows dependency trees and corpus contents were excluded as irrelevant. No remote fetch, dependency changes, checkpoint unpickling or model loading.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report=['# Original behavioral-generation setup audit: stopped before inference','',
      '**No new generations or model loads were run.** Steering sweeps remain paused. The original repository and adapter export are available, but the Qwen/LoRA behavioral generation runner/configuration and generation records could not be recovered. Per the requested fallback, no replacement B configuration or arbitrary decoding experiment was invented.','',
      '## Verified comparison','',
      '\n'.join(['| Setting | Original evidence | Current steering pilot |','|---|---|---|']+[f'| {a} | {b} | {c} |' for a,b,c,_ in table]),
      '', '## Search and exclusions','',
      f'Inspected {ORIGINAL}, HEAD {manifest["original_HEAD"]}, its available local and remote-tracking branch histories, source files, evaluation directory and adapter/checkpoint output metadata. The current repository README describes corpus construction and small-model training, not the later Qwen behavioral experiment. No Qwen evaluation runner, notebook, shell invocation, saved rendered prompt, generated response file or effective generation_config.json was found in the accessible original materials.',
      'nanogpt/generate.py:18–30,33–40 samples a custom GPT checkpoint with defaults temperature=0.8, max_new_tokens=200, optional top-k. These are for a different model/tokenizer, not verified Qwen settings. src/translate.py:119–143 translates corpus chunks using another pipeline; its unused repetition-penalty kwargs and actual greedy translation call are not Qwen behavioral evidence. Historical scripts/inspect/sample*.py at d8e3ef6 inspect PDF pages, not model generations. None of these settings was substituted for B.',
      'The cached Qwen base generation_config.json declares EOS151643, do_sample=false and max_new_tokens=2048. These are repository defaults in the current cached snapshot, not evidence that the original behavioral script used those values or omitted overrides. The adapter has chat_template.jinja; existence of a template is not evidence it was applied. The trainer_state.json records training progress, not generation settings. No unsafe training-state deserialization was attempted.',
      '', '## Exact prompt candidates, not an established original Qwen prompt set','',
      'evaluation/prompts.txt contains 12 neutral prompts without IDs. It is available in the original repository but predates the April adapter export; no provenance links it to the specific Qwen behavioral run. Four candidates were chosen by topic coverage only, without examining model responses. IDs below are assigned source-line locators, not recovered original run IDs.',
      *[f'- {c["candidate_id"]}: {c["prompt_text"]} ({c["intended_role"]}.)' for c in candidates],
      'The daily/yearly-cycle candidate does not explicitly ask about Earth’s daily rotation. Therefore the requested exact four-prompt original-Qwen set, including a direct Earth-rotation prompt, is not established. No wording was invented to fill that gap. candidate_prompts.csv preserves exact available wording and this uncertainty.',
      '', '## Diagnostic status and smallest remaining issue','',
      'A/B execution, generation examples, quality-by-decoding comparisons, GPU resource measurements and seed comparisons are unavailable because B cannot be specified from verified original-run evidence. Zero outputs were produced, rather than silently testing a replacement configuration. This audit used only lightweight CPU/file reads; elapsed time and host RSS are in manifest.json. No intervention hooks were registered and no generation process was started.',
      'We do not yet have a baseline setup demonstrated usable for resuming steering, and this audit cannot attribute the earlier problems to decoding, prompt formatting, adapter behavior or numerical loading. The smallest next requirement is the original Qwen evaluation script/notebook or exact command plus prompt list/rendering and effective generation kwargs (including defaults/overrides), ideally accompanied by one saved original prompt/response record identifying its model/adapter checkpoint. Original model revision and inference precision also remain unresolved.',
      'Once those are available, the already specified ≤32-generation comparison can isolate decoding using the unchanged NF4 loading setup. Do not silently attempt an unquantized 7B load. The unquantized-probe versus NF4-generation compatibility check remains a separate pending task.',
      '', '## Files and reproduction','',
      'setup_comparison.csv provides setting-level source references. candidate_prompts.csv preserves exact source wording and coverage limitations. manifest.json records source hashes, original Git HEAD/branches, search scope, zero-generation status and CPU resource usage. source_snapshots/ preserves relevant readable evidence, including this audit script and historical PDF-inspection code. Previous experiment files and original repository files were not modified.',
      '```bash\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/audit_original_generation_setup.py\n```']
    (OUT/'report.md').write_text('\n\n'.join(report).replace('\n\n\n\n','\n\n')+'\n')
    (OUT/'examples.md').write_text('# No new generation examples\n\nNo inference was run because the original Qwen behavioral generation configuration could not be recovered. Exact available prompt candidates are in candidate_prompts.csv; they are not generated continuations or a verified original Qwen evaluation set.\n')
    print(json.dumps({k:manifest[k] for k in ['status','generations','model_loads','GPU_used','elapsed_seconds','peak_host_ram_bytes']},indent=2))
    print(OUT/'report.md')


if __name__=='__main__':main()
