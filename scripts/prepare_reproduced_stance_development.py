"""Archive the fixed historical screen and reserve prompts; no inference/judging service."""
from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'chat-ptolemaic'
OUT=ROOT/'results/steering/reproduced_protocol_stance_development'
NOTES={
 'questions_004':'Not selected: opposing systems are discussed, attributed to others, or left undecided; a historical voice does not imply geocentric endorsement. Neither base nor final adapter gives a stable explicit binary answer across all five.',
 'questions_009':'Not selected: base answers vary between motion claims/history and contradictions; final-adapter samples often produce headings or question lists instead of a substantive answer. LoRA500 includes objections that must not be treated as endorsements.',
 'questions_012':'Selected: all five base samples explain planetary orbits/relative motion using Sun-centered language. Final-adapter samples0/1/4 are geometry/excerpt-like with unresolved global stance, while2 explicitly states immovable central Earth and3 explicitly uses Earth-concentric planet circles. This is not a five-of-five H-versus-G pair. LoRA500 likewise mixes Earth-centered mechanisms and ambiguous descriptions. Selection relies on mechanisms/explicit claims, not historical words alone.',
 'astro_022':'Not selected: base itself sometimes endorses Sun-around-Earth or stationary-Earth accounts. Final-adapter differences are frequently geometry/style rather than explicit stance; no stable binary contrast.',
 'astro_024':'Selected as the second-best component-level comparison, not a stable binary contrast: base samples0/4 explicitly posit Earth-around-Sun motion; sample1 attributes it to ancients,2 is unresolved and3 has ambiguous rotation-around-Sun wording. Final-adapter0/1 affirm axial rotation (not heliocentrism),2 combines rotating Earth with Earth at universe center,3/4 invoke heavenly revolutions. No final-adapter sample explicitly affirms Earth-around-Sun revolution. LoRA500 also mixes rotation and Earth-centered accounts. Rotation and revolution must be scored separately.',
 'declarative_010':'Not selected: shape, reported opinions and arguments against central Earth are mixed; ancient vocabulary is not evidence for an endorsed geocentric system.',
 'questions_000':'Not selected: base often narrates history rather than deciding; final adapter contains unresolved discussion and objections without a final answer. Explicit evidence varies across seeds.',
 'questions_001':'Not selected: base also supplies historical or agnostic arguments; final adapter mostly questions/objections. Earth motion is not consistently Earth-around-Sun revolution.',
 'questions_005':'Not selected: base is also undecided or stationary-Earth in some samples; final adapter question lists and objections cannot be promoted to endorsed stance.',
 'questions_016':'Not selected: conditional hypotheses, objections and replies dominate both variants. Historical style contrast is clearer than an endorsed stance contrast.'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((OUT/'screening_plan.json').read_text());ids=plan['candidate_ids_fixed_before_answer_inspection']+plan['bounded_additional_candidates_before_answer_inspection'];assert len(ids)==10
 models={'base':'qwen25_base_eval.jsonl','lora500':'qwen25_lora500_eval.jsonl','lora1000':'qwen25_lora1000_eval.jsonl'};allrows={};screen=[];sources={}
 for m,name in models.items():
  p=ORIGINAL/'evaluation/generation_outputs_qwen'/name;sources[str(p)]=sha(p);allrows[m]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  for pid in ids:
   rows=[r for r in allrows[m] if r['prompt_id']==pid];assert len(rows)==5 and {r['sample_id'] for r in rows}==set(range(5))
   screen.extend(dict(model_variant=m,selection_note=NOTES[pid],**r) for r in rows)
 assert len(screen)==150
 with (OUT/'historical_screened_samples.jsonl').open('w') as f:
  for r in screen:f.write(json.dumps(r)+'\n')
 pd.DataFrame([dict(prompt_id=p,selected=p in ['questions_012','astro_024'],note=NOTES[p]) for p in ids]).to_csv(OUT/'screening_notes.csv',index=False)
 allids={r['prompt_id'] for r in allrows['base']};exposed=set(ids)|set(plan['previously_exposed_replay_ids']);reserved=sorted(allids-exposed);assert len(allids)==112 and len(reserved)==96
 dev=['questions_012','astro_024'];assignment=[]
 for pid in sorted(allids):
  assignment.append(dict(prompt_id=pid,prompt=next(r['prompt'] for r in allrows['base'] if r['prompt_id']==pid),partition='development' if pid in dev else 'screened_not_confirmation' if pid in ids else 'previously_exposed_not_confirmation' if pid in exposed else 'reserved_confirmation'))
 pd.DataFrame(assignment).to_csv(OUT/'prompt_partitions.csv',index=False)
 selection=dict(selected_prompt_ids=dev,steering_sample_ids=[0,1,2],historical_screen_sample_ids=[0,1,2,3,4],source_sha256=sources,screened_prompt_ids=ids,reserved_confirmation_ids=reserved,
  important_limitation='No screened prompt gave a clean stable five-sample base-H versus final-LoRA-G opposition. These two are the best available mechanistically interpretable development comparisons. Historical vocabulary is not counted as geocentrism; ambiguous/quoted/objection-only samples remain ambiguous.',
  selection_bias='Unblinded response-informed development selection; not final causal evaluation. All five samples inspected, but first three sample IDs fixed for steering regardless of favorable outputs. Reserved prompts must not be used in this test.',
  budget=dict(new_steered=48,new_baselines=12,zero_hook_checks=2),lora500_role='Historical selection context only; do not load this adapter for steering.')
 (OUT/'development_selection.json').write_text(json.dumps(selection,indent=2))
 lines=['# Historical development-set screen','',selection['important_limitation'],'','All150 existing outputs in the ten-prompt screen were read: five samples each for base, LoRA500 and LoRA1000. No generation was used for selection. These are informal unblinded development observations, not independent semantic labels. LoRA500 is a different run/export, so no monotonic training-dose inference is made.','']
 for pid in ids:lines += [f'## {pid}',NOTES[pid],'']
 lines += ['# Partitions','Two selected development prompts; eight other screened prompts excluded from confirmation; six other previously exposed replay prompts also excluded;96 prompts reserved for later confirmation. The source historical answers already exist, but reserved answers were not used in this selection. No confirmation generations will be run.','', 'The strict five-sample categorical-contrast objective was not met. The controlled development test is retained to examine protocol dependence on the best available comparisons, with mixed/unclear outcomes retained and without a success claim based on vocabulary.']
 (OUT/'screening_report.md').write_text('\n\n'.join(lines));print('Archived150 screened outputs, selected2 with explicit limitations, reserved96; test48 steered +12 fresh baselines +2 zero-hook checks.')
if __name__=='__main__':main()
