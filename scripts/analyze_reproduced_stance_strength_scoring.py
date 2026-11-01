"""Read-only validation and descriptive analysis of supplied sweep annotations."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import argparse,csv,hashlib,itertools,json,re,shutil,sys,tempfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/stance-sweep-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/steering/reproduced_protocol_stance_strength_sweep'
QUALITY=['full_continuation_'+x for x in ['repetition','incoherence','topic_drift','truncation']]
CATS={'stance_label':['geocentric','heliocentric','mixed','unclear'],'earth_rotation':['affirmed','denied','conflicting','unmentioned'],'earth_revolution_around_sun':['affirmed','denied','conflicting','unmentioned'],'style_label':['modern','premodern','mixed','unclear'],**{x:['yes','no','unclear'] for x in QUALITY}}
EXCERPTS=['stance_supporting_excerpt','earth_rotation_supporting_excerpt','earth_revolution_supporting_excerpt','style_supporting_excerpt']
FIELDS=['stance_label',EXCERPTS[0],'earth_rotation',EXCERPTS[1],'earth_revolution_around_sun',EXCERPTS[2],'style_label',EXCERPTS[3],'semantic_notes',*QUALITY,'quality_notes','reviewer_id']
MODELS=['base','lora'];PROMPTS=['questions_012','astro_024'];MAG=[.1,.3,.5]
CONDITIONS=[('baseline',0.)]+[(d,k) for d in ['own','random'] for k in [-.5,-.3,-.1,.1,.3,.5]]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def md(df):
 def c(x):return str(x).replace('|','\\|').replace('\n','<br>')
 return '\n'.join(['| '+' | '.join(map(c,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(c(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=DATA/'scored_analysis_20261030');out=parser.parse_args().output_dir
 out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise RuntimeError('Use a new empty analysis directory; existing results are never overwritten')
 errors=[];flags=[]
 def check(ok,message):
  if not ok:errors.append(message)
 def stop():
  if errors:
   writej(out/'validation_results.json',dict(valid=False,errors=errors));print('\n'.join(errors));sys.exit(1)
 def loadlines(path):
  result=[]
  def unique(items):
   obj={}
   for k,v in items:
    if k in obj:raise ValueError('duplicate JSON key '+k)
    obj[k]=v
   return obj
  for i,line in enumerate(path.read_text().splitlines(),1):
   if not line.strip():continue
   try:
    obj=json.loads(line,object_pairs_hook=unique);assert isinstance(obj,dict);result.append(obj)
   except Exception as e:errors.append(f'{path.name}:{i}: {e}')
  return result
 inputs=[DATA/x for x in ['blinded_review_scored.jsonl','blinded_review.jsonl','condition_key.csv','rubric.md','generations.jsonl','experiment_config.json','intervention_diagnostics.csv']]
 for p in inputs:check(p.is_file(),'Missing '+str(p))
 stop()
 scored=loadlines(inputs[0]);original=loadlines(inputs[1]);key=list(csv.DictReader(inputs[2].open()));gens=loadlines(inputs[4]);cfg=json.loads(inputs[5].read_text())
 for name,rows,field in [('scored',scored,'review_id'),('original',original,'review_id'),('key',key,'review_id'),('generation',gens,'condition_id')]:
  ids=[x.get(field) for x in rows];check(len(rows)==156,f'{name}: expected156, got{len(rows)}');check(all(isinstance(x,str) and x for x in ids),name+': invalid IDs');check(len(set(ids))==156,name+': duplicate IDs')
 stop()
 check({x['review_id'] for x in scored}=={x['review_id'] for x in original}=={x['review_id'] for x in key},'Scored/original/key ID sets differ')
 check({x['condition_id'] for x in key}=={x['condition_id'] for x in gens},'Key/generation IDs differ')
 def identity(k):return (k['model'],k['prompt_id'],int(k['sample_id']),'baseline' if k['kind']=='baseline' else k['direction_source'],float(k['k']))
 expected={(m,p,s,d,k) for m,p,s,(d,k) in itertools.product(MODELS,PROMPTS,range(3),CONDITIONS)}
 try:actual=[identity(k) for k in key];check(set(actual)==expected and len(set(actual))==156,'Incorrect model/prompt/sample/direction/strength coverage')
 except Exception as e:errors.append('Invalid condition metadata: '+repr(e))
 stop()
 original={x['review_id']:x for x in original};keys={x['review_id']:x for x in key};genmap={x['condition_id']:x for x in gens};data=[]
 baseids={(r['model'],r['prompt_id'],r['sample_id']):r['condition_id'] for r in gens if r['kind']=='baseline'}
 for a in scored:
  rid=a['review_id'];o=original[rid];k=keys[rid];g=genmap[k['condition_id']]
  for field in FIELDS:check(field in a and isinstance(a[field],str) and bool(a[field].strip()),f'{rid}: missing/empty/non-string {field}')
  for field,cats in CATS.items():check(a.get(field) in cats,f'{rid}: invalid {field} {a.get(field)!r}')
  for field in ['prompt_context','scoring_window_first_80_words','scored_word_count','full_continuation']:check(a.get(field)==o.get(field),f'{rid}: modified {field}')
  words=list(re.finditer(r'\S+',o['full_continuation']));window=o['full_continuation'] if len(words)<=80 else o['full_continuation'][:words[79].end()]
  check(o['scoring_window_first_80_words']==window and o['scored_word_count']==min(80,len(words)),f'{rid}: original window/word count incorrect')
  check(a['prompt_context']==g['prompt_text'] and a['full_continuation']==g['continuation'],f'{rid}: generation text differs')
  for field in ['model','prompt_id','kind','condition_id']:check(k[field]==g[field],f'{rid}: generation/key {field} mismatch')
  check(int(k['sample_id'])==g['sample_id'] and int(k['sample_seed'])==g['sample_seed'] and float(k['k'])==g['k'] and float(k['gamma'])==g['gamma'],f'{rid}: seed/k/gamma mismatch')
  b=genmap[baseids[g['model'],g['prompt_id'],g['sample_id']]]
  check(k['baseline_id']==b['condition_id'] and g['input_ids']==b['input_ids'] and g['sample_seed']==b['sample_seed'],f'{rid}: incorrect baseline pairing')
  check(g['adapter_active']==(g['model']=='lora'),f'{rid}: adapter state mismatch')
  check(g['generation_config']==b['generation_config'],f'{rid}: baseline settings differ')
  if g['kind']=='baseline':check(g['k']==g['gamma']==0 and g['hook_state'] is None,f'{rid}: baseline steered')
  else:
   check(k['target']=='stance' and int(float(k['layer']))==g['layer']==16 and k['direction_source']==g['direction_source'],f'{rid}: wrong target/layer/direction')
   check(abs(g['gamma']-g['k']*cfg['activation_scale'])<1e-12,f'{rid}: wrong gamma sign/scale')
   h=g['hook_state'];check(h['calls']==g['generated_tokens'] and h['prefill_calls']==1 and h['decode_calls']==g['generated_tokens']-1 and np.isfinite(h['ratios']).all(),f'{rid}: invalid hook diagnostics')
  for field in EXCERPTS:
   if a.get(field) and a[field] not in window:flags.append(dict(review_id=rid,condition_id=g['condition_id'],type='excerpt_not_exact_window_substring',field=field,detail='Excerpt may be edited, composite or outside scored window; review only. Supplied text preserved.',supplied_value=a[field]))
  if a.get('stance_label')=='heliocentric' and a.get('earth_revolution_around_sun')=='denied':flags.append(dict(review_id=rid,condition_id=g['condition_id'],type='stance_motion_tension',field='stance_label / earth_revolution_around_sun',detail='Heliocentric stance with denied Earth-around-Sun revolution; review context; no relabeling.',supplied_value='heliocentric / denied'))
  if a.get('stance_label')=='geocentric' and a.get('earth_revolution_around_sun')=='affirmed':flags.append(dict(review_id=rid,condition_id=g['condition_id'],type='stance_motion_tension',field='stance_label / earth_revolution_around_sun',detail='Geocentric stance with affirmed Earth-around-Sun revolution; review context; no relabeling.',supplied_value='geocentric / affirmed'))
  if (a.get('full_continuation_truncation')=='no' and g['stopping_reason']=='max_new_tokens') or (a.get('full_continuation_truncation')=='yes' and g['stopping_reason']=='eos'):
   flags.append(dict(review_id=rid,condition_id=g['condition_id'],type='truncation_vs_stopping_metadata',field='full_continuation_truncation',detail='Visible-text judgment differs from stopping metadata; not necessarily an annotation error.',supplied_value=a.get('full_continuation_truncation')))
  data.append({**a,**k,'sample_id':g['sample_id'],'sample_seed':g['sample_seed'],'k':g['k'],'gamma':g['gamma'],'direction':'baseline' if g['kind']=='baseline' else g['direction_source'],'condition':'baseline' if g['kind']=='baseline' else f'{g["direction_source"]}{g["k"]:+g}','generated_tokens':g['generated_tokens'],'stopping_reason':g['stopping_reason']})
 # Verify sign using existing raw probe artifacts; no fitting or model import.
 signs=[];pilot=ROOT/'results/steering/pilot_L16_stance_L6_style/directions.npz'
 check(sha(pilot)==cfg['sources'][str(pilot)],'Direction artifact hash differs from experiment provenance')
 with np.load(pilot) as vectors:
  for m in MODELS:
   path=ROOT/f'results/probes/pilot_full/{m}/stance/full_data_coefficients/layer_016_full_data.npz'
   with np.load(path) as probe:
    w=(probe['coef']/probe['scaler_scale']).reshape(-1);u=w/np.linalg.norm(w);mapping=json.loads(str(probe['label_mapping']))
    check(mapping=={'geocentric':0,'heliocentric':1} and str(probe['positive_class'])=='heliocentric' and np.array_equal(u,vectors[f'stance__{m}']),m+': own probe direction/sign mismatch')
    signs.append(dict(model=m,positive_class='heliocentric',label_mapping=mapping,raw_norm=float(np.linalg.norm(w)),direction_matches=True,probe_path=str(path),probe_sha256=sha(path)))
 check(sha(DATA/'rubric.md')==cfg['rubric_sha256'],'Frozen rubric hash differs')
 stop()
 writej(out/'validation_results.json',dict(valid=True,records=156,coverage='2 prompts ×2 models ×3 sample IDs ×13 conditions',errors=[],labels_and_excerpts_preserved=True,text_matches_original_and_generation=True,join='review_id only; never row order',quality_categories=['yes','no','unclear'],annotation_review_flags=len(flags),sign_checks=signs,reviewer_ids=sorted({r['reviewer_id'] for r in scored}),reviewer_independence='Not verifiable from supplied reviewer_id alone. No new judging performed.'))
 pd.DataFrame(flags,columns=['review_id','condition_id','type','field','detail','supplied_value']).to_csv(out/'annotation_review_flags.csv',index=False)
 frame=pd.DataFrame(data).sort_values(['model','prompt_id','sample_id','direction','k'])
 frame.to_csv(out/'joined_annotations.csv',index=False)
 with (out/'joined_annotations.jsonl').open('w') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 counts=[]
 for (m,p,d,k),group in frame.groupby(['model','prompt_id','direction','k']):
  assert len(group)==3
  for field,cats in CATS.items():
   for label in cats:
    n=int((group[field]==label).sum());counts.append(dict(model=m,prompt_id=p,direction=d,k=k,field=field,label=label,count=n,denominator=3,proportion=n/3))
 counts=pd.DataFrame(counts);counts.to_csv(out/'counts_long.csv',index=False)
 wide=counts.pivot(index=['model','prompt_id','direction','k','denominator'],columns=['field','label'],values='count');wide.to_csv(out/'counts_wide.csv')
 paired=[];pairedmd=['# All12 matched model/prompt/sample sets','', 'Labels and excerpts below are verbatim supplied annotations. Each baseline appears once per matched set; repeated contrasts reuse it.','']
 for (m,p,s),group in frame.groupby(['model','prompt_id','sample_id']):
  pair=dict(model=m,prompt_id=p,sample_id=s,sample_seed=int(group.iloc[0]['sample_seed']));tab=[]
  for d,k in CONDITIONS:
   r=group[(group.direction==d)&(group.k==k)].iloc[0];prefix=r['condition']
   for field in ['review_id',*FIELDS]:pair[prefix+'__'+field]=r[field]
   tab.append({'condition':prefix,'stance':r.stance_label,'stance excerpt':r.stance_supporting_excerpt,'rotation':r.earth_rotation,'rotation excerpt':r.earth_rotation_supporting_excerpt,'revolution':r.earth_revolution_around_sun,'revolution excerpt':r.earth_revolution_supporting_excerpt,'style':r.style_label,'style excerpt':r.style_supporting_excerpt,'quality yes/unclear':'; '.join(q.replace('full_continuation_','')+'='+r[q] for q in QUALITY if r[q]!='no'),'notes':r.semantic_notes+' '+r.quality_notes})
  paired.append(pair);pairedmd += [f'## {m} / {p} / sample{s}', '',md(pd.DataFrame(tab)),'']
 pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False);(out/'paired_comparisons.md').write_text('\n'.join(pairedmd))
 # Long paired categories preserve every transition, with no ordering of stance.
 transitions=[];contrasts=[];align=[]
 for (m,p,s),g in frame.groupby(['model','prompt_id','sample_id']):
  b=g[g.direction=='baseline'].iloc[0]
  for d in ['own','random']:
   for mag in MAG:
    minus=g[(g.direction==d)&(g.k==-mag)].iloc[0];plus=g[(g.direction==d)&(g.k==mag)].iloc[0]
    for field in CATS:
     for r in [minus,plus]:transitions.append(dict(model=m,prompt_id=p,sample_id=s,direction=d,k=r.k,field=field,baseline_label=b[field],steered_label=r[field],changed=b[field]!=r[field]))
     contrasts.append(dict(model=m,prompt_id=p,sample_id=s,direction=d,magnitude=mag,field=field,negative_label=minus[field],positive_label=plus[field],changed=minus[field]!=plus[field]))
    for r in [minus,plus]:align.append(dict(model=m,prompt_id=p,sample_id=s,direction=d,k=r.k,baseline_stance=b.stance_label,stance=r.stance_label,baseline_style=b.style_label,style=r.style_label,style_changed=b.style_label!=r.style_label,stance_ambiguous=r.stance_label in ['unclear','mixed'],incoherence=r[QUALITY[1]],topic_drift=r[QUALITY[2]],baseline_incoherence=b[QUALITY[1]],baseline_topic_drift=b[QUALITY[2]]))
 trans=pd.DataFrame(transitions);con=pd.DataFrame(contrasts);trans.to_csv(out/'paired_baseline_transitions.csv',index=False);con.to_csv(out/'paired_sign_contrasts.csv',index=False);pd.DataFrame(align).to_csv(out/'paired_style_quality_context.csv',index=False)
 trans.groupby(['model','prompt_id','direction','k','field','baseline_label','steered_label']).size().rename('count').reset_index().assign(denominator=3).to_csv(out/'baseline_transition_counts.csv',index=False)
 con.groupby(['model','prompt_id','direction','magnitude','field','negative_label','positive_label']).size().rename('count').reset_index().assign(denominator=3).to_csv(out/'sign_contrast_counts.csv',index=False)
 # Transparent category counts, not ordinal stance scores or statistical tests.
 summary=[]
 for (m,p,d,mag),g in con.groupby(['model','prompt_id','direction','magnitude']):
  row=dict(model=m,prompt_id=p,direction=d,magnitude=mag,denominator=3)
  for field,target,title in [('stance_label','heliocentric','H'),('stance_label','geocentric','G'),('earth_rotation','affirmed','rotation_affirmed'),('earth_rotation','denied','rotation_denied'),('earth_revolution_around_sun','affirmed','revolution_affirmed'),('earth_revolution_around_sun','denied','revolution_denied')]:
   a=g[g.field==field];row[title+'_minus']=int((a.negative_label==target).sum());row[title+'_plus']=int((a.positive_label==target).sum());row[title+'_plus_minus_count_difference']=row[title+'_plus']-row[title+'_minus']
  summary.append(row)
 pd.DataFrame(summary).to_csv(out/'signed_category_summary.csv',index=False)
 make_plots(frame,counts,out)
 hashes={str(p):sha(p) for p in inputs+[pilot,Path(__file__)]};writej(out/'provenance.json',dict(input_sha256=hashes,analysis_only=True,labels_preserved=True,no_generation_or_fitting=True));shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(DATA/'rubric.md',out/'rubric.md')
 # Data-backed tables for scientific interpretation, followed by authored assessment.
 text=['# Reproduced-protocol stance sweep: supplied blinded annotations','',
 'Validation PASS:156 unique opaque IDs, exact original text and first80-word windows, complete valid rubric fields and exact2×2×3×13 coverage. Joined strictly by review_id. Supplied labels/excerpts/notes are unchanged. Own-positive adds the heliocentric-positive raw probe direction at saved L16; random signs have no semantic orientation. Both model directions exactly match coefficient/scaler artifacts. All sources remain untouched.','',
 f'Reviewer ID(s): {", ".join(frame.reviewer_id.unique())}. Independence cannot be established solely from a reviewer ID. No new judging was performed. The original rubric file documents that fields were initially blank; this analysis consumes the later supplied scoring. Quality accepts yes/no/unclear as explicitly stated in supplied quality notes.','',
 f'Annotation review flags: {len(flags)}; see annotation_review_flags.csv. Excerpt substring checks flag edited/composite/out-of-window excerpts without changing them. Truncation-versus-stop flags distinguish visible-text judgment from token-cap/EOS metadata and are not automatically annotation errors.','',
 '## Counts','', 'Each model/prompt/direction/signed-k group has n=3. Baseline is one set of3, reused across comparisons rather than treated as fresh evidence. Zero categories, mixed/unclear, conflicting and unmentioned are retained in counts_long.csv/counts_wide.csv.','']
 for field in CATS:
  tab=counts[counts.field==field].pivot(index=['model','prompt_id','direction','k'],columns='label',values='count').reindex(columns=CATS[field],fill_value=0).reset_index();text += [f'### {field} (counts /3)','',md(tab),'']
 text += ['## Paired evidence','', 'paired_comparisons.csv/MD contain all12 model/prompt/sample sets, all13 conditions and all verbatim supporting excerpts/notes. paired_baseline_transitions.csv and baseline_transition_counts.csv retain every categorical baseline→steered change separately at each signed strength. paired_sign_contrasts.csv/sign_contrast_counts.csv compare negative→positive within each magnitude. No ordered stance score is used. signed_category_summary.csv shows separate counts of heliocentric/geocentric and affirmed/denied Earth-motion claims.','',
 '## Figures','', 'category_profiles.png/pdf: stacked proportions for stance, Earth rotation, Earth revolution and style, with all categories and n=3 shown at every signed-k. Each model/prompt is separate, own/random side by side; the same baseline is drawn at0 for context, never new evidence. quality_profiles.png/pdf: yes and unclear flag counts/3 (no=3−yes−unclear) for every quality dimension. Lines join sampled evaluations only as visual guides; no inferential intervals or smoothing.','',
 '## Limits and reproduction','', 'Two response-selected development prompts, three paired decoding realizations, one random vector per model. Seeds are not independent prompts. No significance tests, confirmed winning strength, new judging, generation, fitting, extraction or tuning. Historical description is not automatically endorsement; unmentioned is not denied. Motion claims and stance remain separate. Cross-configuration nonquantized probe/NF4 generation compatibility remains unresolved.','',
 '```bash\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/analyze_reproduced_stance_strength_scoring.py --output-dir results/steering/reproduced_protocol_stance_strength_sweep/scored_analysis_20261030\n```','', 'The script refuses to overwrite a nonempty output directory. Use a fresh directory for reproduction. Input hashes and script snapshot are in provenance.json.']
 (out/'data_report.md').write_text('\n'.join(text)+'\n')
 write_assessment(frame,trans,counts,out)
 print(json.dumps(dict(valid=True,records=156,annotation_review_flags=len(flags),output=str(out)),indent=2))

def write_assessment(frame,trans,counts,out):
 # Category-preserving motion suppression summaries; no stance ordering.
 omission=[]
 for (m,p,d,k,field),g in trans[trans.field.isin(['earth_rotation','earth_revolution_around_sun'])].groupby(['model','prompt_id','direction','k','field']):
  affirmed=g[g.baseline_label=='affirmed']
  omission.append(dict(model=m,prompt_id=p,direction=d,k=k,field=field,total_pairs=3,baseline_affirmed=len(affirmed),affirmed_to_affirmed=int((affirmed.steered_label=='affirmed').sum()),affirmed_to_unmentioned=int((affirmed.steered_label=='unmentioned').sum()),affirmed_to_denied=int((affirmed.steered_label=='denied').sum()),affirmed_to_conflicting=int((affirmed.steered_label=='conflicting').sum())))
 pd.DataFrame(omission).to_csv(out/'motion_omission_vs_denial.csv',index=False)
 compact=[]
 for (m,p,d),g in frame.groupby(['model','prompt_id','direction']):
  if d=='baseline':continue
  row=dict(model=m,prompt=p,direction=d)
  b=frame[(frame.model==m)&(frame.prompt_id==p)&(frame.direction=='baseline')]
  row['baseline']= '/'.join(b.sort_values('sample_id').stance_label.map({'heliocentric':'H','geocentric':'G','mixed':'M','unclear':'U'}))
  for k in [-.5,-.3,-.1,.1,.3,.5]:
   a=g[g.k==k].sort_values('sample_id');row[f'{k:+g}']='/'.join(a.stance_label.map({'heliocentric':'H','geocentric':'G','mixed':'M','unclear':'U'}))
  compact.append(row)
 compact=pd.DataFrame(compact);compact.to_csv(out/'stance_pairs_compact.csv',index=False)
 contrasts=[]
 for (m,p,d),g in frame[frame.direction!='baseline'].groupby(['model','prompt_id','direction']):
  for mag in MAG:
   row=dict(model=m,prompt=p,direction=d,magnitude=mag)
   for sign,name in [(-1,'minus'),(1,'plus')]:
    a=g[g.k==sign*mag];row[name+'_H_G_U']='/'.join(str(int((a.stance_label==c).sum())) for c in ['heliocentric','geocentric','unclear'])
   contrasts.append(row)
 pd.DataFrame(contrasts).to_csv(out/'stance_sign_counts_compact.csv',index=False)
 q=[]
 for (d,mag),g in frame.assign(magnitude=frame.k.abs()).groupby(['direction','magnitude']):
  q.append(dict(direction=d,magnitude=mag,n=len(g),unclear=int((g.stance_label=='unclear').sum()),incoherence_yes=int((g.full_continuation_incoherence=='yes').sum()),topic_drift_yes=int((g.full_continuation_topic_drift=='yes').sum()),repetition_yes=int((g.full_continuation_repetition=='yes').sum())))
 q=pd.DataFrame(q);q.to_csv(out/'quality_ambiguity_overview.csv',index=False)
 report=[
 '# Reproduced-protocol stance strength sweep: descriptive assessment','',
 '**Recommendation: close this stance-steering pilot without further tuning.** There are identifiable prompt-specific suppression effects, but no sufficiently coherent bidirectional semantic stance effect to nominate a fixed intervention for reserved-prompt confirmation. This does not show that these directions cannot steer under any setup. No follow-up is executed.','',
 '## Validation and sign','',
 'PASS: all156 opaque IDs occur once and join by review_id; original prompt context, continuation and first80-word substring/word count match exactly. Every required rubric field is populated with valid categorical labels. Exact coverage: questions_012 (retrograde) and astro_024 (annual solar circuit) ×base/LoRA ×sample_ids0/1/2 ×13 conditions. All supplied labels, excerpts and notes are preserved in joined_annotations.jsonl/CSV. Zero mixed/conflicting counts are retained. Own directions match w=v/scaler_scale and the cached unit vectors, with geocentric=0/heliocentric=1; k>0 adds the positive-probe direction. Random signs have no stipulated semantic meaning. Positive probe sign alone is not behavioral evidence.','',
 'Two review flags concern supplied truncation=no on token-cap-stopped outputs (c806c8e7e001e4e7d317a3f7 and357ab4fe3d3e0d9c02d7303d). A continuation can reach its cap at an apparently complete sentence, so this is a metadata-versus-judgment distinction, not an automatic labeling error. All semantic excerpts are exact substrings of their windows; no duplicate text-window annotation conflicts were found. Reviewer independence cannot be verified from its ID alone.','',
 '## Paired stance labels','',
 'Entries list sample IDs0/1/2 in order: H=heliocentric, G=geocentric, U=unclear, M=mixed (none supplied). Baseline is reused, never counted as a new observation for each contrast. Every cell is three decoding realizations of the same prompt, not three independent prompts. Full supporting excerpts, Earth-motion fields, style and quality notes are in paired_comparisons.md/CSV.','',md(compact),'',
 '## What changes, and what does not','',
 '**Base / retrograde: a clear within-prompt suppression pattern at stronger negative strengths.** Baseline is H3/3. Own− H counts at magnitudes .1/.3/.5 are3/2/0, versus own+2/3/2. Thus the own positive-minus-negative H-count contrasts are−1,+1,+2 out of3, not a uniform signed effect across the tested range. Random− remains H3/3 at all magnitudes; random+ is3/2/3. At−.5, all three own outputs change H→U and affirmed Earth revolution→unmentioned; none becomes G and none denies revolution. Own+.5 also loses one baseline H/affirmed-revolution label, and own+.3 merely retains the ceiling baseline H3/3. This supports a direction-dependent loss of explicit heliocentric content on this prompt, not demonstrated geocentric endorsement or newly induced heliocentrism.','',
 'For sample2, own−.5 is labeled unclear with excerpt “in Ptolemy\'s geocentric model?”; its text describes what was considered in a historical model rather than endorsing it. We retain that evaluator judgment. The paired positive response is H with “planets to appear to move in a direction that is opposite to their respective orbits around the Sun”. Historical vocabulary is not recoded as stance. Base retrograde own−.5 remains modern3/3 and has zero supplied repetition/incoherence/topic-drift flags, so its suppression pattern is not explained away as gross quality collapse in these annotations.','',
 '**Base / annual solar circuit: inconsistent and highly ambiguous.** Baseline H1/U2. Own− H counts are1/0/0 at .1/.3/.5; own+1/0/1. At−.1 the H label moves from baseline sample0 to sample2, rather than preserving the same answer. At+.5 the single H likewise occurs on sample2, not baseline sample0. Random− gives1/0/0 and random+1/0/1: the aggregate H-count pattern matches own at every signed strength, although individual sample assignments can differ. No base annual response is labeled G. At random−.3 one revolution claim is affirmed while stance stays U; separate fields are intentionally not forced to agree.','',
 '**LoRA / annual solar circuit: omission of Earth rotation, not heliocentric conversion.** Baseline is G1/U2 with rotation affirmed3/3. Every own-negative magnitude yields rotation unmentioned3/3; no denial. Own-positive retains rotation on2/3,2/3,1/3 as magnitude increases; these are not gains relative to baseline. Random-negative retains2/3,3/3,2/3 and random-positive2/3,2/3,1/3. The negative own omission differs from the negative random control, but the positive trajectory declines and matches random-positive counts. Stance never becomes H. Own+.3 creates G on sample1 from baseline U—the opposite of the intended positive stance sign. No LoRA steered annual output affirms Earth-around-Sun revolution; baseline denial on sample2 disappears to unmentioned.','',
 '**LoRA / retrograde: no demonstrated positive conversion.** Baseline is G1/U2. Own− is G1/U2 only at−.1 and U3/3 at−.3/−.5; own+ is U3/3 at+.1, G1/U2 at+.3, and U3/3 at+.5. The new G at+.3 is sample0, while the baseline G is sample2. Random−/random+ retain the baseline G at .1 and become U3/3 at .3/.5. No H or affirmed rotation/revolution labels occur anywhere on this LoRA prompt. A retained G at own−.1 is not a newly induced G. The higher-strength responses mostly lose an assessable stance.','',
 '**Omission versus denial:** none of the72 own-steered outputs is labeled denied for either Earth-motion field. There are no newly created motion-denial labels relative to paired baselines anywhere in the grid; the random−.1 retrograde denial repeats the same LoRA baseline sample2 denial. Do not interpret unmentioned as denied, or rotation alone as heliocentrism. The evaluator includes an inferred revolution denial for the central-Earth LoRA annual baseline; we preserve that judgment without upgrading it to a literal verbal negation. motion_omission_vs_denial.csv gives the exact paired categories.','',
 '## Ambiguity, style and quality','',
 'Across unique outputs: H39, G7, U110, M0. The12 unique baselines are H4/G2/U6. All78 LoRA outputs lack an H label. None of the changes justifies an ordinal score for U or G/M/H. High ambiguity limits what can be learned from increasing strength.','',
 'The table below pools the two signs, two models and two prompts solely to describe workload-wide quality. Each nonbaseline row has24 outputs; the baseline row has12 unique outputs. It does not supply independent replications or a quality-adjusted semantic effect. Full prompt/model/signed-k counts and paired no→yes/yes→no transitions are saved separately.','',md(q),'',
 'Own |k|=.5 has21/24 unclear and12/24 incoherence flags, versus own |k|=.1 at16/24 and5/24. The rise is not uniform across cases: base retrograde own−.5 has no incoherence/drift flags, whereas LoRA retrograde own+.5 has topic drift3/3 and incoherence2/3; LoRA annual own−.5 has incoherence3/3. Relative to their own baselines, own+.5 creates4 new incoherence yes-flags out of12 pairs and own−.5 creates4/12, while each also resolves one baseline yes-flag. Those are separate paired comparisons reusing the same12 baselines, not24 new baseline observations. Random controls also change quality: at |k|=.5 topic drift is9/24, versus own8/24. No repetition is flagged anywhere (0/156); do not infer repetition merely from topic drift. Truncation labels are yes148/no4/unclear4.','',
 'Style is stable modern for all39 base retrograde outputs; stable premodern for all39 LoRA annual outputs. Base annual style varies in both own and random conditions. LoRA retrograde own+.3 has mixed1/premodern2, own+.5 modern1/premodern2, and random−.1 mixed1/premodern2. These isolated style shifts occur without any H conversion. The strongest base retrograde suppression is therefore not contingent on a premodern style label; elsewhere ambiguity, style changes and quality costs co-occur. Supplied quality labels are provisional observations and never used to discard or relabel a semantic outcome.','',
 '## Decision and files','',
 'There is localized evidence that the own direction changes whether certain Earth-motion claims are expressed: base retrograde heliocentric/revolution content and LoRA annual rotation. The negative effects are omission, not a replicated switch to geocentric stance. Positive additions do not reproducibly create heliocentric answers, effects do not generalize across these two prompts/models, and higher magnitudes frequently lose interpretability. The random vector also changes labels, style and quality. A single random vector per model is a limited control, not a null distribution.','',
 '**Close this pilot without selecting a winning strength or further tuning.** Preserve the prompt-specific suppression observations as exploratory results, but do not nominate a fixed bidirectional stance intervention for reserved prompts from this evidence. This is absence of a demonstrated general signed stance effect, not proof of causal irrelevance. These response-selected development prompts and three paired samples do not support population significance tests. Neither reserved-prompt confirmation nor any other follow-up was run.','',
 'Figures: category_profiles.png/PDF and quality_profiles.png/PDF. All category counts/denominators: counts_long.csv, counts_wide.csv and data_report.md. Paired evidence: paired_comparisons.csv/MD. Separate baseline and sign contrasts: paired_baseline_transitions.csv, baseline_transition_counts.csv, paired_sign_contrasts.csv, sign_contrast_counts.csv, signed_category_summary.csv. motion_omission_vs_denial.csv and paired_style_quality_context.csv retain omission and co-occurrence distinctions. validation_results.json, annotation_review_flags.csv, provenance.json and the script snapshot document checks. No original inputs changed.','',
 'Reproduce into a fresh directory with `/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/analyze_reproduced_stance_strength_scoring.py --output-dir <new-analysis-directory>`. This is CPU-only analysis, with no model imports, generation, extraction, fitting, new judging or sweeps.'
 ]
 (out/'report.md').write_text('\n'.join(report)+'\n')

def make_plots(frame,counts,out):
 colors={'geocentric':'#b34a35','heliocentric':'#277da8','mixed':'#a77cb5','unclear':'#cccccc','affirmed':'#308b6c','denied':'#bd563e','conflicting':'#ba93c7','unmentioned':'#d0d0d0','modern':'#277da8','premodern':'#bd8540'}
 fields=['stance_label','earth_rotation','earth_revolution_around_sun','style_label']
 xvals=[-.5,-.3,-.1,0,.1,.3,.5]
 fig,axes=plt.subplots(4,8,figsize=(24,11),sharey=True)
 for col,(m,p,d) in enumerate(itertools.product(MODELS,PROMPTS,['own','random'])):
  for ri,field in enumerate(fields):
   ax=axes[ri,col];bottom=np.zeros(7)
   for label in CATS[field]:
    vals=[int(counts[(counts.model==m)&(counts.prompt_id==p)&(counts.direction==('baseline' if k==0 else d))&(counts.k==k)&(counts.field==field)&(counts.label==label)]['count'].iloc[0]) for k in xvals]
    ax.bar(range(7),np.array(vals)/3,bottom=bottom,color=colors[label],label=label,width=.8);bottom+=np.array(vals)/3
   ax.set_xticks(range(7),[str(x) for x in xvals],rotation=90,fontsize=7);ax.set_ylim(0,1.02);ax.set_yticks([0,1/3,2/3,1],['0/3','1/3','2/3','3/3']);ax.grid(axis='y',alpha=.15)
   if ri==0:ax.set_title(f'{m} | {"retrograde" if p=="questions_012" else "annual Sun"}\n{d}; n=3 per k',fontsize=9)
   if col==0:ax.set_ylabel(field.replace('_label','').replace('_',' '),fontsize=10)
   if col==7:ax.legend(loc='upper left',bbox_to_anchor=(1.03,1),fontsize=8)
   if ri==3:ax.set_xlabel('signed k (0 = shared baseline)',fontsize=8)
 fig.suptitle('Supplied categorical labels: proportions, no ordinal stance score\nBaseline displayed in both direction panels for context only; mixed/unclear retained',fontsize=12)
 fig.tight_layout(rect=(0,0,.97,.94));fig.savefig(out/'category_profiles.png',dpi=180);fig.savefig(out/'category_profiles.pdf');plt.close(fig)
 fig,axes=plt.subplots(4,4,figsize=(15,10),sharey=True)
 for col,(m,p) in enumerate(itertools.product(MODELS,PROMPTS)):
  for ri,field in enumerate(QUALITY):
   ax=axes[ri,col]
   for d,color in [('own','#277da8'),('random','#bd563e')]:
    for label,style,marker in [('yes','-','o'),('unclear',':','x')]:
     vals=[int(counts[(counts.model==m)&(counts.prompt_id==p)&(counts.direction==('baseline' if k==0 else d))&(counts.k==k)&(counts.field==field)&(counts.label==label)]['count'].iloc[0]) for k in xvals]
     ax.plot(xvals,vals,style,color=color,marker=marker,label=f'{d}: {label}',linewidth=1,markersize=4)
   # no-counts remain available in full stacked category count CSV.
   ax.set_xticks(xvals,[str(x) for x in xvals],rotation=60);ax.set_yticks([0,1,2,3],['0/3','1/3','2/3','3/3']);ax.grid(alpha=.2)
   if ri==0:ax.set_title(f'{m} | {"retrograde" if p=="questions_012" else "annual Sun"}; n=3')
   if col==0:ax.set_ylabel(field.replace('full_continuation_','').replace('_',' '))
   if col==3 and ri==0:ax.legend(fontsize=7)
   if ri==3:ax.set_xlabel('signed k (0 = shared baseline)')
 fig.suptitle('Full-continuation quality: yes and unclear counts /3\nNo = 3 − yes − unclear for each flag; connecting lines are visual guides only')
 fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'quality_profiles.png',dpi=180);fig.savefig(out/'quality_profiles.pdf');plt.close(fig)

if __name__=='__main__':main()
