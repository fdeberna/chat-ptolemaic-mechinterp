"""Validate supplied annotations, preserve labels, and describe paired categories."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/steering/sampled_own_random_k01'
OUT=DATA/'stance_subset_annotation_analysis'
CONDITIONS=['baseline','own_minus','own_plus','random_minus','random_plus']
STANCE=['geocentric','heliocentric','mixed','unclear']
STYLE=['modern','premodern','mixed','unclear']
EARTH=['affirmed','denied','conflicting','unmentioned']
QUALITY=['full_continuation_repetition','full_continuation_incoherence','full_continuation_topic_drift','full_continuation_truncation']
FIELDS=['stance_label','stance_supporting_excerpt','earth_rotation','earth_rotation_supporting_excerpt','earth_revolution_around_sun','earth_revolution_supporting_excerpt','style_label','style_supporting_excerpt','semantic_notes',*QUALITY,'quality_notes','reviewer_id']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def condition(key):
    return 'baseline' if key['kind']=='baseline' else key['direction_source']+('_minus' if float(key['k'])<0 else '_plus')


def markdown(frame):
    def cell(x):return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report():
    f=pd.read_csv(OUT/'joined_annotations.csv',keep_default_na=False)
    stance=pd.read_csv(OUT/'stance_counts.csv');quality=pd.read_csv(OUT/'ambiguity_quality_counts.csv')
    labels=f.pivot(index=['model','prompt_id','seed'],columns='condition',values='stance_label').reindex(columns=CONDITIONS).reset_index()
    count_table=stance.rename(columns={'prompt_id':'prompt','geocentric':'G','heliocentric':'H','mixed':'M','unclear':'U'}).drop(columns='total')
    qtable=quality[['model','prompt_id','condition',*[(q+'_yes') for q in QUALITY]]].rename(columns={'prompt_id':'prompt',**{q+'_yes':label for q,label in zip(QUALITY,['repetition','incoherence','topic drift','truncation'])}})
    secondary=f.groupby(['model','condition']).style_label.value_counts().unstack(fill_value=0).reindex(columns=STYLE,fill_value=0).reset_index()
    compact_labels=labels.replace({'geocentric':'G','heliocentric':'H','mixed':'M','unclear':'U'}).rename(columns={'prompt_id':'prompt'})
    text=[
      '# Supplied blinded stance annotations: descriptive paired analysis','',
      '**Conclusion: no consistent signed stance effect is demonstrated across this pilot.** LoRA annual_sun shows a localized own+ heliocentric pattern; own− does not induce new geocentric labels relative to baseline. Retrograde is mostly ambiguous, and base retrograde own/random label patterns are identical. This is not proof that the direction cannot steer.',
      '## Validation and sign provenance','',
      'PASS: exactly 60 unique scored opaque IDs match the unscored subset and condition key, with all rubric fields populated and allowed labels intact. Exact coverage: two prompts × two models × three seeds × five conditions. Prompt context, 80-word window and full continuation match the original blinded records and saved generation artifacts. No records were missing, duplicated, excluded or relabeled.',
      'Positive-class metadata is geocentric=0, heliocentric=1 for both source probes. Recomputed w=v/scaler_scale and u=w/||w|| exactly match the saved own directions. Archived hook/runner and source-vector hashes match run provenance; each steered row is stance L16, gamma=k×73.83589057971676, so k=±0.1 gives gamma=±7.383589057971676. The hook adds gamma*u. Own+ is therefore the heliocentric-positive **probe** sign; this does not itself establish a behavioral effect. Random direction signs are geometrically defined, without an expected semantic orientation.',
      'Annotations are used exactly as supplied, including supporting excerpts, notes, mixed and unclear. Reviewer ID: '+', '.join(f.reviewer_id.unique())+'. This analysis performs no additional judging and does not infer evaluator independence merely from the file name. Labels refer to the first substantive answer in the frozen first-80-word window; full-continuation quality flags are separate.',
      '## 1. Stance counts','',
      'Each row contains three seeds for one model/prompt/condition. G=geocentric, H=heliocentric, M=mixed, U=unclear. Zero mixed counts are retained.',
      markdown(count_table),
      '## 2. Paired labels and verbatim evidence','',
      markdown(compact_labels),
      'The complete paired table, including separate Earth-rotation/revolution labels, all corresponding supporting excerpts, style labels/excerpts and quality notes, is saved in [paired_comparison.csv](paired_comparison.csv) and [paired_comparison.md](paired_comparison.md). These are the same 12 model/prompt/seed pairs, not 12 independent prompts. Rotation labels overall: affirmed2, unmentioned58; revolution-around-Sun labels: affirmed28, denied1, unmentioned31. These fields are reported separately, without inferring heliocentrism from Earth rotation.',
      '## 3. Directional interpretation','',
      '- **LoRA / annual_sun:** own+ changes both baseline geocentric seeds0 and1 to heliocentric; seed2 remains heliocentric. Thus own+ is H3/3, compared with baseline H1/G2. Own− gives H1/G1/U1: seed1 retains its existing G label, seed0 becomes unclear, seed2 remains H. It creates no new G label. Comparing own− to own+ gives one G→H pair and one U→H pair, not three independent prompt replications.',
      '- The corresponding random+ labels remain at baseline H1/G2, while random− gives H2/U1, including G→H for seed1 at the negative sign. The own+ pattern is more aligned with the intended positive sign **on this one LoRA prompt**, but the random control also changes stance and the contrast is not reproduced across prompts.',
      '- **Base / annual_sun:** baseline and own+ are already H3/3, a ceiling for observing further positive changes. Own− changes seed0 from H to U, not G. Both random signs remain H3/3.',
      '- **Base / retrograde:** own and random have exactly the same supplied stance labels for each seed/sign. Plus is H1/U2 and minus H2/U1, opposite to the expected positive-versus-negative H count contrast. Seed0 loses its H label at plus; seed1 becomes H at minus; seed2 becomes H at plus. There is no own-direction specificity here.',
      '- **LoRA / retrograde:** all15 outputs are unclear; no signed stance effect can be assessed from those labels. Across all12 matched seed cases, own− never creates a new geocentric label relative to baseline. Counts/paired categorical transitions are saved in directional_contrasts.csv and paired_stance_transitions.csv; no ordered stance score is constructed.',
      'For example, LoRA annual_sun seed1 baseline is labeled G with the supplied excerpt “the motion of the sphere in which the stars are carried, which revolves around the axis of the world from east to west”; own+ is labeled H with “the motion of the earth in its orbit around the sun”. Own− retains G on that seed, while random− is also H. These illustrate both the localized positive result and the control limitation; they are not new judgments.',
      '## 4. Ambiguity, quality and secondary style check','',
      'Overall stance labels: H29, G5, M0, U26. Annual_sun has 3/30 unclear; retrograde has 23/30 unclear. All15 LoRA retrograde rows are unclear. Ambiguity is therefore concentrated rather than uniformly making the entire subset uninformative. Unclear/mixed rows are retained and never treated as an intermediate stance.',
      'Evaluator quality flags below are yes-counts out of three. These are full-continuation observations, not exclusions or corrections to semantic labels.',
      markdown(qtable),
      'Overall evaluator yes-flags: repetition30/60, incoherence24/60, topic drift14/60, truncation57/60. Truncation is the evaluator’s apparently-unfinished-text judgment, not automatically the token-cap metadata; both are retained separately in joined_annotations.csv and ambiguity_quality_counts.csv. Paired no→yes and yes→no changes are saved in paired_quality_changes.csv, so baseline defects are not automatically assigned to steering.',
      'Examples from the supplied flags: base annual_sun own− has topic drift3/3 versus baseline2/3; base retrograde own− has incoherence2/3 versus baseline1/3. LoRA annual_sun own+ reduces repetition from3/3 to1/3 but retains incoherence2/3. Those quality differences neither establish nor erase the supplied stance labels.',
      'Secondary style counts below pool the two prompts descriptively (six outputs per model/condition); full prompt-specific counts and seed-paired transitions are saved separately.',
      markdown(secondary),
      'All30 base style labels are modern and unchanged. LoRA annual_sun own+ has modern2/mixed1 versus baseline premodern1/mixed1/unclear1, with two style-label changes: seed1 premodern→modern and seed2 unclear→modern; seed0 stays mixed while stance changes G→H. Every annual_sun steering condition changes style on2/3 seeds, including both random signs. LoRA retrograde own− changes style2/3 times and own+1/3 times, despite all stance labels remaining unclear. Style co-variation and random changes limit any claim of a clean stance-only response.',
      '## 5. Scope of the conclusion','',
      '**No clear consistent signed effect across models/prompts; one prompt-specific LoRA own+ signal, with substantial retrograde ambiguity.** Own+ favors the evaluator’s heliocentric label for LoRA annual_sun, but not as a replicated cross-prompt pattern. Own− does not demonstrate induced geocentric responses relative to baseline. The base retrograde control reproduces the own-direction label changes, and the LoRA retrograde responses are uninformative about stance.',
      'Only two development prompts and three seeds per prompt were used. Seeds are paired decoding realizations, not independent prompts; no significance tests, generalization claims or causal percentages are supplied. Absence of demonstrated steering is not proof of causal irrelevance or inability to steer under another authorized setup. This analysis ran no generation, fitting, strength tuning, judging or NF4 compatibility extraction.',
      '## Files and reproduction','',
      'joined_annotations.jsonl/CSV preserve all supplied annotation values joined to provenance. stance_counts.csv, ambiguity_quality_counts.csv and style_counts.csv contain complete condition/prompt counts; paired_comparison.csv/MD contains all12 pairs with verbatim evidence; paired_stance_transitions.csv, paired_style_transitions.csv, paired_quality_changes.csv and directional_contrasts.csv retain categorical comparisons. validation_report.json/provenance.json record checks and input hashes. Source annotations and prior outputs were not modified.',
      '```bash\n/home/fdebe/miniconda3/envs/copernicus2/bin/python -B scripts/analyze_scored_stance_subset.py\n```',
      'The script refuses to overwrite a nonempty output directory. Use a fresh output location in the script for a new analysis; no inference is performed.'
    ]
    (OUT/'report.md').write_text('\n\n'.join(text).replace('\n\n\n\n','\n\n')+'\n')
    provenance=json.loads((OUT/'provenance.json').read_text());provenance['script_sha256']=sha(Path(__file__));(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));shutil.copy2(__file__,OUT/Path(__file__).name)


def main():
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):raise RuntimeError('Refusing to overwrite existing annotation-analysis files')
    paths=[DATA/'blinded_stance_subset_scored.jsonl',DATA/'blinded_stance_subset.jsonl',DATA/'condition_key.csv',DATA/'config.json',DATA/'direction_audit.json',DATA/'rubric.md']
    issues=[]
    def check(ok,message):
        if not ok:issues.append(message)
    def objects(p):
        result=[]
        for line,text in enumerate(p.read_text().splitlines(),1):
            if not text.strip():continue
            try:
                def pairs(items):
                    obj={}
                    for key,value in items:
                        if key in obj:raise ValueError(f'duplicate JSON key {key}')
                        obj[key]=value
                    return obj
                obj=json.loads(text,object_pairs_hook=pairs)
                if not isinstance(obj,dict):raise ValueError('record is not an object')
                result.append(obj)
            except (ValueError,TypeError) as error:issues.append(f'{p.name}:{line}: {error}')
        return result
    def stop_if_invalid():
        if issues:
            (OUT/'validation_report.json').write_text(json.dumps(dict(valid=False,issues=issues),indent=2))
            print('VALIDATION FAILED; stopped without joined analysis:\n'+'\n'.join(issues));sys.exit(1)
    for p in paths:check(p.is_file(),f'Missing file: {p}')
    stop_if_invalid()
    scored=objects(paths[0]);original=objects(paths[1])
    with paths[2].open(newline='') as f:keys=list(csv.DictReader(f))
    for name,rows in [('scored',scored),('unscored subset',original),('condition key',keys)]:
        ids=[r.get('review_id') for r in rows];check(all(isinstance(i,str) and i for i in ids),f'{name}: missing/invalid opaque ID')
        check(len(ids)==len(set(ids)),f'{name}: duplicate opaque IDs')
    check(len(scored)==60,'Scored records must number exactly 60')
    check(len(original)==60,'Unscored subset must number exactly 60')
    check(len({k['condition_id'] for k in keys})==len(keys),'Condition key duplicates condition IDs')
    stop_if_invalid()
    selected=[k for k in keys if k['prompt_id'] in ['annual_sun','retrograde'] and (k['kind']=='baseline' or (k['target']=='stance' and k['direction_source'] in ['own','random'] and float(k['k']) in [-.1,.1]))]
    check(len(selected)==60,'Filtered condition key must contain exactly 60 records')
    expected={(m,p,s,c) for m in ['base','lora'] for p in ['annual_sun','retrograde'] for s in [0,1,2] for c in CONDITIONS}
    actual=[(k['model'],k['prompt_id'],int(k['seed']),condition(k)) for k in selected]
    check(set(actual)==expected and len(actual)==len(set(actual)),'Incorrect or duplicate model/prompt/seed/condition coverage')
    check({r['review_id'] for r in scored}=={r['review_id'] for r in original}=={k['review_id'] for k in selected},'Missing/extra/mismatched scored/unscored/key opaque IDs')
    stop_if_invalid()
    unscored={r['review_id']:r for r in original};keymap={k['review_id']:k for k in selected};cfg=json.loads(paths[3].read_text());audit=json.loads(paths[4].read_text())
    data_rows=[]
    for a in scored:
        rid=a['review_id'];base=unscored[rid];key=keymap[rid]
        for field in FIELDS:check(field in a and isinstance(a[field],str) and bool(a[field].strip()),f'{rid}: missing/empty rubric field {field}')
        for field in ['prompt_context','scoring_window_first_80_words','scored_word_count','full_continuation']:check(a.get(field)==base.get(field),f'{rid}: altered/mismatched {field}')
        for field,allowed in [('stance_label',STANCE),('style_label',STYLE),('earth_rotation',EARTH),('earth_revolution_around_sun',EARTH)]+[(q,['yes','no']) for q in QUALITY]:check(a.get(field) in allowed,f'{rid}: invalid {field}: {a.get(field)!r}')
        path=DATA/('reused_baselines' if key['kind']=='baseline' else 'conditions')/f'{key["condition_id"]}.json'
        check(path.is_file(),f'{rid}: missing generation artifact {path}')
        if not path.is_file():continue
        r=json.loads(path.read_text())
        for field in ['condition_id','model','prompt_id']:check(r[field]==key[field],f'{rid}: generation/key mismatch {field}')
        check(r['seed']==int(key['seed']) and r['k']==float(key['k']),f'{rid}: seed/strength mismatch')
        check(r['prompt_text']==a['prompt_context'] and r['continuation']==a['full_continuation'],f'{rid}: generation/annotation text mismatch')
        if key['kind']=='baseline':check(float(key['k'])==0 and float(key['gamma'])==0 and r['target'] is None,f'{rid}: baseline unexpectedly steered')
        else:
            check(r['target']=='stance' and r['direction_source']==key['direction_source'] and r['layer']==int(float(key['layer']))==16,f'{rid}: wrong direction/target/layer')
            check(abs(r['gamma']-float(key['gamma']))<1e-12 and abs(r['gamma']-r['k']*cfg['scales']['stance'])<1e-12,f'{rid}: gamma sign/scale mismatch')
        data_rows.append({**a,**key,'condition':condition(key),'seed':int(key['seed'])})
    # Verify own direction orientation using saved coefficients, cached unit vectors,
    # original source hashes, and the archived hook/runner used for generation.
    sign_checks=[];pilot=ROOT/'results/steering/pilot_L16_stance_L6_style'
    check(sha(DATA/'reconstruction/steering.py')==cfg['sources'][str(ROOT/'src/ptolemaic_mechinterp/interventions/steering.py')],'Archived hook hash mismatch')
    check(sha(DATA/'reconstruction/run_sampled_steering.py')==cfg['sources'][str(ROOT/'scripts/run_sampled_steering.py')],'Archived runner hash mismatch')
    check(sha(pilot/'directions.npz')==cfg['sources'][str(pilot/'directions.npz')],'Saved unit-vector artifact hash mismatch')
    with np.load(pilot/'directions.npz') as saved:
        for model in ['base','lora']:
            info=next(x for x in audit if x['model']==model and x['target']=='stance');p=Path(info['probe_path'])
            check(sha(p)==info['probe_sha256'],f'{model}: source probe artifact hash mismatch')
            with np.load(p) as probe:
                mapping=json.loads(str(probe['label_mapping']));w=(probe['coef']/probe['scaler_scale']).reshape(-1);u=w/np.linalg.norm(w)
                check(mapping=={'geocentric':0,'heliocentric':1} and str(probe['positive_class'])=='heliocentric',f'{model}: unexpected positive-class mapping')
                check(np.isfinite(w).all() and np.linalg.norm(w)>0 and np.array_equal(u,saved[f'stance__{model}']),f'{model}: own direction is not saved positive raw unit vector')
                sign_checks.append(dict(model=model,positive_class='heliocentric',mapping=mapping,raw_norm=float(np.linalg.norm(w)),unit_vector_matches=True,positive_k_increases_positive_probe_logit=True,semantic_effect_not_implied=True))
    for m in ['base','lora']:
        for p in ['annual_sun','retrograde']:
            for s in [0,1,2]:
                group=[k for k in selected if k['model']==m and k['prompt_id']==p and int(k['seed'])==s];b=next(k for k in group if k['kind']=='baseline')
                check(all(k['baseline_condition_id']==b['condition_id'] for k in group),f'{m}/{p}/{s}: wrong paired baseline ID')
    stop_if_invalid()
    validation=dict(valid=True,records=60,coverage=dict(models=['base','lora'],prompts=['annual_sun','retrograde'],seeds=[0,1,2],conditions=CONDITIONS),rubric_fields_complete=True,labels_preserved=True,text_matches_unscored_and_generation_artifacts=True,direction_sign_checks=sign_checks)
    (OUT/'validation_report.json').write_text(json.dumps(validation,indent=2))
    f=pd.DataFrame(data_rows)
    # Preserve every supplied annotation value, including notes and excerpts.
    with (OUT/'joined_annotations.jsonl').open('w') as handle:
        for row in data_rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    f.to_csv(OUT/'joined_annotations.csv',index=False)
    groups=['model','prompt_id','condition']
    stance=f.groupby(groups).stance_label.value_counts().unstack(fill_value=0).reindex(columns=STANCE,fill_value=0).reset_index();stance['total']=stance[STANCE].sum(axis=1);stance.to_csv(OUT/'stance_counts.csv',index=False)
    style=f.groupby(groups).style_label.value_counts().unstack(fill_value=0).reindex(columns=STYLE,fill_value=0).reset_index();style['total']=style[STYLE].sum(axis=1);style.to_csv(OUT/'style_counts.csv',index=False)
    quality=[];quality_changes=[];styles=[];transitions=[];contrasts=[];pairs=[]
    pair_fields=['review_id','stance_label','stance_supporting_excerpt','earth_rotation','earth_rotation_supporting_excerpt','earth_revolution_around_sun','earth_revolution_supporting_excerpt','style_label','style_supporting_excerpt','semantic_notes',*QUALITY,'quality_notes']
    for (model,prompt,seed),g in f.groupby(['model','prompt_id','seed'],sort=True):
        by=g.set_index('condition');wide=dict(model=model,prompt_id=prompt,seed=int(seed))
        for c in CONDITIONS:
            for field in pair_fields:wide[f'{c}__{field}']=by.loc[c,field]
        pairs.append(wide)
    paired=pd.DataFrame(pairs);paired.to_csv(OUT/'paired_comparison.csv',index=False)
    for (m,p,c),g in f.groupby(groups,sort=True):
        quality.append(dict(model=m,prompt_id=p,condition=c,total=len(g),mixed=int((g.stance_label=='mixed').sum()),unclear=int((g.stance_label=='unclear').sum()),**{q+'_yes':int((g[q]=='yes').sum()) for q in QUALITY},max_new_tokens_stops=int((g.stopping_reason=='max_new_tokens').sum())))
    for (m,p),g in f.groupby(['model','prompt_id'],sort=True):
        by={c:g[g.condition==c].set_index('seed') for c in CONDITIONS};b=by['baseline']
        for c in CONDITIONS[1:]:
            for seed in [0,1,2]:
                transitions.append(dict(model=m,prompt_id=p,seed=seed,condition=c,baseline_stance=b.loc[seed,'stance_label'],steered_stance=by[c].loc[seed,'stance_label']))
                styles.append(dict(model=m,prompt_id=p,seed=seed,condition=c,baseline_style=b.loc[seed,'style_label'],steered_style=by[c].loc[seed,'style_label']))
            quality_changes.append(dict(model=m,prompt_id=p,condition=c,**{q+'_no_to_yes':sum(b.loc[s,q]=='no' and by[c].loc[s,q]=='yes' for s in [0,1,2]) for q in QUALITY},**{q+'_yes_to_no':sum(b.loc[s,q]=='yes' and by[c].loc[s,q]=='no' for s in [0,1,2]) for q in QUALITY}))
        for d in ['own','random']:
            minus,plus=by[d+'_minus'],by[d+'_plus']
            contrasts.append(dict(model=m,prompt_id=p,direction=d,n_seed_pairs=3,baseline_heliocentric=int((b.stance_label=='heliocentric').sum()),baseline_geocentric=int((b.stance_label=='geocentric').sum()),plus_heliocentric=int((plus.stance_label=='heliocentric').sum()),minus_heliocentric=int((minus.stance_label=='heliocentric').sum()),minus_geocentric=int((minus.stance_label=='geocentric').sum()),plus_geocentric=int((plus.stance_label=='geocentric').sum()),minus_G_to_plus_H=sum(minus.loc[s,'stance_label']=='geocentric' and plus.loc[s,'stance_label']=='heliocentric' for s in [0,1,2]),minus_H_to_plus_G=sum(minus.loc[s,'stance_label']=='heliocentric' and plus.loc[s,'stance_label']=='geocentric' for s in [0,1,2]),baseline_not_H_to_plus_H=sum(b.loc[s,'stance_label']!='heliocentric' and plus.loc[s,'stance_label']=='heliocentric' for s in [0,1,2]),baseline_not_G_to_minus_G=sum(b.loc[s,'stance_label']!='geocentric' and minus.loc[s,'stance_label']=='geocentric' for s in [0,1,2])))
    pd.DataFrame(quality).to_csv(OUT/'ambiguity_quality_counts.csv',index=False)
    pd.DataFrame(quality_changes).to_csv(OUT/'paired_quality_changes.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'paired_stance_transitions.csv',index=False)
    pd.DataFrame(styles).to_csv(OUT/'paired_style_transitions.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'directional_contrasts.csv',index=False)
    blocks=['# Paired annotations with verbatim supporting excerpts','', 'All labels/excerpts below are supplied evaluator annotations, unchanged. Each block is one model × prompt × seed.','']
    for (m,p,s),g in f.groupby(['model','prompt_id','seed'],sort=True):
        b=g.set_index('condition').loc[CONDITIONS].reset_index()
        blocks += [f'## {m} / {p} / seed {s}','',markdown(b[['condition','stance_label','earth_rotation','earth_revolution_around_sun','style_label']]),'',markdown(b[['condition','stance_supporting_excerpt','earth_rotation_supporting_excerpt','earth_revolution_supporting_excerpt','style_supporting_excerpt']]),'']
    (OUT/'paired_comparison.md').write_text('\n'.join(blocks))
    provenance=dict(source_sha256={str(p):sha(p) for p in paths},script_sha256=sha(Path(__file__)),reviewer_ids=f.reviewer_id.unique().tolist(),label_changes=0,tests='None; descriptive counts and categorical paired transitions only. Seeds are not independent prompts.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));shutil.copy2(__file__,OUT/Path(__file__).name)
    write_report()
    print('VALIDATED 60 records, exact coverage, complete fields, unchanged texts/labels, and direction signs.')
    print(stance.to_string(index=False));print('\nPAIRED LABELS\n'+f.pivot(index=['model','prompt_id','seed'],columns='condition',values='stance_label').reindex(columns=CONDITIONS).to_string())
    print('\nQUALITY\n'+pd.DataFrame(quality).to_string(index=False));print('\nCONTRASTS\n'+pd.DataFrame(contrasts).to_string(index=False));print('\nSTYLE\n'+style.to_string(index=False))


if __name__=='__main__':main()
