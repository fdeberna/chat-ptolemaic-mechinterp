"""No-reset Euler sensitivity along sampled piecewise-linear activations (CPU only)."""
from __future__ import annotations

import os
import tempfile
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["MPLBACKEND"]="Agg"
os.environ.setdefault("MPLCONFIGDIR",tempfile.mkdtemp(prefix="full-sensitivity-mpl-"))

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info,threadpool_limits

from decompose_local_probe_sensitivity import forcing,norm,ratio
from ptolemaic_mechinterp.probes.dataset import encode_binary_target
from validate_logistic_probe_sensitivity import (
    NORM_EPS,EXPECTED_LABEL_MAPPINGS,analytical_beta_dot,fit_fixed_scaler_probe,
    gradient_relative_residual,sample_loss_weights,strength_dir,
)

GRID=(0.,.01,.02,.04,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)
REPRESENTATIVE=(6,16,28)
METRICS=("r_norm","raw_weight_gain","perpendicular_fraction","weight_norm",
         "inverse_weight_norm","angular_speed")


def atomic_json(path,value):
    def clean(item):
        if isinstance(item,dict): return {k:clean(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)): return [clean(v) for v in item]
        if isinstance(item,float) and not np.isfinite(item): return None
        return item
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False));temp.replace(path)


def load_layer(root,layer):
    data={};reference=None
    for t in GRID:
        path=strength_dir(root,t)
        manifest=json.loads((path/'manifest.json').read_text())
        md=pd.read_csv(path/manifest['metadata_file'])
        mask=md.layer_index.to_numpy()==layer;selected=md.loc[mask].reset_index(drop=True)
        keys=selected[['prompt_id','stance','style']].astype(str).to_numpy()
        if reference is None: reference=keys
        else: np.testing.assert_array_equal(keys,reference)
        if len(selected)!=120: raise ValueError('Expected 120 aligned prompts')
        with np.load(path/manifest['activation_file']) as a:
            full=a['activations'];x=np.asarray(full[mask],dtype=np.float64)
        del full
        if not np.isfinite(x).all(): raise ValueError('Nonfinite activations')
        data[t]=(x,selected)
    return data


def state(beta,scaler):
    w=beta[:-1]/scaler.scale_;wn=norm(w)
    u=w/wn if wn>NORM_EPS else np.full_like(w,np.nan)
    return w,u,wn


def evaluate(beta,z,d,y,weights,scaler):
    r=forcing(z,beta,d,y,weights)
    bd,solve=analytical_beta_dot(z0=z,y01=y,beta0=beta,dz_dt=d,sample_weights=weights,
                               cg_rtol=1e-9,cg_atol=1e-11,cg_maxiter=2000)
    w,u,wn=state(beta,scaler);wd=bd[:-1]/scaler.scale_
    perp=wd-u*float(u@wd);ud=perp/wn if wn>NORM_EPS else np.full_like(wd,np.nan)
    rn,wdn,pn,un=norm(r),norm(wd),norm(perp),norm(ud)
    product=rn*ratio(wdn,rn)*ratio(pn,wdn)*ratio(1.,wn)
    scalars=dict(D_rows=d.shape[0],D_columns=d.shape[1],D_frobenius_norm=norm(d),
        D_rms=norm(d)/np.sqrt(d.size),r_norm=rn,standardized_beta_gain=ratio(norm(bd),rn),
        raw_weight_gain=ratio(wdn,rn),perpendicular_fraction=ratio(pn,wdn),weight_norm=wn,
        inverse_weight_norm=ratio(1.,wn),angular_gain=ratio(un,rn),angular_speed=un,
        identity_product=product,identity_relative_error=ratio(abs(un-product),un),
        r_near_zero=rn<=NORM_EPS,w_dot_near_zero=wdn<=NORM_EPS,w_near_zero=wn<=NORM_EPS,
        u_dot_near_zero=un<=NORM_EPS,**solve)
    return scalars,dict(beta=beta.copy(),r=r,beta_dot=bd,raw_w_dot=wd,u_dot=ud)


def reference_probes(out,signature,layer,target,z,y,weights,scaler,metadata):
    path=out/'references'/f'{target}_layer_{layer:03d}.npz'
    if path.exists():
        with np.load(path) as a:
            meta=json.loads(str(a['metadata_json']))
            if meta['signature']==signature:
                return a['beta'].copy(),a['gradient_relative_residual'].copy()
        raise ValueError('Reference signature mismatch')
    betas=[];residuals=[];warnings=[]
    for t in GRID:
        fit=fit_fixed_scaler_probe(z[t],y,scaler=scaler,max_iter=10000,random_seed=0,
                                  class_weight='balanced',tol=1e-10)
        betas.append(fit.beta);residuals.append(fit.gradient_relative_residual)
        warnings.append(len(fit.convergence_warnings))
    betas=np.stack(betas)
    np.savez_compressed(path,beta=betas,raw_w=betas[:,:-1]/scaler.scale_,
        raw_b=betas[:,-1]-(betas[:,:-1]/scaler.scale_)@scaler.mean_,
        strengths=np.asarray(GRID),scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
        labels=y,loss_weights=weights,gradient_relative_residual=residuals,
        convergence_warning_count=warnings,metadata_json=np.asarray(json.dumps(metadata)))
    return betas,np.asarray(residuals)


def integrate(z,refs,y,weights,scaler,n,layer,target):
    beta=refs[0].copy();u0=state(beta,scaler)[1];S=0.
    rows=[];endpoints=[];vectors=[]
    endpoints.append(dict(target=target,layer=layer,substeps=n,t=0.,direct_direction_error=0.,
        relative_direction_error=np.nan,relative_raw_weight_error=0.,predicted_gradient_residual=
        gradient_relative_residual(z=z[0.],y01=y,beta=beta,sample_weights=weights),
        D=0.,reference_D=0.,Theta=0.,reference_Theta=0.,S=0.,cg_failure_count=0,
        movement_near_zero=True,raw_reference_norm_near_zero=state(refs[0],scaler)[2]<=NORM_EPS))
    cg_failures=0
    for interval,(a,b) in enumerate(zip(GRID[:-1],GRID[1:])):
        d=(z[b]-z[a])/(b-a);dt=(b-a)/n
        for step in range(n+1):
            t=b if step==n else (a if step==0 else a+step*dt)
            zt=z[a]+(step/n)*(z[b]-z[a])
            diagnostics,vector=evaluate(beta,zt,d,y,weights,scaler)
            w,u,wn=state(beta,scaler);dot=float(np.clip(u0@u,-1.,1.))
            label='outgoing' if step==0 else ('incoming' if step==n else 'interior')
            row=dict(target=target,layer=layer,substeps=n,t=t,interval=interval,interval_start=a,
                interval_end=b,step=step,step_size=dt,one_sided=label,state_kind='integrated',
                D=1.-dot,Theta=float(np.arccos(dot)),S=S,
                predicted_gradient_residual=gradient_relative_residual(z=zt,y01=y,beta=beta,sample_weights=weights),
                **diagnostics)
            rows.append(row);vectors.append(vector)
            cg_failures+=int(diagnostics['cg_info']!=0)
            if step<n:
                # Left-rectangle quadrature uses exactly the Euler derivative evaluation.
                S+=dt*diagnostics['angular_speed']
                beta=beta+dt*vector['beta_dot']
        ref_w,ref_u,ref_norm=state(refs[interval+1],scaler)
        w,u,_=state(beta,scaler);move=norm(ref_u-u0);direct=norm(u-ref_u)
        dot=float(np.clip(u0@u,-1.,1.));refdot=float(np.clip(u0@ref_u,-1.,1.))
        endpoints.append(dict(target=target,layer=layer,substeps=n,t=b,direct_direction_error=direct,
            relative_direction_error=ratio(direct,move),relative_raw_weight_error=ratio(norm(w-ref_w),ref_norm),
            predicted_gradient_residual=rows[-1]['predicted_gradient_residual'],D=1.-dot,
            reference_D=1.-refdot,Theta=float(np.arccos(dot)),reference_Theta=float(np.arccos(refdot)),
            S=S,cg_failure_count=cg_failures,movement_near_zero=move<=NORM_EPS,
            raw_reference_norm_near_zero=ref_norm<=NORM_EPS,
            solve_relative_residual=diagnostics['solve_relative_residual']))
    arrays={key:np.stack([v[key] for v in vectors]) for key in vectors[0]}
    del vectors
    return pd.DataFrame(rows),pd.DataFrame(endpoints),arrays


def reference_decomposition(out,layer,target,z,refs,y,weights,scaler):
    rows=[];arrays=[]
    for interval,(a,b) in enumerate(zip(GRID[:-1],GRID[1:])):
        d=(z[b]-z[a])/(b-a)
        for knot,label in ((interval,'outgoing'),(interval+1,'incoming')):
            diag,vector=evaluate(refs[knot],z[GRID[knot]],d,y,weights,scaler)
            rows.append(dict(target=target,layer=layer,t=GRID[knot],interval=interval,
                             one_sided=label,state_kind='reference',**diag));arrays.append(vector)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'reference_diagnostics'/f'{target}_layer_{layer:03d}.csv',index=False)
    np.savez_compressed(out/'reference_diagnostics'/f'{target}_layer_{layer:03d}.npz',
        **{k:np.stack([a[k] for a in arrays]) for k in arrays[0]},
        t=frame.t.to_numpy(),interval=frame.interval.to_numpy(),one_sided=frame.one_sided.to_numpy(dtype=str))
    return frame


def analyse_case(out,signature,layer,target,z,y,weights,scaler,metadata):
    done=out/'completion'/f'{target}_layer_{layer:03d}.json'
    if done.exists():
        info=json.loads(done.read_text());assert info['signature']==signature
        print(f'Reusing {target} layer {layer}',flush=True);return info
    refs,residuals=reference_probes(out,signature,layer,target,z,y,weights,scaler,metadata)
    refdiag=reference_decomposition(out,layer,target,z,refs,y,weights,scaler) if layer in REPRESENTATIVE else None
    previous_endpoints=None;previous_rows=None;attempts=[]
    for n in (8,16,32,64):
        tick=time.perf_counter()
        rows,endpoints,arrays=integrate(z,refs,y,weights,scaler,n,layer,target)
        maxerr=float(endpoints.relative_direction_error.max())
        trajectory_change=np.nan;S_change=np.nan;profile_change=np.nan
        if previous_endpoints is not None:
            # Direct vector agreement between complete no-reset refinement runs at knots.
            prior=np.load(out/'integrated'/f'{target}_layer_{layer:03d}.npz')
            current_indices=np.flatnonzero(rows.one_sided.to_numpy()=='incoming')
            old_indices=prior['knot_indices']
            wcur=arrays['beta'][current_indices,:-1]/scaler.scale_
            wold=prior['beta'][old_indices,:-1]/scaler.scale_
            ucur=wcur/np.linalg.norm(wcur,axis=1)[:,None];uold=wold/np.linalg.norm(wold,axis=1)[:,None]
            u0=state(refs[0],scaler)[1]
            refw=refs[1:,:-1]/scaler.scale_;refu=refw/np.linalg.norm(refw,axis=1)[:,None]
            movement=np.linalg.norm(refu-u0,axis=1)
            changes=np.divide(np.linalg.norm(ucur-uold,axis=1),movement,
                              out=np.full_like(movement,np.nan),where=movement>NORM_EPS)
            trajectory_change=float(np.nanmax(changes));prior.close()
            current_S=endpoints.S.to_numpy()[1:];old_S=previous_endpoints.S.to_numpy()[1:]
            S_changes=np.divide(np.abs(current_S-old_S),current_S,out=np.full_like(current_S,np.nan),where=current_S>NORM_EPS)
            S_change=float(np.nanmax(S_changes))
            # Doubling supplies every previous evaluation point, including both sides of kinks.
            cur=rows[rows.step%2==0];old=previous_rows
            assert len(cur)==len(old)
            profile_changes={}
            for metric in METRICS:
                cv,ov=cur[metric].to_numpy(),old[metric].to_numpy()
                differences=np.divide(np.abs(cv-ov),np.abs(cv),out=np.full_like(cv,np.nan),where=np.abs(cv)>NORM_EPS)
                profile_changes[metric]=float(np.nanmax(differences))
            profile_change=max(profile_changes.values())
        else: profile_changes={m:np.nan for m in METRICS}
        target_met=(maxerr<=.02 and trajectory_change<=.02 and S_change<=.02
                    and profile_change<=.05 and endpoints.cg_failure_count.max()==0)
        knot_indices=np.flatnonzero(rows.one_sided.to_numpy()=='incoming')
        np.savez_compressed(out/'integrated'/f'{target}_layer_{layer:03d}.npz',**arrays,
            t=rows.t.to_numpy(),interval=rows.interval.to_numpy(),step_size=rows.step_size.to_numpy(),
            one_sided=rows.one_sided.to_numpy(dtype=str),knot_indices=knot_indices,
            S=rows.S.to_numpy(),D=rows.D.to_numpy(),Theta=rows.Theta.to_numpy(),
            scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,labels=y,loss_weights=weights,
            solve_relative_residual=rows.solve_relative_residual.to_numpy(),cg_info=rows.cg_info.to_numpy(),
            metadata_json=np.asarray(json.dumps(dict(metadata,substeps=n,quadrature='left rectangle at Euler evaluation points'))))
        rows.to_csv(out/'refinements'/f'{target}_L{layer:02d}_n{n}_evaluations.csv',index=False)
        endpoints.to_csv(out/'refinements'/f'{target}_L{layer:02d}_n{n}_endpoints.csv',index=False)
        attempt=dict(substeps=n,max_relative_direction_error=maxerr,
            successive_trajectory_relative_change=trajectory_change,path_length_relative_change=S_change,
            max_common_point_profile_relative_change=profile_change,profile_relative_changes=profile_changes,
            target_met=bool(target_met),elapsed_seconds=time.perf_counter()-tick)
        attempts.append(attempt)
        atomic_json(out/'completion'/f'{target}_layer_{layer:03d}_progress.json',dict(signature=signature,attempts=attempts))
        print(f'{target} L{layer} n={n}: max_relative_error={maxerr:.4g}, S={endpoints.S.iloc[-1]:.4g}, '
              f'S_change={S_change:.3g}, profile_change={profile_change:.3g}',flush=True)
        previous_rows,previous_endpoints=rows,endpoints
        del arrays;gc.collect()
        if target_met: break
    rows.to_csv(out/'selected'/f'{target}_layer_{layer:03d}_evaluations.csv',index=False)
    endpoints.to_csv(out/'selected'/f'{target}_layer_{layer:03d}_endpoints.csv',index=False)
    if refdiag is not None:
        joined=rows[rows.one_sided!='interior'].merge(refdiag,on=['target','layer','t','interval','one_sided'],suffixes=('_integrated','_reference'))
        # Round knot times to avoid floating-point join exclusions.
        if len(joined)!=26:
            left=rows[rows.one_sided!='interior'].copy();right=refdiag.copy()
            left['t']=left.t.round(8);right['t']=right.t.round(8)
            joined=left.merge(right,on=['target','layer','t','interval','one_sided'],suffixes=('_integrated','_reference'))
        assert len(joined)==26
        for metric in METRICS:
            denominator=joined[metric+'_reference'].abs()
            joined[metric+'_relative_difference_denominator_reference']=(
                (joined[metric+'_integrated']-joined[metric+'_reference']).abs()/denominator.where(denominator>NORM_EPS))
        retained=np.flatnonzero(rows.one_sided.to_numpy()!='interior')
        with np.load(out/'integrated'/f'{target}_layer_{layer:03d}.npz') as integrated, \
             np.load(out/'reference_diagnostics'/f'{target}_layer_{layer:03d}.npz') as reference:
            for vector_name in ('r','beta_dot','raw_w_dot','u_dot'):
                left=integrated[vector_name][retained];right=reference[vector_name]
                ln=np.linalg.norm(left,axis=1);rn=np.linalg.norm(right,axis=1)
                joined[vector_name+'_cosine_integrated_reference']=np.divide(
                    np.sum(left*right,axis=1),ln*rn,out=np.full_like(rn,np.nan),where=(ln>NORM_EPS)&(rn>NORM_EPS))
                joined[vector_name+'_vector_difference_denominator_reference']=np.divide(
                    np.linalg.norm(left-right,axis=1),rn,out=np.full_like(rn,np.nan),where=rn>NORM_EPS)
                joined[vector_name+'_reference_near_zero']=rn<=NORM_EPS
        joined.to_csv(out/'reference_diagnostics'/f'{target}_layer_{layer:03d}_comparison.csv',index=False)
    info=dict(signature=signature,target=target,layer=layer,selected_substeps=n,attempts=attempts,
        practical_target_met=bool(target_met),max_reference_gradient_residual=float(residuals.max()),
        max_solve_residual=float(rows.solve_relative_residual.max()),cg_failures=int((rows.cg_info!=0).sum()),
        max_predicted_gradient_residual=float(rows.predicted_gradient_residual.max()),
        endpoint_S=float(endpoints.S.iloc[-1]),endpoint_Theta=float(endpoints.Theta.iloc[-1]),
        endpoint_reference_Theta=float(endpoints.reference_Theta.iloc[-1]),
        endpoint_Theta_over_S=ratio(endpoints.Theta.iloc[-1],endpoints.S.iloc[-1]))
    atomic_json(done,info)
    return info


def plot_outputs(out,endpoints,evaluations):
    plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
    for i,target in enumerate(('stance','style')):
        for j,layer in enumerate(REPRESENTATIVE):
            e=evaluations[(evaluations.target==target)&(evaluations.layer==layer)]
            k=endpoints[(endpoints.target==target)&(endpoints.layer==layer)]
            axes[i,j].plot(e.t,e.D,label='Integrated',linewidth=1.5)
            axes[i,j].scatter(k.t,k.reference_D,label='Independent fits',s=16,color='black')
            axes[i,j].set(title=f'{target}, L{layer}',xlabel='t',ylabel='1 − cos(u₀,u(t))')
            axes[i,j].legend(fontsize=7)
    fig.savefig(plots/'representative_rotation.png',dpi=160);plt.close(fig)
    for target in ('stance','style'):
        fig,axes=plt.subplots(7,4,figsize=(13,16),sharex=True,constrained_layout=True)
        for layer,ax in zip(range(1,29),axes.flat):
            e=evaluations[(evaluations.target==target)&(evaluations.layer==layer)]
            k=endpoints[(endpoints.target==target)&(endpoints.layer==layer)]
            ax.plot(e.t,e.D,linewidth=1);ax.scatter(k.t,k.reference_D,s=5,color='black');ax.set_title(f'L{layer}',fontsize=8)
        fig.suptitle(f'{target}: sampled piecewise-linear path rotation')
        fig.savefig(plots/f'{target}_all_layer_rotation.png',dpi=130);plt.close(fig)
    for metric in ('direct_direction_error','relative_direction_error'):
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,target in zip(axes,('stance','style')):
            matrix=endpoints[(endpoints.target==target)&(endpoints.t>0)].pivot(index='layer',columns='t',values=metric)
            im=ax.imshow(matrix,aspect='auto',origin='lower');fig.colorbar(im,ax=ax)
            ax.set(title=f'{target}: {metric}',xlabel='Sampled t',ylabel='Layer')
            ax.set_xticks(range(len(matrix.columns)),[f'{x:g}' for x in matrix.columns],rotation=60)
        fig.savefig(plots/f'{metric}_heatmap.png',dpi=150);plt.close(fig)
    for metric in METRICS:
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
        for i,target in enumerate(('stance','style')):
            for j,layer in enumerate(REPRESENTATIVE):
                e=evaluations[(evaluations.target==target)&(evaluations.layer==layer)]
                # Separate segments; dots mark both one-sided boundary values.
                for _,segment in e.groupby('interval',sort=True):
                    axes[i,j].plot(segment.t,segment[metric],color=f'C{j}',linewidth=1.1)
                    edge=segment[segment.one_sided!='interior']
                    axes[i,j].scatter(edge.t,edge[metric],s=7,color=f'C{j}')
                axes[i,j].set(title=f'{target}, L{layer}',xlabel='t',ylabel=metric)
        fig.savefig(plots/f'{metric}_representative_profiles.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for i,target in enumerate(('stance','style')):
            for j,side in enumerate(('incoming','outgoing')):
                matrix=evaluations[(evaluations.target==target)&(evaluations.one_sided==side)].pivot(index='layer',columns='t',values=metric)
                im=axes[i,j].imshow(matrix,aspect='auto',origin='lower');fig.colorbar(im,ax=axes[i,j])
                axes[i,j].set(title=f'{target}, {side}: {metric}',xlabel='Knot t',ylabel='Layer')
                axes[i,j].set_xticks(range(len(matrix.columns)),[f'{x:g}' for x in matrix.columns],rotation=60)
        fig.savefig(plots/f'{metric}_one_sided_heatmaps.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,target in zip(axes,('stance','style')):
        k=endpoints[(endpoints.target==target)&np.isclose(endpoints.t,1.)].sort_values('layer')
        ax.plot(k.layer,k.S,label='Integrated angular path length S')
        ax.plot(k.layer,k.Theta,label='Integrated endpoint angle Θ')
        ax.plot(k.layer,k.reference_Theta,'--',label='Reference endpoint angle')
        ax.set(title=target,xlabel='Layer',ylabel='Radians');ax.legend(fontsize=7)
    fig.savefig(plots/'endpoint_angle_and_path_length.png',dpi=160);plt.close(fig)


def final_report(out,infos,metadata,started):
    ends=pd.concat([pd.read_csv(p) for p in sorted((out/'selected').glob('*_endpoints.csv'))],ignore_index=True)
    evaluations=pd.concat([pd.read_csv(p) for p in sorted((out/'selected').glob('*_evaluations.csv'))],ignore_index=True)
    ends.to_csv(out/'endpoint_metrics.csv',index=False);evaluations.to_csv(out/'path_metrics.csv',index=False)
    summary=pd.DataFrame([{k:v for k,v in info.items() if k!='attempts'} for info in infos])
    summary.to_csv(out/'layer_summary.csv',index=False)
    plot_outputs(out,ends,evaluations)
    elapsed=time.perf_counter()-started
    failed=[f"{i['target']} L{i['layer']}" for i in infos if not i['practical_target_met']]
    resources=dict(elapsed_seconds=elapsed,peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   gpu_used=False,threadpools=metadata['threadpools'])
    atomic_json(out/'analysis_resources.json',resources)
    lines=['# Full-strength sampled-path probe sensitivity','',
        'This tracks regularized full-data probe optima along a sampled, piecewise-linear activation path. '
        'Numerical substeps are not model samples. References after t=0 never reset or correct integration.','',
        '## Activation series','',
        'A separate NF4/double-quantized, FP16-compute GPU series was regenerated at all 14 strengths. '
        'The old unquantized activations were not mixed into this analysis. Endpoint differences in the '
        'resource pilot are material; these results describe this new quantized model series.',
        f"Series: `{metadata['series_identifier']}`. Source/configuration identifiers are in run_metadata.json.",'',
        '## Objective and coordinates','',
        'Fixed full-data t=0 scaler per layer; float64 analysis; liblinear C=1, summed class-weighted '
        'logistic loss plus 0.5||beta||², including penalized intercept with intercept_scaling=1. '
        'Weights are fixed across t. beta=(v,c); raw w=v/scale and b=c−w·mean. No damping or dense Hessians. '
        'Response gain depends on curvature AND forcing alignment; raw gain includes scaler conversion. '
        'Cross-layer comparisons are conditional on different base scalers. No causal percentages.','',
        '## Accuracy and numerical refinement','',
        f"Maximum reference gradient residual: {summary.max_reference_gradient_residual.max():.3e}; "
        f"maximum solve residual: {summary.max_solve_residual.max():.3e}; CG failures: {summary.cg_failures.sum()}.",
        f"Maximum decomposition identity relative error: {evaluations.identity_relative_error.max():.3e}.",
        'Each refinement restarts only at the t=0 reference. Practical screens are maximum relative '
        'direction error ≤2%, successive knot trajectory change ≤2%, path-length change at every knot ≤2%, '
        'and maximum scalar-profile change at common evaluations (including both sides of kinks) ≤5%. '
        'These are not significance thresholds.',
        f"Selected substep counts: {summary.selected_substeps.value_counts().sort_index().to_dict()}.",
        f"Cases not meeting all screens at the 64-substep cap: {failed or 'none'}.",
        'Path length uses left-rectangle quadrature at the same evaluations as Euler. Both incoming '
        'and outgoing slopes are retained at kinks. Refinement CSVs preserve scalar trajectories; '
        'binary files retain the selected complete no-reset trajectory only.','',
        '## Reference-state decomposition checks','']
    comps=list((out/'reference_diagnostics').glob('*_comparison.csv'))
    if comps:
        comparisons=pd.concat([pd.read_csv(p) for p in comps],ignore_index=True)
        comparisons.to_csv(out/'reference_state_comparisons.csv',index=False)
        for metric in METRICS:
            field=metric+'_relative_difference_denominator_reference'
            lines.append(f"- {metric}: integrated/reference one-sided relative difference median/max "
                         f"{comparisons[field].median():.3g}/{comparisons[field].max():.3g} (reference denominator).")
        for vector_name in ('r','beta_dot','raw_w_dot','u_dot'):
            field=vector_name+'_cosine_integrated_reference'
            lines.append(f"- {vector_name}: integrated/reference cosine median/min "
                         f"{comparisons[field].median():.4f}/{comparisons[field].min():.4f}.")
    lines+=['','## Descriptive depth/task patterns','']
    for target in ('stance','style'):
        k=summary[summary.target==target]
        lines.append(f"- {target}: endpoint reference angle range {k.endpoint_reference_Theta.min():.3f}–"
                     f"{k.endpoint_reference_Theta.max():.3f} rad; integrated path length range "
                     f"{k.endpoint_S.min():.3f}–{k.endpoint_S.max():.3f} rad; Θ/S range "
                     f"{k.endpoint_Theta_over_S.min():.3f}–{k.endpoint_Theta_over_S.max():.3f}.")
        for field in ('endpoint_reference_Theta','endpoint_S'):
            early=k.loc[k.layer.between(1,8),field].mean();late=k.loc[k.layer.between(20,28),field].mean()
            lines.append(f"  {field}: early/late layer means {early:.3f}/{late:.3f} rad.")
    candidates=evaluations.groupby(['interval_start','interval_end']).angular_speed.max().sort_values(ascending=False).head(3)
    lines+=['','## Limits and possible later checks','',
        'Endpoint agreement establishes coefficient tracking for the constructed activation path, '
        'not fidelity of its interior to actual model activations. Large path-length excess reflects '
        'bending and/or backtracking, not exclusively cancellation. Endpoint agreement alone does not '
        'validate path length or local profiles; use the separate refinement results.',
        f"Candidate actual midpoint-validation intervals (largest sampled-path angular speeds): {list(candidates.index)}. "
        'No midpoints or transformer Jacobians were extracted in this pass.','',
        '## Resources and reproduction','',
        f"Analysis elapsed {elapsed:.1f} s; peak process RAM {resources['peak_ram_bytes']/2**30:.2f} GiB. "
        'One CPU process, at most two BLAS/OpenMP threads. Extraction resources are recorded separately '
        'in ../extraction_resources.json.',
        '```bash',
        'PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/extract_full_strength_series.py',
        'PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/analyze_full_strength_sensitivity.py',
        '```',
        'GPU extraction requires execution outside the GPU-blocking sandbox. Analysis is CPU-only. '
        'Both commands resume compatible completed work; incomplete CPU cases restart at t=0.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-root',default='results/full_strength_nf4_fp16/activations')
    parser.add_argument('--output-root',default='results/full_strength_nf4_fp16/analysis')
    parser.add_argument('--representative-only',action='store_true')
    args=parser.parse_args();started=time.perf_counter()
    root=Path(args.activation_root).resolve();out=Path(args.output_root).resolve()
    for directory in ('references','integrated','refinements','selected','completion','reference_diagnostics'):
        (out/directory).mkdir(parents=True,exist_ok=True)
    series=json.loads((root/'series_config.json').read_text())
    sources={}
    for t in GRID:
        path=strength_dir(root,t);manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['series_identifier']==series['series_identifier']
        digest=hashlib.sha256()
        with (path/manifest['activation_file']).open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest()!=manifest['activation_sha256']:
            raise ValueError(f'Activation checksum mismatch at t={t}')
        sources[str(t)]=dict(path=str(path),activation_sha256=manifest['activation_sha256'])
    objective=dict(C=1.,loss_normalization='sum',class_weight='balanced',sample_weight=None,
        solver='liblinear',dtype='float64',penalty='l2',intercept_penalized=True,intercept_scaling=1.,fit_tol=1e-10,max_iter=10000,
        random_state=0,cg_rtol=1e-9,cg_atol=1e-11,cg_maxiter=2000,damping=None)
    signature=hashlib.sha256(json.dumps(dict(series=series['series_identifier'],objective=objective,
        sources=sources,algorithm='Euler-left-path-quadrature-v1'),sort_keys=True).encode()).hexdigest()
    metadata=dict(signature=signature,series_identifier=series['series_identifier'],objective=objective,
        sources=sources,grid=GRID,near_zero_threshold=NORM_EPS,algorithm='Euler, no resets',
        implementation_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('scripts/analyze_full_strength_sensitivity.py','scripts/decompose_local_probe_sensitivity.py',
             'scripts/validate_logistic_probe_sensitivity.py')},
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pandas','threadpoolctl')})
    infos=[]
    with threadpool_limits(limits=2):
        metadata['threadpools']=threadpool_info()
        assert all(p['num_threads']<=2 for p in metadata['threadpools'])
        atomic_json(out/'run_metadata.json',metadata)
        order=list(REPRESENTATIVE)+[l for l in range(1,29) if l not in REPRESENTATIVE]
        if args.representative_only: order=list(REPRESENTATIVE)
        for layer in order:
            data=load_layer(root,layer);scaler=StandardScaler().fit(data[0.][0])
            z={t:scaler.transform(data[t][0]).astype(np.float64) for t in GRID}
            for target in ('stance','style'):
                y,mapping=encode_binary_target(data[0.][1],target)
                assert mapping==EXPECTED_LABEL_MAPPINGS[target]
                weights=sample_loss_weights(y,'balanced')
                case_metadata=dict(metadata,layer=layer,target=target,label_mapping=mapping,
                    prompt_ids=data[0.][1].prompt_id.astype(str).tolist())
                infos.append(analyse_case(out,signature,layer,target,z,y,weights,scaler,case_metadata))
            del data,z;gc.collect()
    final_report(out,infos,metadata,started)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__': main()
