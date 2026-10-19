"""Full-data instantaneous probe sensitivity from matching FP32 activations/direct JVPs."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
import tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='instantaneous-profiles-'))

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info,threadpool_limits
from decompose_local_probe_sensitivity import forcing,norm,ratio
from ptolemaic_mechinterp.probes.dataset import encode_binary_target
from validate_logistic_probe_sensitivity import EXPECTED_LABEL_MAPPINGS,NORM_EPS,analytical_beta_dot,fit_fixed_scaler_probe,sample_loss_weights

GRID=(0.,.01,.02,.04,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)
METRICS={'r_norm':'||r|| (standardized probe-gradient units per t)',
 'standardized_beta_gain':'||β_dot|| / ||r|| (standardized coordinates)',
 'w_dot_norm':'||w_dot|| (raw activation coordinates per t)',
 'raw_weight_gain':'||w_dot|| / ||r|| (raw response gain)',
 'perpendicular_fraction':'||w_dot_perp|| / ||w_dot|| (unitless)',
 'weight_norm':'||w|| (raw activation coordinates)',
 'inverse_weight_norm':'1 / ||w|| (inverse raw weight units)',
 'directional_gain':'||u_dot|| / ||r|| (angular response gain)',
 'angular_speed':'||u_dot|| (radians per unit t)'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean_json(value),indent=2,sort_keys=True,allow_nan=False));tmp.replace(path)


def save_npz(path,**arrays):
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)


def name(t):return f't_{t:.4f}'.replace('.','p')


def compute_case(z,dz,scaler,y,weights,target,layer,t,metadata,scaler_path):
    row=dict(target=target,layer=layer,t=t,sensitivity_source='direct scalar FP32 forward AD',
        D_rows=z.shape[0],D_columns=z.shape[1],z_dot_frobenius_norm=norm(dz),z_dot_rms=norm(dz)/np.sqrt(dz.size),
        fit_attempts=0,solve_attempts=0,status='pending',nonfinite_values=False)
    fit=None
    for tol,max_iter in ((1e-10,10000),(1e-12,20000)):
        row['fit_attempts']+=1
        try:
            fit=fit_fixed_scaler_probe(z,y,scaler=scaler,max_iter=max_iter,random_seed=0,class_weight='balanced',tol=tol)
            row.update(optimality_relative_residual=float(fit.gradient_relative_residual),fit_tol=tol,
                fit_iterations=fit.n_iter,convergence_warning_count=len(fit.convergence_warnings))
            if np.isfinite(fit.beta).all() and fit.gradient_relative_residual<1e-6:break
        except (ValueError,RuntimeError,FloatingPointError) as error:row['fit_error']=str(error)
    payload={k:np.full(3585 if k in ('beta','beta_dot','r') else 3584,np.nan,dtype=np.float64)
             for k in ('beta','beta_dot','r','w','w_dot','w_dot_perp','u','u_dot')}
    payload.update(metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)),
                   scaler_file=np.asarray(str(scaler_path)),labels=y,loss_weights=weights)
    if fit is None or not np.isfinite(fit.beta).all() or fit.gradient_relative_residual>=1e-6:
        row.update(status='fit_unresolved',fit_ok=False,solve_ok=False,nonfinite_values=fit is None or not np.isfinite(fit.beta).all())
        if fit is not None:payload['beta']=fit.beta
        return row,payload
    beta=fit.beta;w=fit.raw_w;wn=norm(w);u=w/wn if wn>NORM_EPS else np.full_like(w,np.nan)
    r=forcing(z,beta,dz,y,weights)
    bd=np.full_like(beta,np.nan);info=dict(cg_info=-999,solve_relative_residual=np.nan)
    for rtol,atol,limit in ((1e-9,1e-11,2000),(1e-11,1e-13,4000)):
        row['solve_attempts']+=1
        try:
            bd,info=analytical_beta_dot(z0=z,y01=y,beta0=beta,dz_dt=dz,sample_weights=weights,
                                       cg_rtol=rtol,cg_atol=atol,cg_maxiter=limit)
            if info['cg_info']==0 and info['solve_relative_residual']<1e-8 and np.isfinite(bd).all():break
        except (ValueError,RuntimeError,FloatingPointError) as error:row['solve_error']=str(error)
    wd=bd[:-1]/scaler.scale_;perp=wd-u*float(u@wd);ud=perp/wn if wn>NORM_EPS else np.full_like(w,np.nan)
    rn,wdn,pn,un=norm(r),norm(wd),norm(perp),norm(ud)
    raw_gain=ratio(wdn,rn);fraction=ratio(pn,wdn);inverse=ratio(1.,wn)
    product=rn*raw_gain*fraction*inverse
    finite=all(np.isfinite(v).all() for v in (beta,r,bd,w,wd,u,ud))
    solve_ok=info['cg_info']==0 and info['solve_relative_residual']<1e-8 and finite
    row.update(status='ok' if solve_ok else 'solve_unresolved',fit_ok=True,solve_ok=bool(solve_ok),
        nonfinite_values=not finite,r_norm=rn,standardized_beta_gain=ratio(norm(bd),rn),w_dot_norm=wdn,
        raw_weight_gain=raw_gain,perpendicular_fraction=fraction,weight_norm=wn,inverse_weight_norm=inverse,
        directional_gain=ratio(un,rn),angular_speed=un,identity_product=product,
        identity_absolute_error=float(abs(un-product)),identity_relative_error=ratio(abs(un-product),un),
        r_near_zero=rn<=NORM_EPS,w_dot_near_zero=wdn<=NORM_EPS,w_near_zero=wn<=NORM_EPS,
        angular_speed_near_zero=un<=NORM_EPS,**info)
    payload.update(beta=beta,beta_dot=bd,r=r,w=w,w_dot=wd,w_dot_perp=perp,u=u,u_dot=ud,
                   raw_b=np.asarray(fit.raw_b),metrics_json=np.asarray(json.dumps(clean_json(row),sort_keys=True)))
    return row,payload


def make_plots(out,frame):
    plots=out/'plots';plots.mkdir(exist_ok=True)
    for metric,label in METRICS.items():
        label=textwrap.fill(label,width=34)
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
        for i,target in enumerate(('stance','style')):
            for j,layer in enumerate((6,16,28)):
                f=frame[(frame.target==target)&(frame.layer==layer)].sort_values('t')
                axes[i,j].plot(f.t,f[metric],'-o',markersize=3,linewidth=.8)
                axes[i,j].set(title=f'{target}, layer {layer}',xlabel='LoRA strength t',ylabel=label)
        fig.suptitle('Instantaneous direct-JVP evaluations; connecting lines are visual guides',fontsize=11)
        fig.savefig(plots/f'{metric}_representative.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for ax,target in zip(axes,('stance','style')):
            f=frame[frame.target==target].pivot(index='layer',columns='t',values=metric).reindex(index=range(1,29),columns=GRID)
            im=ax.imshow(f,origin='lower',aspect='auto',interpolation='nearest');fig.colorbar(im,ax=ax,label=label)
            ax.set_xticks(range(14),[f'{t:g}' for t in GRID],rotation=60)
            ax.set_yticks(range(28),range(1,29),fontsize=6);ax.set(title=target,xlabel='Sampled t (categorical columns)',ylabel='Transformer layer')
        fig.suptitle('Instantaneous values at actual sampled strengths; no interpolation',fontsize=11)
        fig.savefig(plots/f'{metric}_heatmap.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,target in zip(axes,('stance','style')):
            for t in (0.,.5,1.):
                f=frame[(frame.target==target)&(frame.t==t)].sort_values('layer')
                ax.plot(f.layer,f[metric],'-o',label=f't={t:g}',markersize=2,linewidth=.8)
            ax.set(title=target,xlabel='Transformer layer',ylabel=label);ax.legend(fontsize=8)
        fig.suptitle('Instantaneous depth profiles; lines are visual guides',fontsize=11)
        fig.savefig(plots/f'{metric}_depth.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
    for i,target in enumerate(('stance','style')):
        for j,layer in enumerate((6,16,28)):
            f=frame[(frame.target==target)&(frame.layer==layer)].sort_values('t')
            axes[i,j].plot(f.t,f.direct_fit_rotation_D,'-o',markersize=3,linewidth=.8)
            axes[i,j].set(title=f'{target}, layer {layer}',xlabel='t',ylabel='1 − u(0)·u(t), direct fits')
    fig.suptitle('Matching FP32 full-data geometry; no integrated coefficients',fontsize=11)
    fig.savefig(plots/'direct_fit_rotation_representative.png',dpi=160);plt.close(fig)
    for target in ('stance','style'):
        fig,axes=plt.subplots(7,4,figsize=(12,16),sharex=True,constrained_layout=True)
        for layer,ax in zip(range(1,29),axes.flat):
            f=frame[(frame.target==target)&(frame.layer==layer)].sort_values('t')
            ax.plot(f.t,f.direct_fit_rotation_D,'-o',markersize=2,linewidth=.7);ax.set(title=f'Layer {layer}',xlabel='t',ylabel='1 − u₀·u(t)')
        fig.suptitle(f'{target}: matching FP32 direct-fit rotation; markers are sampled fits',fontsize=11)
        fig.savefig(plots/f'direct_fit_rotation_{target}_all_layers.png',dpi=130);plt.close(fig)


def write_report(root,out,frame,resources):
    extraction=json.loads((root/'extraction_resources.json').read_text())
    successful=frame[frame.status=='ok']
    lines=['# Full instantaneous FP32 probe-sensitivity decomposition','',
        f'Completed table: {len(frame)}/784 target × layer × strength rows; successful numerical rows: '
        f'{len(successful)}. All 120 aligned prompts, 14 strengths and transformer layers 1–28 are included.',
        'This series contains matching FP32 primal activations and direct scalar JVPs. '
        'Earlier Flash/FP16 activation stores and piecewise-linear cumulative figures remain separate. '
        'No secant activation derivatives or integrated parameter estimates enter these calculations.','',
        '## Numerical quality','',
        f"Maximum starting optimality relative residual: {frame.optimality_relative_residual.max():.3e}; "
        f"maximum linear-solve relative residual: {frame.solve_relative_residual.max():.3e}.",
        f"Nonzero CG statuses: {int((frame.cg_info.fillna(-999)!=0).sum())}; "
        f"fit retries: {int((frame.fit_attempts>1).sum())}; solve retries: {int((frame.solve_attempts>1).sum())}.",
        f"Maximum identity absolute/relative error: {frame.identity_absolute_error.max():.3e} / "
        f"{frame.identity_relative_error.max():.3e}. Nonfinite-vector rows: {int(frame.nonfinite_values.sum())}.",
        f"Near-zero denominator flags: {frame[['r_near_zero','w_dot_near_zero','w_near_zero','angular_speed_near_zero']].sum().to_dict()}.",
        f"Unresolved rows: {frame.loc[frame.status!='ok',['target','layer','t','status']].to_dict('records') or 'none'}.",
        'Optimality residual uses ||g|| / max(||data_gradient|| + ||beta||, 1e-12). '
        'Solve residual uses ||H beta_dot + r|| / max(||r||, 1e-12), matching the validated helpers. '
        'For scalar gains and identity relative errors, denominators ≤1e-12 are explicitly flagged '
        'and ratios left undefined (NaN), never replaced by a small denominator. Targets are optimality '
        '<1e-6 and solve <1e-8 with CG success; these are numerical checks, not significance thresholds.','',
        '## Objective and coordinates','',
        'All fits are regularized full-data fits: liblinear C=1, summed class-weighted logistic loss, '
        'fixed balanced class weights, no additional sample weights, and L2 penalty on standardized '
        'beta=(v,c), including the intercept with intercept_scaling=1. Label orientation is geocentric→0, '
        'heliocentric→1 and modern→0, premodern→1. A single scaler is fit on the new FP32 t=0 data '
        'per layer and reused unchanged: z=(h−mu0)/scale0, z_dot=h_dot/scale0.',
        'Each sensitivity is evaluated at that strength’s independently fitted optimum. '
        'The forcing r is the probe-gradient perturbation at fixed beta, not a probe-independent '
        'magnitude of LoRA. H is applied matrix-free, with no damping or dense inverse. '
        'Response gain depends jointly on curvature and forcing alignment. Raw-weight gain includes '
        'the scaler conversion; perpendicular fraction and inverse weight norm remain separate factors. '
        'Cross-layer norms and gains depend on the different layerwise base scalers and stated coordinates. '
        'No causal percentages are assigned to Hessian versus forcing.','',
        '## Descriptive depth and strength trends','']
    trends=[]
    for target in ('stance','style'):
        for t in (0.,.5,1.):
            f=frame[(frame.target==target)&(frame.t==t)]
            values=[]
            for metric in ('r_norm','raw_weight_gain','perpendicular_fraction','inverse_weight_norm','angular_speed'):
                early=f.loc[f.layer.between(1,8),metric].mean();late=f.loc[f.layer.between(20,28),metric].mean()
                trends.append(dict(target=target,t=t,metric=metric,early_layers_1_8_mean=early,late_layers_20_28_mean=late))
                values.append(f'{metric} {early:.4g}→{late:.4g}')
            lines.append(f"- {target}, t={t:g}, early→late layer means: "+'; '.join(values)+'.')
        for layer in (6,16,28):
            f=frame[(frame.target==target)&(frame.layer==layer)].sort_values('t')
            peak=f.loc[f.angular_speed.idxmax()]
            lines.append(f"- {target} layer {layer}: sampled angular-speed range "
                         f"{f.angular_speed.min():.4g}–{f.angular_speed.max():.4g} rad/t; sampled maximum at t={peak.t:g}.")
        endpoint=frame[(frame.target==target)&(frame.t==1.)]
        lines.append(f"- {target}: direct-fit endpoint angular separation range "
                     f"{endpoint.direct_fit_rotation_angle.min():.4g}–{endpoint.direct_fit_rotation_angle.max():.4g} rad.")
    pd.DataFrame(trends).to_csv(out/'descriptive_depth_summary.csv',index=False)
    lines+=['',
        'These are descriptive comparisons at actual sampled t values, not causal attribution or proof '
        'of a monotone trend. Profile markers are instantaneous evaluations; connecting lines are visual '
        'guides only. Heatmap columns are explicitly categorical sampled strengths, with layer labels 1–28. '
        'There are no incoming/outgoing secant branches, inferred midpoints, smoothed derivatives, '
        'integrated coefficients or path-length estimates.','',
        '## Numerical configuration, resources and limits','',
        'The validated pipeline retains packed NF4 weights and quant_state, reconstructs the same '
        'FP16-valued matrices and promotes them to FP32 per operation. FP32 activations/tangents, math '
        'SDPA, no autocast/TF32, frozen parameters, evaluation mode, native differentiable LoRA scaling '
        'including intrinsic alpha/r, batch size 1, no cache, no reverse graph and no logits. '
        'Packed-state hashes are checked against the successful pilot before extraction and again after '
        'the final pass. No full unquantized model or cross-layer dequantized cache is retained.',
        f"Extraction elapsed {extraction['elapsed_seconds']/60:.2f} min; completed new pairs "
        f"{extraction['completed_new_pairs']}; forward compute {extraction['forward_seconds']/60:.2f} min. "
        f"Peak GPU allocation/reservation: {extraction['peak_allocated_bytes']/2**30:.2f}/"
        f"{extraction['peak_reserved_bytes']/2**30:.2f} GiB; host RAM {extraction['peak_host_ram_bytes']/2**30:.2f} GiB.",
        f"CPU analysis elapsed {resources['elapsed_seconds']/60:.2f} min, peak host RAM "
        f"{resources['peak_host_ram_bytes']/2**30:.2f} GiB, one process with ≤2 numerical threads and float64 calculations.",
        'GPU allocator peaks exclude driver/context allocations; host RSS includes file-backed pages '
        'and model loading. Per-session extraction resource logs are retained. GPU extraction exits '
        'before CPU analysis. Model loading and CPU copies/I/O are included in extraction elapsed time.',
        'The preceding bounded FP32 pilot supplies finite-difference implementation evidence; no new '
        'finite-difference sweep was run. This experiment computes instantaneous derivatives of the '
        'specified quantized model arithmetic and objective at sampled strengths. It does not establish '
        'transformer-level mechanisms, statistical generalization or a result for a different precision/model.','',
        '## Artifacts and reproduction','',
        'Matching activation/JVP arrays: ../activations/t_*/h.npy and h_dot.npy, shape [120,28,3584]. '
        'Each committed prompt has a completion JSON with ID, t, hashes, timing and finite checks. '
        'Strength manifests are committed only after all prompts. No repeated whole-dataset compression is used.',
        'metrics.csv is the tidy 784-row table. artifacts/*.npz retain beta, beta_dot, r, raw w/w_dot, '
        'u/u_dot and perpendicular vectors, with labels, fixed weights and source/objective metadata. '
        'scalers/layer_*.npz retain mean, scale, variance, prompt IDs and labels/weights. '
        'rows/*.json and atomic NPZ commits support analysis resumption. No dense Hessians or duplicated '
        'activation arrays are saved under analysis. Code snapshots and source hashes are preserved.',
        '','```bash',
        'PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/extract_instantaneous_fp32.py',
        'PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/analyze_instantaneous_fp32.py',
        '```',
        'GPU extraction requires execution outside the GPU-blocking sandbox. CPU analysis runs only '
        'after extraction exits. Compatible completed work is reused. No CV, bootstrap, steering, '
        'generation, retraining, spectral analysis, reintegration or extra strength sampling was performed.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='results/instantaneous_nf4_fp32_jvp')
    args=parser.parse_args();root=Path(args.root).resolve();out=root/'analysis'
    for directory in ('artifacts','rows','scalers','reconstruction'):(out/directory).mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();series=json.loads((root/'series_config.json').read_text());prompts=pd.read_csv(root/'prompts.csv')
    sources={}
    for t in GRID:
        folder=root/'activations'/name(t);manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['series_identifier']==series['series_identifier'] and manifest['completed_prompts']==120
        assert manifest['prompt_ids']==prompts.prompt_id.tolist() and manifest['layers']==list(range(1,29))
        assert sha(folder/'h.npy')==manifest['h_sha256'] and sha(folder/'h_dot.npy')==manifest['h_dot_sha256']
        sources[str(t)]=dict(path=str(folder),h_sha256=manifest['h_sha256'],h_dot_sha256=manifest['h_dot_sha256'])
    objective=dict(solver='liblinear',C=1.,loss_normalization='sum',class_weight='balanced',sample_weight=None,
        penalty='l2',intercept_penalized=True,intercept_scaling=1.,random_state=0,
        fit_retries=[[1e-10,10000],[1e-12,20000]],solve_retries=[[1e-9,1e-11,2000],[1e-11,1e-13,4000]],damping=None)
    implementations={p:sha(p) for p in ('scripts/analyze_instantaneous_fp32.py','scripts/decompose_local_probe_sensitivity.py','scripts/validate_logistic_probe_sensitivity.py')}
    config=dict(series_identifier=series['series_identifier'],sources=sources,objective=objective,
        implementation_sha256=implementations,near_zero_threshold=NORM_EPS,
        derivative_source='direct FP32 scalar forward AD; no secants; no integration',probe_dtype='float64',
        scaler='full-data new FP32 t=0 activations per layer; fixed across t',
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pandas','threadpoolctl')})
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['analysis_identifier']=identifier
    if (out/'run_metadata.json').exists():
        previous=json.loads((out/'run_metadata.json').read_text())
        if previous['analysis_identifier']!=identifier:
            raise RuntimeError('Analysis configuration/source mismatch; preserve this output and choose a new root')
    for p in implementations:shutil.copyfile(p,out/'reconstruction'/Path(p).name)
    labels={};weights={}
    for target in ('stance','style'):
        labels[target],mapping=encode_binary_target(prompts,target);assert mapping==EXPECTED_LABEL_MAPPINGS[target]
        weights[target]=sample_loss_weights(labels[target],'balanced')
    rows=[]
    with threadpool_limits(limits=2):
        config['threadpools']=threadpool_info();assert all(p['num_threads']<=2 for p in config['threadpools'])
        atomic_json(out/'run_metadata.json',config)
        for layer in range(1,29):
            base=np.load(root/'activations'/name(0.)/'h.npy',mmap_mode='r')
            scaler=StandardScaler().fit(np.asarray(base[:,layer-1,:],dtype=np.float64));del base
            scaler_path=out/'scalers'/f'layer_{layer:02d}.npz'
            save_npz(scaler_path,mean=scaler.mean_,scale=scaler.scale_,variance=scaler.var_,
                prompt_ids=prompts.prompt_id.to_numpy(dtype=str),stance_labels=labels['stance'],style_labels=labels['style'],
                stance_loss_weights=weights['stance'],style_loss_weights=weights['style'],metadata_json=np.asarray(json.dumps(config)))
            for t in GRID:
                folder=root/'activations'/name(t)
                h=np.load(folder/'h.npy',mmap_mode='r');hd=np.load(folder/'h_dot.npy',mmap_mode='r')
                x=np.asarray(h[:,layer-1,:],dtype=np.float64);xd=np.asarray(hd[:,layer-1,:],dtype=np.float64)
                del h,hd
                assert np.isfinite(x).all() and np.isfinite(xd).all()
                z=scaler.transform(x);dz=xd/scaler.scale_
                for target in ('stance','style'):
                    stem=f'{target}_L{layer:02d}_{name(t)}';row_path=out/'rows'/f'{stem}.json';vector_path=out/'artifacts'/f'{stem}.npz'
                    if row_path.exists() and vector_path.exists():
                        row=json.loads(row_path.read_text());assert row['analysis_identifier']==identifier
                    else:
                        meta=dict(analysis_identifier=identifier,series_identifier=series['series_identifier'],
                                  target=target,layer=layer,t=t,objective=objective,source=sources[str(t)],
                                  label_mapping=EXPECTED_LABEL_MAPPINGS[target])
                        row,payload=compute_case(z,dz,scaler,labels[target],weights[target],target,layer,t,meta,scaler_path)
                        row['analysis_identifier']=identifier;row['vector_file']=str(vector_path)
                        save_npz(vector_path,**payload);atomic_json(row_path,row)
                    rows.append(row)
                    temp=out/'metrics.tmp.csv';pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(out/'metrics.csv')
                del x,xd,z,dz
            print(f'Completed layer {layer}/28, rows={len(rows)}/784, elapsed_min={(time.perf_counter()-started)/60:.2f}',flush=True)
            gc.collect()
    frame=pd.DataFrame(rows)
    for (target,layer),group in frame.groupby(['target','layer']):
        valid=group[group.t==0.].iloc[0]
        with np.load(valid.vector_file) as a:u0=a['u'].copy()
        for index,row in group.iterrows():
            with np.load(row.vector_file) as a:ut=a['u']
            dot=float(np.clip(u0@ut,-1.,1.))
            frame.loc[index,'direct_fit_rotation_D']=1.-dot
            frame.loc[index,'direct_fit_rotation_angle']=float(np.arccos(dot))
    frame.to_csv(out/'metrics.csv',index=False)
    make_plots(out,frame)
    resources=dict(elapsed_seconds=time.perf_counter()-started,peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   gpu_used=False,threadpools=config['threadpools'])
    atomic_json(out/'analysis_resources.json',resources)
    write_report(root,out,frame,resources)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':main()
