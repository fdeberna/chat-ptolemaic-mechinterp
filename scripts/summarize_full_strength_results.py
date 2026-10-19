"""Finalize acceptance/resource reporting from completed scalar artifacts; no fitting."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def main():
    root=Path('results/full_strength_nf4_fp16')
    out=root/'analysis'
    infos=[json.loads(p.read_text()) for p in sorted((out/'completion').glob('*.json'))
           if not p.name.endswith('_progress.json')]
    if len(infos)!=56: raise RuntimeError(f'Expected 56 completed cases, found {len(infos)}')
    rows=[]
    for info in infos:
        last=info['attempts'][-1]
        row=dict(target=info['target'],layer=info['layer'],selected_substeps=last['substeps'],
            max_relative_direction_error=last['max_relative_direction_error'],
            successive_trajectory_relative_change=last['successive_trajectory_relative_change'],
            max_path_length_relative_change=last['path_length_relative_change'],
            max_common_profile_relative_change=last['max_common_point_profile_relative_change'],
            direction_target_met=last['max_relative_direction_error']<=.02,
            trajectory_stable=last['successive_trajectory_relative_change']<=.02,
            path_length_stable=last['path_length_relative_change']<=.02,
            profiles_stable=last['max_common_point_profile_relative_change']<=.05,
            all_screens_met=info['practical_target_met'])
        row.update({f'{key}_max_relative_refinement_change':value
                    for key,value in last['profile_relative_changes'].items()})
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'refinement_summary.csv',index=False)
    full=json.loads((out/'analysis_resources.json').read_text())
    representative=json.loads((out/'representative_analysis_resources.json').read_text())
    resources=dict(analysis_elapsed_seconds=full['elapsed_seconds']+representative['elapsed_seconds'],
        analysis_peak_ram_bytes=max(full['peak_ram_bytes'],representative['peak_ram_bytes']),
        representative_elapsed_seconds=representative['elapsed_seconds'],
        all_layer_extension_elapsed_seconds=full['elapsed_seconds'],gpu_used_for_analysis=False,
        extraction=json.loads((root/'extraction_resources.json').read_text()),
        post_extraction_gpu_memory_mib=0,
        post_extraction_memory_verification='nvidia-smi: 0 MiB used; extraction process exited',
        note='Extraction allocator measurement preceded process exit; all GPU memory was then released.')
    (out/'resource_totals.json').write_text(json.dumps(resources,indent=2,sort_keys=True))
    lines=['','## Final acceptance counts and limitations','',
           'The 2% direction screen and sensitivity-profile stability are distinct. The following '
           'counts apply to the selected, bounded refinements, without claiming convergence beyond the cap.']
    points=pd.read_csv(out/'path_metrics.csv')
    lines.append(f"Maximum predicted-state gradient residual: {points.predicted_gradient_residual.max():.3e}. "
                 'Predicted states are not optimizer-corrected; their nonzero optimality residual is '
                 'reported separately from reference-fit accuracy and CG solve accuracy.')
    warning_count=0
    for path in (out/'references').glob('*.npz'):
        with np.load(path) as archive: warning_count+=int(archive['convergence_warning_count'].sum())
    lines.append(f'Reference convergence warnings: {warning_count}.')
    for target in ('stance','style'):
        f=frame[frame.target==target]
        lines.append(f"- {target}: direction target {f.direction_target_met.sum()}/28; successive trajectory "
                     f"agreement {f.trajectory_stable.sum()}/28; path-length stability {f.path_length_stable.sum()}/28; "
                     f"all-factor profile stability {f.profiles_stable.sum()}/28.")
        lines.append(f"  Maximum relative direction error: {f.max_relative_direction_error.max():.2%}. "
                     f"Direction failures: {f.loc[~f.direction_target_met,'layer'].tolist()}.")
        lines.append(f"  Maximum path-length refinement change: {f.max_path_length_relative_change.max():.2%}; "
                     f"maximum scalar-profile refinement change: {f.max_common_profile_relative_change.max():.2%}.")
    lines += ['','Unstable factors are retained and flagged. Numerical path length can be substantially '
              'more stable than individual forcing/gain factors; neither endpoint agreement nor stable '
              'angular speed makes those factors independently reliable.',
              f"Total CPU analysis elapsed across the representative and extension sessions: "
              f"{resources['analysis_elapsed_seconds']/60:.2f} min; peak RAM "
              f"{resources['analysis_peak_ram_bytes']/2**30:.2f} GiB. GPU use for analysis: none.",
              'No work beyond 64 substeps or beyond the specified activation grid was launched.']
    summary=pd.read_csv(out/'layer_summary.csv')
    lines+=['','## Representative endpoint geometry','',
            '| Task | Layer | Integrated S (rad) | Integrated Θ (rad) | Θ/S | Reference Θ (rad) |',
            '|---|---:|---:|---:|---:|---:|']
    for _,row in summary[summary.layer.isin([6,16,28])].sort_values(['target','layer']).iterrows():
        lines.append(f"| {row.target} | {int(row.layer)} | {row.endpoint_S:.4f} | "
                     f"{row.endpoint_Theta:.4f} | {row.endpoint_Theta_over_S:.4f} | {row.endpoint_reference_Theta:.4f} |")
    report=out/'report.md'
    base=report.read_text().split('\n## Final acceptance counts and limitations')[0]
    report.write_text(base+'\n'.join(lines)+'\n')
    (out/'plots/captions.md').write_text('''# Figure scope

All figures describe the sampled, piecewise-linear activation path of the separately labeled
NF4/FP16 series. Numerical integration points are not additional model samples or verified
instantaneous model derivatives. Interpolation boundaries have separate incoming/outgoing
derivatives; sensitivity-profile line segments are deliberately not joined across kinks.

`representative_rotation.png`: stance/style × layers 6/16/28; solid integrated directions,
independently fitted reference dots. `*_all_layer_rotation.png`: supplementary full-depth curves.
`*_direction_error_heatmap.png`: sampled-knot direction accuracy. `*_representative_profiles.png`
and `*_one_sided_heatmaps.png`: probe-gradient perturbation, raw-weight response gain,
perpendicular fraction, raw weight norm/inverse norm, and angular speed along this constructed
path. Gain depends on forcing alignment as well as curvature; raw gain includes scaler conversion.
`endpoint_angle_and_path_length.png`: endpoint angular separation and left-quadrature angular
path length versus depth. Excess path length reflects bending and/or backtracking.
''')
    reconstruction=out/'reconstruction';reconstruction.mkdir(exist_ok=True)
    shutil.copyfile(__file__,reconstruction/Path(__file__).name)
    (out/'README.md').write_text('''# Reproduction

The isolated dependency directory `/tmp/probe-sensitivity-deps` contains scikit-learn 1.6.1,
threadpoolctl 3.6.0 and joblib 1.4.2. Other versions are in run_metadata.json.
From the repository root:

```bash
PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/extract_full_strength_series.py
PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/analyze_full_strength_sensitivity.py --representative-only
cp results/full_strength_nf4_fp16/analysis/analysis_resources.json results/full_strength_nf4_fp16/analysis/representative_analysis_resources.json
PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/analyze_full_strength_sensitivity.py
PYTHONPATH=/tmp/probe-sensitivity-deps:src python -B scripts/summarize_full_strength_results.py
```

GPU extraction requires execution outside Codex's GPU-blocking sandbox. Weights are loaded
offline from the pinned cached revision. The adapter is identified by SHA256. Completed
activation stores and completed CPU cases are reused only when identifiers match.

`references/` stores full-data reference coefficients, raw coordinates, scalers, labels/weights,
and optimality residuals. `integrated/` stores the selected complete no-reset trajectory and
all retained r, beta_dot, raw_w_dot and u_dot vectors. Raw w and b can be reconstructed from
beta and the stored scaler without duplicate coefficient arrays. Both one-sided interval
derivatives are retained. `reference_diagnostics/` is independent of integration and never
feeds corrections into it. `refinements/` preserves scalar outputs from each resolution;
only the selected refinement's large binary arrays are retained. CSV zero-denominator
ratios are NaN and carry explicit flags. No activation arrays or dense Hessians are saved
in the analysis directory.

`reconstruction/` preserves scripts/configuration. Numeric analysis uses the exact validated
objective; reports distinguish convergence screens from scientific significance.
''')
    print(frame.groupby('target')[['direction_target_met','path_length_stable','profiles_stable','all_screens_met']].sum())


if __name__=='__main__': main()
