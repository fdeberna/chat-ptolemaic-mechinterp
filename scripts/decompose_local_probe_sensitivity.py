"""CPU-only local sensitivity decomposition from saved activations; no model loading."""

from __future__ import annotations

import os
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="probe-decomposition-mpl-"))

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.dataset import encode_binary_target
from validate_logistic_probe_sensitivity import (
    EXPECTED_LABEL_MAPPINGS, NORM_EPS, analytical_beta_dot, fit_fixed_scaler_probe,
    gradient_relative_residual, sample_loss_weights, strength_dir,
)

DEFINITIONS = {
    "D01": "(z(0.01)-z(0))/0.01",
    "D02": "(z(0.02)-z(0))/0.02",
    "D04": "(z(0.04)-z(0))/0.04",
    "Dsecond": "(-3*z(0)+4*z(0.01)-z(0.02))/0.02",
}
STRENGTHS = (0., .01, .02, .04)


def load_saved_layer_data(activation_root, layer, expected_prompts):
    """Read one layer at a time, releasing decompressed full arrays immediately."""
    data = {}
    reference = None
    for t in STRENGTHS:
        path = strength_dir(activation_root, t)
        manifest = json.loads((path / "manifest.json").read_text())
        metadata = pd.read_csv(path / manifest.get("metadata_file", "metadata.csv"))
        mask = metadata.layer_index.to_numpy() == layer
        selected = metadata.loc[mask].reset_index(drop=True)
        if len(selected) != expected_prompts:
            raise ValueError(f"Unexpected prompt count at layer {layer}, t={t}")
        ids = selected.prompt_id.astype(str).tolist()
        if reference is None:
            reference = ids
        elif ids != reference:
            raise ValueError("Prompt order mismatch")
        with np.load(path / manifest.get("activation_file", "activations.npz")) as archive:
            full = archive["activations"]
            if len(full) != len(metadata):
                raise ValueError("Activation/metadata row mismatch")
            x = np.asarray(full[mask], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite activations")
        del full
        data[t] = {"x": x, "metadata": selected}
    return data


def norm(x):
    return float(np.linalg.norm(x))


def ratio(a, b):
    return a / b if b > NORM_EPS else np.nan


def forcing(z, beta, d, y, weights):
    """Same partial-gradient expression as analytical_beta_dot, beta held fixed."""
    ypm = np.where(y == 1, 1., -1.)
    q = expit(-ypm * (z @ beta[:-1] + beta[-1]))
    curvature = weights * q * (1 - q)
    rw = curvature * (d @ beta[:-1])
    r = np.concatenate((z.T @ rw, [rw.sum()]))
    r[:-1] -= d.T @ (weights * ypm * q)
    return r


def gradient(z, beta, y, weights):
    ypm = np.where(y == 1, 1., -1.)
    a = -weights * ypm * expit(-ypm * (z @ beta[:-1] + beta[-1]))
    return beta + np.concatenate((z.T @ a, [a.sum()]))


def save_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def comparisons(vectors, target, layer):
    rows = []
    for left, right in itertools.combinations(DEFINITIONS, 2):
        for quantity in ("D", "r", "w_dot", "u_dot"):
            a, b = vectors[left][quantity].ravel(), vectors[right][quantity].ravel()
            na, nb = norm(a), norm(b)
            rows.append(dict(target=target, layer=layer, left=left, right=right,
                             quantity=quantity, left_norm=na, right_norm=nb,
                             left_near_zero=na <= NORM_EPS, right_near_zero=nb <= NORM_EPS,
                             cosine=float(a @ b / (na * nb)) if min(na, nb) > NORM_EPS else np.nan,
                             norm_ratio_left_over_right=ratio(na, nb),
                             relative_difference_denominator_right=ratio(norm(a-b), nb)))
    return rows


def figures_and_report(out, rows, pairs):
    frame, stability = pd.DataFrame(rows), pd.DataFrame(pairs)
    metrics = [("r_norm", "Probe-gradient perturbation norm (standardized)"),
               ("raw_weight_gain", "Raw-weight response gain"),
               ("perpendicular_fraction", "Perpendicular fraction"),
               ("inverse_weight_norm", "Inverse raw weight norm"),
               ("angular_speed", "Angular speed")]
    for metric, label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        for ax, target in zip(axes, ("stance", "style")):
            for name in DEFINITIONS:
                f = frame[(frame.target == target) & (frame.estimate == name)].sort_values("layer")
                ax.plot(f.layer, f[metric], label=name, linewidth=1.2)
            ax.set(xlabel="Layer", ylabel=label, title=target)
            ax.legend(fontsize=7)
        fig.savefig(out / "plots" / f"{metric}.png", dpi=160)
        plt.close(fig)
    lines = ["# Local sensitivity decomposition", "",
             "All calculations use full-data t=0 scalers, float64, C=1, summed weighted loss, "
             "L2 regularization including the intercept, and intercept_scaling=1. "
             "No damping. Layers are processed sequentially with at most two numerical threads.", "",
             "## Numerical accuracy", "",
             f"Maximum starting optimality relative residual: {frame.optimality_relative_residual.max():.3e}.",
             f"Maximum CG relative residual: {frame.solve_relative_residual.max():.3e}; "
             f"nonzero CG statuses: {(frame.cg_info != 0).sum()}.",
             f"Maximum decomposition identity relative error: {frame.identity_relative_error.max():.3e}.",
             f"Maximum finite-difference forcing verification relative error: {frame.forcing_fd_relative_error.max():.3e}.",
             f"Compatible prior fits reused: {frame.drop_duplicates(['target','layer']).fit_source.str.startswith('reused').sum()}/56.",
             "", "## Derivative and response stability", "",
             "Pairwise ratios are left/right; relative differences use the right vector norm. "
             "Near-zero vectors are flagged; no derivative estimates are averaged."]
    for target in ("stance", "style"):
        for quantity in ("D", "r", "w_dot", "u_dot"):
            f = stability[(stability.target == target) & (stability.quantity == quantity)]
            lines.append(f"- {target}, {quantity}: pairwise cosine median/min "
                         f"{f.cosine.median():.3f}/{f.cosine.min():.3f}; right-normalized relative "
                         f"difference median/max {f.relative_difference_denominator_right.median():.3f}/"
                         f"{f.relative_difference_denominator_right.max():.3f}.")
    lines += ["", "## Layer profiles", "",
              "The following late/early ratios compare mean values on layers 20–28 and 1–8 "
              "within each estimate. This is descriptive, conditional on different base scalers at each layer."]
    for target in ("stance", "style"):
        for metric, _ in metrics:
            values = []
            for name in DEFINITIONS:
                f = frame[(frame.target == target) & (frame.estimate == name)]
                values.append((name, ratio(f.loc[f.layer.between(20,28), metric].mean(),
                                           f.loc[f.layer.between(1,8), metric].mean())))
            lines.append(f"- {target}, {metric}: " + "; ".join(f"{n}={v:.3f}" for n,v in values)
                         + ". Direction of depth contrast agrees across estimates: "
                         + str(all(v > 1 for _,v in values) or all(v < 1 for _,v in values)) + ".")
    numerical_ok = (frame.optimality_relative_residual.max() < 1e-6
                    and frame.solve_relative_residual.max() < 1e-8
                    and (frame.cg_info == 0).all())
    angular = stability[stability.quantity == "u_dot"]
    stable = (angular.cosine.min() > .95
              and angular.relative_difference_denominator_right.max() < .2)
    lines += ["", "## Interpretation", "",
              f"Numerical adequacy screen (optimality <1e-6, solves <1e-8, CG success): {numerical_ok}. "
              "These thresholds are numerical screens, not scientific significance tests.",
              f"Angular-response agreement screen (all pair cosines >0.95 and differences <0.2): {stable}.",
              "Instantaneous interpretation is not supported when estimates disagree. Results then remain "
              "responses to the specified finite intervals; even agreement on these four estimates would "
              "not establish convergence as the interval shrinks to zero.",
              "r is a probe-gradient perturbation, not a probe-independent LoRA magnitude. Response gain "
              "depends jointly on curvature and forcing alignment. Perpendicular fraction and inverse "
              "weight norm are separate factors. Norms depend on coordinates and objective normalization.",
              "No causal percentages are assigned to gradient versus Hessian, and no t=1 extrapolation "
              "is made. No new extraction or environment changes are launched."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", default="results/logistic_probe_sensitivity/activations")
    parser.add_argument("--output-root", default="results/local_probe_sensitivity_decomposition")
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    args = parser.parse_args()
    root, out = resolve_repo_path(args.activation_root), resolve_repo_path(args.output_root)
    config = load_probe_config(args.probe_config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    sources = {}
    for t in STRENGTHS:
        p = strength_dir(root, t)
        sources[str(t)] = {"path": str(p), "manifest_sha256": hashlib.sha256((p/"manifest.json").read_bytes()).hexdigest(),
                           "activation_bytes": (p/"activations.npz").stat().st_size}
    objective = dict(C=1., loss_normalization="sum", penalty="l2", intercept_penalized=True,
                     intercept_scaling=1., class_weight=config.class_weight, sample_weight=None,
                     fit_tol=1e-10, fit_max_iter=10000, cg_rtol=1e-9, cg_atol=1e-11,
                     cg_maxiter=2000, damping=None, dtype="float64", random_state=config.random_seed)
    metadata = dict(objective=objective, sources=sources, definitions=DEFINITIONS,
                    near_zero_threshold=NORM_EPS, scaler="full-data t=0 per layer",
                    numerical_threads=2, pairwise_denominator="right vector norm",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python_executable=sys.executable, python_version=sys.version,
                    arguments=vars(args), software_versions={p: importlib.metadata.version(p)
                    for p in ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")},
                    objective_form="0.5*||beta||^2 + C*sum_i loss_weight_i*log(1+exp(-y_i*[z_i,1]@beta))")
    prior_metadata_path = resolve_repo_path("results/logistic_probe_sensitivity/run_metadata.json")
    prior_objective = json.loads(prior_metadata_path.read_text())["logistic_objective"]
    prior_compatible = all(prior_objective.get(key) == objective[key] for key in
                           ("C", "intercept_penalized", "intercept_scaling", "class_weight", "sample_weight"))
    metadata["prior_objective_compatible"] = prior_compatible
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    rows, pairs = [], []
    with threadpool_limits(limits=2):
        metadata["threadpools"] = threadpool_info()
        if any(p["num_threads"] > 2 for p in metadata["threadpools"]):
            raise RuntimeError("Numerical thread cap not respected")
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        for layer in range(1,29):
            data = load_saved_layer_data(activation_root=root, layer=layer, expected_prompts=120)
            scaler = StandardScaler().fit(data[0.]["x"])
            z = {t: scaler.transform(data[t]["x"]).astype(np.float64) for t in STRENGTHS}
            ds = dict(D01=(z[.01]-z[0.])/.01, D02=(z[.02]-z[0.])/.02,
                      D04=(z[.04]-z[0.])/.04, Dsecond=(-3*z[0.]+4*z[.01]-z[.02])/.02)
            for target in ("stance", "style"):
                y, mapping = encode_binary_target(data[0.]["metadata"], target)
                assert mapping == EXPECTED_LABEL_MAPPINGS[target]
                for t in STRENGTHS:
                    yt, mt = encode_binary_target(data[t]["metadata"], target)
                    assert mt == mapping and np.array_equal(y, yt)
                weights = sample_loss_weights(y, config.class_weight)
                prior = resolve_repo_path(f"results/logistic_probe_sensitivity/probe_artifacts/{target}/layer_{layer:03d}_fixed_scaler_probes.npz")
                beta = None
                if prior.exists() and prior_compatible:
                    with np.load(prior) as a:
                        if (np.array_equal(a['scaler_mean'], scaler.mean_) and
                            np.array_equal(a['scaler_scale'], scaler.scale_)):
                            candidate = a['beta'][np.flatnonzero(a['strengths'] == 0.)[0]].copy()
                            if gradient_relative_residual(z=z[0.], y01=y, beta=candidate, sample_weights=weights) < 1e-8:
                                beta = candidate
                fit_source = f"reused:{prior}" if beta is not None else "fresh full-data liblinear tol=1e-10"
                warning_count = 0
                if beta is None:
                    fit = fit_fixed_scaler_probe(z[0.], y, scaler=scaler, max_iter=10000,
                                                random_seed=config.random_seed, class_weight=config.class_weight, tol=1e-10)
                    beta = fit.beta
                    warning_count = len(fit.convergence_warnings)
                residual = gradient_relative_residual(z=z[0.], y01=y, beta=beta, sample_weights=weights)
                w = beta[:-1]/scaler.scale_
                wn = norm(w)
                u = w/wn if wn > NORM_EPS else np.full_like(w,np.nan)
                vectors, payload = {}, dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                    labels=y, loss_weights=weights, beta=beta, raw_w=w, u=u,
                    prompt_ids=data[0.]['metadata'].prompt_id.astype(str).to_numpy(dtype=str))
                artifact_meta = dict(metadata, target=target, layer=layer, label_mapping=mapping,
                                     fit_source=fit_source, optimality_relative_residual=residual)
                for name, d in ds.items():
                    r = forcing(z[0.], beta, d, y, weights)
                    bd, info = analytical_beta_dot(z0=z[0.], y01=y, beta0=beta, dz_dt=d,
                                                  sample_weights=weights, cg_rtol=1e-9, cg_atol=1e-11, cg_maxiter=2000)
                    wd = bd[:-1]/scaler.scale_
                    perp = wd-u*float(u@wd)
                    ud = perp/wn if wn > NORM_EPS else np.full_like(wd,np.nan)
                    rn, wdn, pn, un = norm(r),norm(wd),norm(perp),norm(ud)
                    gain, fraction, inv = ratio(wdn,rn), ratio(pn,wdn), ratio(1.,wn)
                    product = rn*gain*fraction*inv
                    eps = 1e-5/max(1.,norm(d)/np.sqrt(d.size))
                    fd = (gradient(z[0.]+eps*d,beta,y,weights)-gradient(z[0.]-eps*d,beta,y,weights))/(2*eps)
                    row = dict(target=target, layer=layer, estimate=name, D_rows=d.shape[0],D_columns=d.shape[1],
                        D_frobenius_norm=norm(d), D_rms=norm(d)/np.sqrt(d.size),r_norm=rn,
                        standardized_beta_gain=ratio(norm(bd),rn),raw_weight_gain=gain,
                        perpendicular_fraction=fraction,inverse_weight_norm=inv,angular_gain=ratio(un,rn),
                        angular_speed=un,identity_product=product,identity_relative_error=ratio(abs(un-product),un),
                        forcing_fd_relative_error=ratio(norm(fd-r),rn),optimality_relative_residual=residual,
                        fit_source=fit_source,convergence_warning_count=warning_count,
                        r_near_zero=rn<=NORM_EPS,w_dot_near_zero=wdn<=NORM_EPS,w_near_zero=wn<=NORM_EPS,
                        u_dot_near_zero=un<=NORM_EPS,**info)
                    rows.append(row)
                    vectors[name] = dict(D=d,r=r,w_dot=wd,u_dot=ud)
                    for key,value in dict(r=r,beta_dot=bd,raw_w_dot=wd,u_dot=ud,
                                          solve_relative_residual=info['solve_relative_residual'],cg_info=info['cg_info']).items():
                        payload[f"{name}_{key}"] = value
                pairs.extend(comparisons(vectors,target,layer))
                payload['metadata_json'] = np.asarray(json.dumps(artifact_meta,sort_keys=True))
                np.savez_compressed(out/'artifacts'/f'{target}_layer_{layer:03d}.npz',**payload)
                save_csv(out/'decomposition.csv',rows)
                save_csv(out/'pairwise_stability.csv',pairs)
                print(f"layer={layer} target={target} optimality={residual:.3e}",flush=True)
            del data,z,ds,vectors,payload
    figures_and_report(out,rows,pairs)
    print(f"Complete: {out}",flush=True)


if __name__ == "__main__":
    main()
