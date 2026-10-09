"""Validate local implicit-differentiation sensitivity of full-data probes."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ptolemaic_mechinterp.activations.extraction import (
    extract_final_token_hidden_states,
)
from ptolemaic_mechinterp.activations.storage import load_activation_store, save_activation_store
from ptolemaic_mechinterp.config import load_extraction_config, load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.data.prompts import read_prompt_jsonl
from ptolemaic_mechinterp.models.loader import load_model_and_tokenizer
from ptolemaic_mechinterp.probes.comparison import (
    assert_label_mapping_equal,
    validate_activation_store_pair,
    validate_factor_balance,
)
from ptolemaic_mechinterp.probes.dataset import encode_binary_target

TARGETS = ("stance", "style")
EXPECTED_LABEL_MAPPINGS = {
    "stance": {"geocentric": 0, "heliocentric": 1},
    "style": {"modern": 0, "premodern": 1},
}
NORM_EPS = 1e-12
DEFAULT_STRENGTHS = (0.0, 0.01, 0.02, 0.04)
DEFAULT_ENDPOINT_CHECK_STRENGTHS = (1.0,)


@dataclass(frozen=True)
class FixedScalerProbe:
    beta: np.ndarray
    raw_w: np.ndarray
    raw_b: float
    unit_w: np.ndarray
    raw_norm: float
    gradient_relative_residual: float
    n_iter: int
    convergence_warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-config", default="configs/extraction.yaml")
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument("--saved-base-activations", default="results/activations/pilot_full/base")
    parser.add_argument("--saved-lora-activations", default="results/activations/pilot_full/lora")
    parser.add_argument("--output-root", default="results/logistic_probe_sensitivity")
    parser.add_argument("--strengths", type=float, nargs="+", default=list(DEFAULT_STRENGTHS))
    parser.add_argument(
        "--endpoint-check-strengths",
        type=float,
        nargs="*",
        default=list(DEFAULT_ENDPOINT_CHECK_STRENGTHS),
        help="Extra strengths extracted only for endpoint compatibility checks.",
    )
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--cg-rtol", type=float, default=1e-9)
    parser.add_argument("--cg-atol", type=float, default=1e-11)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--tight-tol", type=float, default=1e-8)
    parser.add_argument("--tight-max-iter", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extraction_config = load_extraction_config(args.extraction_config)
    probe_config = load_probe_config(args.probe_config)
    output_root = resolve_repo_path(args.output_root)
    activation_root = output_root / "activations"
    probe_artifact_root = output_root / "probe_artifacts"
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    activation_root.mkdir(parents=True, exist_ok=True)
    probe_artifact_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    saved_base = load_activation_store(resolve_repo_path(args.saved_base_activations))
    saved_lora = load_activation_store(resolve_repo_path(args.saved_lora_activations))
    structure = validate_activation_store_pair(saved_base, saved_lora, expected_prompt_count=120)
    balance = validate_factor_balance(saved_base.metadata)
    validate_factor_balance(saved_lora.metadata)

    selected_prompt_ids = (
        saved_base.metadata.loc[saved_base.metadata["layer_index"] == 0, "prompt_id"]
        .astype(str)
        .tolist()
    )
    prompts = prompts_in_saved_order(extraction_config.prompt_dataset_path, selected_prompt_ids)
    all_extract_strengths = sorted(set(args.strengths).union(args.endpoint_check_strengths))
    tokenizer_check = check_tokenizer_compatibility(extraction_config, prompts)

    stores = extract_or_load_strengths(
        strengths=all_extract_strengths,
        activation_root=activation_root,
        force_extract=args.force_extract,
        extraction_config=extraction_config,
        prompts=prompts,
    )
    endpoint_checks = endpoint_compatibility_checks(
        stores=stores,
        saved_base=saved_base,
        saved_lora=saved_lora,
    )

    analysis_strengths = sorted(args.strengths)
    required = {0.0, 0.01, 0.02, 0.04}
    missing = sorted(required.difference(round_strength(t) for t in analysis_strengths))
    if missing:
        raise ValueError(f"Required pilot strengths are missing: {missing}")
    analysis_stores = {round_strength(t): stores[round_strength(t)] for t in analysis_strengths}
    validate_strength_store_alignment(analysis_stores)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "strengths": analysis_strengths,
        "endpoint_check_strengths": sorted(args.endpoint_check_strengths),
        "output_root": str(output_root),
        "fixed_scaler": "StandardScaler fitted once per target/layer on t=0 activations",
        "logistic_objective": {
            "solver": "liblinear",
            "objective_form": (
                "0.5 * ||[v,c]||_2^2 + C * sum_i class_weight_i * "
                "log(1 + exp(-y_i * [z_i,1]^T [v,c]))"
            ),
            "C": 1.0,
            "lambda_in_reported_formula": 1.0,
            "intercept_penalized": True,
            "intercept_scaling": 1.0,
            "class_weight": probe_config.class_weight,
            "sample_weight": None,
            "tol": 1e-4,
            "max_iter": probe_config.max_iter,
            "random_state": probe_config.random_seed,
        },
        "cg": {"rtol": args.cg_rtol, "atol": args.cg_atol, "maxiter": args.cg_maxiter},
        "structure": structure,
        "balance": balance,
        "tokenizer_check": tokenizer_check,
        "endpoint_checks": endpoint_checks,
        "note": (
            "Activation derivatives are finite differences from new forward passes, not "
            "transformer-Jacobian predictions."
        ),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results, derivative_rows, probe_rows, tight_rows = run_sensitivity_analysis(
        stores=analysis_stores,
        saved_base=saved_base,
        targets=TARGETS,
        max_iter=probe_config.max_iter,
        class_weight=probe_config.class_weight,
        random_seed=probe_config.random_seed,
        cg_rtol=args.cg_rtol,
        cg_atol=args.cg_atol,
        cg_maxiter=args.cg_maxiter,
        tight_tol=args.tight_tol,
        tight_max_iter=args.tight_max_iter,
        probe_artifact_root=probe_artifact_root,
    )
    results_frame = pd.DataFrame(results)
    derivatives_frame = pd.DataFrame(derivative_rows)
    probes_frame = pd.DataFrame(probe_rows)
    tight_frame = pd.DataFrame(tight_rows)
    endpoint_frame = pd.DataFrame(endpoint_checks)
    tokenizer_frame = pd.DataFrame(tokenizer_check)

    results_frame.to_csv(output_root / "sensitivity_step_results.csv", index=False)
    derivatives_frame.to_csv(output_root / "activation_derivative_diagnostics.csv", index=False)
    probes_frame.to_csv(output_root / "probe_fit_diagnostics.csv", index=False)
    tight_frame.to_csv(output_root / "optimizer_tolerance_diagnostics.csv", index=False)
    endpoint_frame.to_csv(output_root / "endpoint_compatibility.csv", index=False)
    tokenizer_frame.to_csv(output_root / "tokenizer_compatibility.csv", index=False)

    make_plots(results_frame, plots_root)
    write_report(
        path=output_root / "logistic_probe_sensitivity_report.md",
        results=results_frame,
        derivatives=derivatives_frame,
        probes=probes_frame,
        tight=tight_frame,
        endpoint=endpoint_frame,
        tokenizer=tokenizer_frame,
        metadata=metadata,
    )
    print("logistic_probe_sensitivity_validation_complete")
    print(f"results: {output_root}")


def extract_or_load_strengths(
    *,
    strengths: list[float],
    activation_root: Path,
    force_extract: bool,
    extraction_config: Any,
    prompts: list[Any],
) -> dict[float, Any]:
    stores: dict[float, Any] = {}
    missing = [
        strength
        for strength in strengths
        if force_extract or not (strength_dir(activation_root, strength) / "manifest.json").exists()
    ]
    if missing:
        loaded = load_model_and_tokenizer(
            model_name_or_path=extraction_config.model.model_name_or_path,
            adapter_path=extraction_config.model.adapter_path,
            device=extraction_config.model.device,
            dtype=extraction_config.model.dtype,
            load_in_4bit=extraction_config.model.load_in_4bit,
            tokenizer_kwargs=extraction_config.model.tokenizer_kwargs,
        )
        if loaded.model.training:
            raise RuntimeError("Loaded model is not in eval mode.")
        for strength in missing:
            print(f"extracting adapter strength t={strength:g}")
            set_lora_strength(loaded.model, strength)
            extracted = extract_final_token_hidden_states(
                loaded.model,
                loaded.tokenizer,
                prompts,
                condition=f"lora_t_{strength:g}",
                batch_size=extraction_config.batch_size,
                device=extraction_config.model.device,
                max_length=extraction_config.max_length,
                model_name_or_path=extraction_config.model.model_name_or_path,
                adapter_path=extraction_config.model.adapter_path,
            )
            output_dir = strength_dir(activation_root, strength)
            save_activation_store(
                output_dir,
                extracted,
                extraction_config=serializable_config(
                    asdict(extraction_config)
                    | {
                        "condition": f"lora_t_{strength:g}",
                        "adapter_strength_t": strength,
                        "adapter_multiplier_definition": (
                            "PEFT LoraLayer.set_scale(adapter, t): trained DeltaW scaling "
                            "multiplied once by t"
                        ),
                        "selected_prompt_ids": [prompt.prompt_id for prompt in prompts],
                    }
                ),
            )
            stores[round_strength(strength)] = load_activation_store(output_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for strength in strengths:
        key = round_strength(strength)
        if key not in stores:
            stores[key] = load_activation_store(strength_dir(activation_root, strength))
    return stores


def run_sensitivity_analysis(
    *,
    stores: dict[float, Any],
    saved_base: Any,
    targets: tuple[str, ...],
    max_iter: int,
    class_weight: str | None,
    random_seed: int,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
    tight_tol: float,
    tight_max_iter: int,
    probe_artifact_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    layers = [
        int(layer)
        for layer in sorted(stores[0.0].metadata["layer_index"].unique())
        if int(layer) != 0
    ]
    results: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    tight_rows: list[dict[str, Any]] = []

    for target in targets:
        print(f"analyzing target={target}")
        y_by_layer, label_mapping = encoded_labels_by_layer(stores[0.0], target)
        saved_y_by_layer, saved_label_mapping = encoded_labels_by_layer(saved_base, target)
        assert_label_mapping_equal(label_mapping, saved_label_mapping, target=target)
        if label_mapping != EXPECTED_LABEL_MAPPINGS[target]:
            raise ValueError(f"Unexpected label mapping for {target}: {label_mapping}")
        for layer in layers:
            y01 = y_by_layer[layer]
            if not np.array_equal(y01, saved_y_by_layer[layer]):
                raise ValueError(f"Saved and new labels differ for {target} layer {layer}.")
            x0_raw = layer_matrix(stores[0.0], layer).astype(np.float64)
            scaler = StandardScaler().fit(x0_raw)
            z = {
                strength: scaler.transform(layer_matrix(store, layer).astype(np.float64))
                for strength, store in stores.items()
                if strength in {0.0, 0.01, 0.02, 0.04}
            }
            class_weights = sample_loss_weights(y01, class_weight)
            probes = {
                strength: fit_fixed_scaler_probe(
                    z[strength],
                    y01,
                    scaler=scaler,
                    max_iter=max_iter,
                    random_seed=random_seed,
                    class_weight=class_weight,
                    tol=1e-4,
                )
                for strength in sorted(z)
            }
            save_probe_artifacts(
                probe_artifact_root=probe_artifact_root,
                target=target,
                layer=layer,
                scaler=scaler,
                probes=probes,
            )
            for strength, probe in probes.items():
                probe_rows.append(
                    {
                        "target": target,
                        "layer": layer,
                        "strength_t": strength,
                        "raw_norm": probe.raw_norm,
                        "gradient_relative_residual": probe.gradient_relative_residual,
                        "n_iter": probe.n_iter,
                        "convergence_warning_count": len(probe.convergence_warnings),
                        "convergence_warnings": json.dumps(probe.convergence_warnings),
                    }
                )
            derivative_estimates = {
                "forward_0p01": (z[0.01] - z[0.0]) / 0.01,
                "forward_0p02": (z[0.02] - z[0.0]) / 0.02,
                "second_order_0p01_0p02": (-3.0 * z[0.0] + 4.0 * z[0.01] - z[0.02])
                / 0.02,
            }
            derivative_rows.extend(
                derivative_diagnostics(target, layer, derivative_estimates, z)
            )
            for derivative_name, dz_dt in derivative_estimates.items():
                beta_dot, solve_info = analytical_beta_dot(
                    z0=z[0.0],
                    y01=y01,
                    beta0=probes[0.0].beta,
                    dz_dt=dz_dt,
                    sample_weights=class_weights,
                    cg_rtol=cg_rtol,
                    cg_atol=cg_atol,
                    cg_maxiter=cg_maxiter,
                )
                w_dot = beta_dot[:-1] / scaler.scale_
                u0 = probes[0.0].unit_w
                w0 = probes[0.0].raw_w
                u_dot = (w_dot - u0 * float(u0 @ w_dot)) / probes[0.0].raw_norm
                orthogonality = float(abs(u0 @ u_dot))
                save_derivative_artifact(
                    probe_artifact_root=probe_artifact_root,
                    target=target,
                    layer=layer,
                    derivative_name=derivative_name,
                    beta_dot=beta_dot,
                    w_dot=w_dot,
                    u_dot=u_dot,
                    solve_info=solve_info,
                    u_dot_orthogonality_abs=orthogonality,
                )
                for strength in [0.01, 0.02, 0.04]:
                    results.append(
                        compare_prediction(
                            target=target,
                            layer=layer,
                            derivative_name=derivative_name,
                            strength=strength,
                            w0=w0,
                            u0=u0,
                            w_t=probes[strength].raw_w,
                            u_t=probes[strength].unit_w,
                            w_dot=w_dot,
                            u_dot=u_dot,
                            solve_info=solve_info,
                            u_dot_orthogonality_abs=orthogonality,
                            uses_same_step_for_activation_derivative=(
                                (derivative_name == "forward_0p01" and strength == 0.01)
                                or (derivative_name == "forward_0p02" and strength == 0.02)
                            ),
                        )
                    )
            tight_rows.extend(
                tolerance_check_rows(
                    target=target,
                    layer=layer,
                    z=z,
                    y01=y01,
                    scaler=scaler,
                    class_weight=class_weight,
                    random_seed=random_seed,
                    default_probes=probes,
                    tight_tol=tight_tol,
                    tight_max_iter=tight_max_iter,
                )
            )
    return results, derivative_rows, probe_rows, tight_rows


def fit_fixed_scaler_probe(
    z_train: np.ndarray,
    y01: np.ndarray,
    *,
    scaler: StandardScaler,
    max_iter: int,
    random_seed: int,
    class_weight: str | None,
    tol: float,
) -> FixedScalerProbe:
    model = LogisticRegression(
        solver="liblinear",
        random_state=random_seed,
        max_iter=max_iter,
        class_weight=class_weight,
        tol=tol,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z_train, y01)
    beta = np.concatenate(
        [np.asarray(model.coef_, dtype=np.float64).reshape(-1), np.asarray(model.intercept_)]
    )
    raw_w = beta[:-1] / scaler.scale_
    raw_b = float(beta[-1] - raw_w @ scaler.mean_)
    raw_norm = float(np.linalg.norm(raw_w))
    unit_w = raw_w / raw_norm if raw_norm > NORM_EPS else np.full_like(raw_w, np.nan)
    grad_rel = gradient_relative_residual(
        z=z_train,
        y01=y01,
        beta=beta,
        sample_weights=sample_loss_weights(y01, class_weight),
    )
    return FixedScalerProbe(
        beta=beta,
        raw_w=raw_w,
        raw_b=raw_b,
        unit_w=unit_w,
        raw_norm=raw_norm,
        gradient_relative_residual=grad_rel,
        n_iter=int(np.max(model.n_iter_)),
        convergence_warnings=[
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ],
    )


def analytical_beta_dot(
    *,
    z0: np.ndarray,
    y01: np.ndarray,
    beta0: np.ndarray,
    dz_dt: np.ndarray,
    sample_weights: np.ndarray,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ypm = y_pm(y01)
    scores = augmented_matvec(z0, beta0)
    q = expit(-ypm * scores)
    beta_dot_input = dz_dt @ beta0[:-1]
    curvature = sample_weights * q * (1.0 - q)
    r_weights = curvature * beta_dot_input
    r = np.concatenate([z0.T @ r_weights, [float(np.sum(r_weights))]])
    r[:-1] -= dz_dt.T @ (sample_weights * ypm * q)

    def hessian_matvec(vector: np.ndarray) -> np.ndarray:
        xv = augmented_matvec(z0, vector)
        hv = np.concatenate([z0.T @ (curvature * xv), [float(np.sum(curvature * xv))]])
        return hv + vector

    operator = LinearOperator((beta0.size, beta0.size), matvec=hessian_matvec, dtype=np.float64)
    beta_dot, info = cg(
        operator,
        -r,
        rtol=cg_rtol,
        atol=cg_atol,
        maxiter=cg_maxiter,
    )
    residual = hessian_matvec(beta_dot) + r
    rel_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(r), NORM_EPS))
    return beta_dot, {"cg_info": int(info), "solve_relative_residual": rel_residual}


def compare_prediction(
    *,
    target: str,
    layer: int,
    derivative_name: str,
    strength: float,
    w0: np.ndarray,
    u0: np.ndarray,
    w_t: np.ndarray,
    u_t: np.ndarray,
    w_dot: np.ndarray,
    u_dot: np.ndarray,
    solve_info: dict[str, Any],
    u_dot_orthogonality_abs: float,
    uses_same_step_for_activation_derivative: bool,
) -> dict[str, Any]:
    delta_w = w_t - w0
    delta_u = u_t - u0
    pred_delta_w = strength * w_dot
    pred_delta_u = strength * u_dot
    u_pred = normalize(w0 + pred_delta_w)
    observed_divergence = 1.0 - cosine_or_nan(u0, u_t)
    predicted_divergence = 0.5 * (strength**2) * float(u_dot @ u_dot)
    return {
        "target": target,
        "layer": layer,
        "derivative_estimate": derivative_name,
        "strength_t": strength,
        "uses_same_step_for_activation_derivative": uses_same_step_for_activation_derivative,
        "w_displacement_cosine": displacement_cosine(pred_delta_w, delta_w),
        "w_norm_ratio_predicted_over_measured": norm_ratio(pred_delta_w, delta_w),
        "w_relative_error": relative_error(pred_delta_w, delta_w),
        "w_absolute_error": float(np.linalg.norm(pred_delta_w - delta_w)),
        "u_displacement_cosine": displacement_cosine(pred_delta_u, delta_u),
        "u_norm_ratio_predicted_over_measured": norm_ratio(pred_delta_u, delta_u),
        "u_relative_error": relative_error(pred_delta_u, delta_u),
        "u_absolute_error": float(np.linalg.norm(pred_delta_u - delta_u)),
        "observed_base_to_t_divergence": observed_divergence,
        "leading_order_predicted_divergence": predicted_divergence,
        "predicted_divergence_over_observed": safe_ratio(
            predicted_divergence,
            observed_divergence,
        ),
        "normalized_finite_step_u_distance": float(np.linalg.norm(u_pred - u_t)),
        "normalized_finite_step_u_cosine": cosine_or_nan(u_pred, u_t),
        "u_dot_orthogonality_abs": u_dot_orthogonality_abs,
        **solve_info,
    }


def tolerance_check_rows(
    *,
    target: str,
    layer: int,
    z: dict[float, np.ndarray],
    y01: np.ndarray,
    scaler: StandardScaler,
    class_weight: str | None,
    random_seed: int,
    default_probes: dict[float, FixedScalerProbe],
    tight_tol: float,
    tight_max_iter: int,
) -> list[dict[str, Any]]:
    rows = []
    tight = {
        strength: fit_fixed_scaler_probe(
            z[strength],
            y01,
            scaler=scaler,
            max_iter=tight_max_iter,
            random_seed=random_seed,
            class_weight=class_weight,
            tol=tight_tol,
        )
        for strength in [0.0, 0.04]
    }
    for space, attr in [("w", "raw_w"), ("u", "unit_w")]:
        default_delta = getattr(default_probes[0.04], attr) - getattr(default_probes[0.0], attr)
        tight_delta = getattr(tight[0.04], attr) - getattr(tight[0.0], attr)
        rows.append(
            {
                "target": target,
                "layer": layer,
                "space": space,
                "default_tol": 1e-4,
                "tight_tol": tight_tol,
                "default_delta_norm": float(np.linalg.norm(default_delta)),
                "tight_delta_norm": float(np.linalg.norm(tight_delta)),
                "delta_relative_difference": relative_error(tight_delta, default_delta),
                "delta_cosine": displacement_cosine(tight_delta, default_delta),
                "tight_gradient_residual_t0": tight[0.0].gradient_relative_residual,
                "tight_gradient_residual_t0p04": tight[0.04].gradient_relative_residual,
                "tight_convergence_warnings": len(tight[0.0].convergence_warnings)
                + len(tight[0.04].convergence_warnings),
            }
        )
    return rows


def derivative_diagnostics(
    target: str,
    layer: int,
    derivative_estimates: dict[str, np.ndarray],
    z: dict[float, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    names = sorted(derivative_estimates)
    for name, estimate in derivative_estimates.items():
        rows.append(
            {
                "target": target,
                "layer": layer,
                "derivative_estimate": name,
                "dz_dt_norm": float(np.linalg.norm(estimate)),
                "mean_abs_dz_dt": float(np.mean(np.abs(estimate))),
                "max_abs_dz_dt": float(np.max(np.abs(estimate))),
                "activation_step_rms_t0_to_t0p01": rms(z[0.01] - z[0.0]),
                "activation_step_rms_t0p01_to_t0p02": rms(z[0.02] - z[0.01]),
            }
        )
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "derivative_estimate": f"{left}_vs_{right}",
                    "dz_dt_norm": np.nan,
                    "mean_abs_dz_dt": np.nan,
                    "max_abs_dz_dt": np.nan,
                    "estimate_cosine": cosine_or_nan(
                        derivative_estimates[left].reshape(-1),
                        derivative_estimates[right].reshape(-1),
                    ),
                    "relative_difference": relative_error(
                        derivative_estimates[left].reshape(-1),
                        derivative_estimates[right].reshape(-1),
                    ),
                    "activation_step_rms_t0_to_t0p01": rms(z[0.01] - z[0.0]),
                    "activation_step_rms_t0p01_to_t0p02": rms(z[0.02] - z[0.01]),
                }
            )
    return rows


def make_plots(results: pd.DataFrame, plots_root: Path) -> None:
    for target in TARGETS:
        for strength in [0.01, 0.02, 0.04]:
            rows = results[
                (results["target"] == target)
                & (results["strength_t"] == strength)
                & (results["derivative_estimate"].isin(["forward_0p01", "second_order_0p01_0p02"]))
            ].sort_values("layer")
            for metric, ylabel in [
                ("u_displacement_cosine", "u displacement alignment"),
                ("u_norm_ratio_predicted_over_measured", "predicted/measured u norm ratio"),
                ("u_relative_error", "u relative displacement error"),
            ]:
                plt.figure(figsize=(9, 4.8))
                for name, group in rows.groupby("derivative_estimate", sort=True):
                    plt.plot(group["layer"], group[metric], marker="o", label=name)
                plt.xlabel("Layer")
                plt.ylabel(ylabel)
                plt.title(f"{target.capitalize()} local prediction at t={strength:g}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    plots_root / f"{target}_t{strength:g}_{metric}_by_layer.png",
                    dpi=170,
                )
                plt.close()
        representative_layers = representative_nonzero_layers(results, target)
        representative = results[
            (results["target"] == target)
            & (results["layer"].isin(representative_layers))
            & (results["derivative_estimate"].isin(["forward_0p01", "forward_0p02"]))
        ].sort_values(["layer", "derivative_estimate", "strength_t"])
        for metric, ylabel in [
            ("u_displacement_cosine", "u displacement alignment"),
            ("u_norm_ratio_predicted_over_measured", "predicted/measured u norm ratio"),
            ("u_relative_error", "u relative displacement error"),
        ]:
            plt.figure(figsize=(9, 5))
            for (layer, name), group in representative.groupby(
                ["layer", "derivative_estimate"],
                sort=True,
            ):
                plt.plot(
                    group["strength_t"],
                    group[metric],
                    marker="o",
                    label=f"L{int(layer)} {name}",
                )
            plt.xlabel("Strength t")
            plt.ylabel(ylabel)
            plt.title(f"{target.capitalize()} representative step-size comparison")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(
                plots_root / f"{target}_representative_steps_{metric}.png",
                dpi=170,
            )
            plt.close()


def representative_nonzero_layers(results: pd.DataFrame, target: str) -> list[int]:
    layers = sorted(
        int(layer)
        for layer in results.loc[results["target"] == target, "layer"].drop_duplicates()
    )
    if len(layers) <= 3:
        return layers
    return [layers[0], layers[len(layers) // 2], layers[-1]]


def save_probe_artifacts(
    *,
    probe_artifact_root: Path,
    target: str,
    layer: int,
    scaler: StandardScaler,
    probes: dict[float, FixedScalerProbe],
) -> None:
    output_dir = probe_artifact_root / target
    output_dir.mkdir(parents=True, exist_ok=True)
    strengths = np.asarray(sorted(probes), dtype=np.float64)
    np.savez_compressed(
        output_dir / f"layer_{layer:03d}_fixed_scaler_probes.npz",
        strengths=strengths,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        beta=np.stack([probes[float(strength)].beta for strength in strengths]),
        raw_w=np.stack([probes[float(strength)].raw_w for strength in strengths]),
        raw_b=np.asarray([probes[float(strength)].raw_b for strength in strengths]),
        unit_w=np.stack([probes[float(strength)].unit_w for strength in strengths]),
        gradient_relative_residual=np.asarray(
            [probes[float(strength)].gradient_relative_residual for strength in strengths]
        ),
    )


def save_derivative_artifact(
    *,
    probe_artifact_root: Path,
    target: str,
    layer: int,
    derivative_name: str,
    beta_dot: np.ndarray,
    w_dot: np.ndarray,
    u_dot: np.ndarray,
    solve_info: dict[str, Any],
    u_dot_orthogonality_abs: float,
) -> None:
    output_dir = probe_artifact_root / target
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = derivative_name.replace(".", "p")
    np.savez_compressed(
        output_dir / f"layer_{layer:03d}_{safe_name}_derivative.npz",
        beta_dot=beta_dot,
        w_dot=w_dot,
        u_dot=u_dot,
        cg_info=np.asarray(solve_info["cg_info"]),
        solve_relative_residual=np.asarray(solve_info["solve_relative_residual"]),
        u_dot_orthogonality_abs=np.asarray(u_dot_orthogonality_abs),
    )


def write_report(
    *,
    path: Path,
    results: pd.DataFrame,
    derivatives: pd.DataFrame,
    probes: pd.DataFrame,
    tight: pd.DataFrame,
    endpoint: pd.DataFrame,
    tokenizer: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Logistic-Probe Local Sensitivity Validation",
        "",
        "This experiment tests activation-change -> optimal-probe-change sensitivity. "
        "Activation derivatives are estimated by finite differences from new forward passes, "
        "not from transformer Jacobians. Agreement validates the local formula implementation; "
        "explaining accumulation with depth remains a subsequent analysis.",
        "",
        "## Setup",
        "",
        f"- Strengths analyzed: {metadata['strengths']}.",
        "- Layer 0 was excluded from direction analysis; the previous scaler diagnostic reported "
        "near-zero layer-0 norms and excluded layer 0 from summary interpretation.",
        "- Fixed scaler: fitted on full-data t=0 activations per target/layer and reused "
        "for all t.",
        "- Logistic objective matched to sklearn/liblinear: penalized intercept, default C=1.0, "
        "and class weights folded into the data-loss weights.",
        "",
        "## Endpoint And Tokenizer Checks",
        "",
    ]
    if not tokenizer.empty:
        mismatch_count = int((~tokenizer["input_ids_match"]).sum())
        lines.append(f"- Base-vs-adapter tokenizer prompt mismatches: {mismatch_count}.")
    for _, row in endpoint.iterrows():
        lines.append(
            f"- t={float(row['strength_t']):g} vs saved {row['saved_condition']}: "
            f"max abs activation diff {float(row['max_abs_diff']):.3e}, "
            f"RMS diff {float(row['rms_diff']):.3e}."
        )
    lines.extend(["", "## Numerical Reliability", ""])
    lines.append(
        f"- Probe convergence warnings: {int(probes['convergence_warning_count'].sum())}."
    )
    lines.append(
        "- Median gradient relative residual: "
        f"{float(probes['gradient_relative_residual'].median()):.3e}; max "
        f"{float(probes['gradient_relative_residual'].max()):.3e}."
    )
    lines.append(
        "- Median CG solve residual: "
        f"{float(results['solve_relative_residual'].median()):.3e}; max "
        f"{float(results['solve_relative_residual'].max()):.3e}."
    )
    lines.append(
        "- Tight-tolerance median displacement relative difference: "
        f"{float(tight['delta_relative_difference'].median()):.3e}; max "
        f"{float(tight['delta_relative_difference'].max()):.3e}."
    )
    lines.extend(["", "## Local Prediction Accuracy", ""])
    for target in TARGETS:
        for derivative_name in ["forward_0p01", "forward_0p02", "second_order_0p01_0p02"]:
            subset = results[
                (results["target"] == target)
                & (results["derivative_estimate"] == derivative_name)
            ]
            holdout = subset[subset["strength_t"] == 0.04]
            lines.append(
                f"- {target} {derivative_name}: median u-displacement cosine "
                f"{subset['u_displacement_cosine'].median():.3f}, median relative error "
                f"{subset['u_relative_error'].median():.3f}; at t=0.04 median relative error "
                f"{holdout['u_relative_error'].median():.3f}."
            )
    lines.extend(["", "## Finite-Step Nonlinearity", ""])
    for target in TARGETS:
        subset = results[
            (results["target"] == target)
            & (results["derivative_estimate"] == "forward_0p01")
        ]
        by_step = subset.groupby("strength_t")["u_relative_error"].median().sort_index()
        lines.append(
            f"- {target}: median u relative error by t using forward_0p01 derivative: "
            + ", ".join(f"{step:g}: {value:.3f}" for step, value in by_step.items())
            + "."
        )
    lines.extend(["", "## Activation-Derivative Stability", ""])
    pairwise = derivatives[derivatives["derivative_estimate"].str.contains("_vs_")]
    for target in TARGETS:
        subset = pairwise[pairwise["target"] == target]
        lines.append(
            f"- {target}: median pairwise derivative-estimate cosine "
            f"{subset['estimate_cosine'].median():.3f}; median relative difference "
            f"{subset['relative_difference'].median():.3f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_lora_strength(model: torch.nn.Module, strength: float) -> None:
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise ImportError("PEFT is required for adapter strength scaling.") from exc
    scaled_layers = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        active_adapters = module.active_adapters
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        for adapter in active_adapters:
            module.set_scale(adapter, float(strength))
            scaled_layers += 1
    if scaled_layers == 0:
        raise RuntimeError("No active LoRA layers were found to scale.")


def endpoint_compatibility_checks(
    *,
    stores: dict[float, Any],
    saved_base: Any,
    saved_lora: Any,
) -> list[dict[str, Any]]:
    rows = []
    for strength, saved_condition, saved_store in [
        (0.0, "base", saved_base),
        (1.0, "lora", saved_lora),
    ]:
        if strength not in stores:
            continue
        validate_activation_store_pair(saved_base, stores[strength], expected_prompt_count=120)
        diff = stores[strength].activations.astype(np.float64) - saved_store.activations.astype(
            np.float64
        )
        rows.append(
            {
                "strength_t": strength,
                "saved_condition": saved_condition,
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "rms_diff": rms(diff),
            }
        )
    return rows


def check_tokenizer_compatibility(
    extraction_config: Any,
    prompts: list[Any],
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        extraction_config.model.model_name_or_path,
        **extraction_config.model.tokenizer_kwargs,
    )
    adapter_tokenizer = AutoTokenizer.from_pretrained(
        extraction_config.model.adapter_path,
        **extraction_config.model.tokenizer_kwargs,
    )
    rows = []
    for prompt in prompts:
        base_ids = base_tokenizer(prompt.text, add_special_tokens=True)["input_ids"]
        adapter_ids = adapter_tokenizer(prompt.text, add_special_tokens=True)["input_ids"]
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "input_ids_match": base_ids == adapter_ids,
                "base_length": len(base_ids),
                "adapter_length": len(adapter_ids),
            }
        )
    return rows


def validate_strength_store_alignment(stores: dict[float, Any]) -> None:
    reference = stores[0.0]
    for strength, store in stores.items():
        if strength == 0.0:
            continue
        validate_activation_store_pair(reference, store, expected_prompt_count=120)


def prompts_in_saved_order(path: Path, prompt_ids: list[str]) -> list[Any]:
    records = read_prompt_jsonl(path)
    by_id = {record.prompt_id: record for record in records}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if missing:
        raise ValueError(f"Saved prompt IDs are missing from prompt dataset: {missing[:5]}")
    return [by_id[prompt_id] for prompt_id in prompt_ids]


def strength_dir(root: Path, strength: float) -> Path:
    return root / f"t_{round_strength(strength):.4f}".replace(".", "p")


def round_strength(strength: float) -> float:
    return round(float(strength), 8)


def layer_matrix(store: Any, layer: int) -> np.ndarray:
    return np.asarray(
        store.activations[store.metadata["layer_index"].to_numpy() == layer],
        dtype=np.float64,
    )


def encoded_labels_by_layer(
    store: Any,
    target: str,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    y_all, mapping = encode_binary_target(store.metadata, target)
    labels = {}
    for layer in sorted(int(item) for item in store.metadata["layer_index"].unique()):
        labels[layer] = y_all[store.metadata["layer_index"].to_numpy() == layer]
    return labels, mapping


def sample_loss_weights(y01: np.ndarray, class_weight: str | None) -> np.ndarray:
    if class_weight is None:
        return np.ones_like(y01, dtype=np.float64)
    if class_weight != "balanced":
        raise ValueError(f"Unsupported class_weight for analytical objective: {class_weight}")
    counts = np.bincount(y01, minlength=2).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"Class weights undefined for counts: {counts.tolist()}")
    return len(y01) / (2.0 * counts[y01])


def y_pm(y01: np.ndarray) -> np.ndarray:
    return np.where(y01 == 1, 1.0, -1.0)


def augmented_matvec(z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return z @ beta[:-1] + beta[-1]


def gradient_relative_residual(
    *,
    z: np.ndarray,
    y01: np.ndarray,
    beta: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    ypm = y_pm(y01)
    q = expit(-ypm * augmented_matvec(z, beta))
    data_grad = np.concatenate(
        [z.T @ (-sample_weights * ypm * q), [float(np.sum(-sample_weights * ypm * q))]]
    )
    grad = data_grad + beta
    denominator = max(np.linalg.norm(data_grad) + np.linalg.norm(beta), NORM_EPS)
    return float(np.linalg.norm(grad) / denominator)


def cosine_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= NORM_EPS or right_norm <= NORM_EPS:
        return float("nan")
    return float(left @ right / (left_norm * right_norm))


def displacement_cosine(predicted: np.ndarray, measured: np.ndarray) -> float:
    if np.linalg.norm(predicted) <= NORM_EPS or np.linalg.norm(measured) <= NORM_EPS:
        return float("nan")
    return cosine_or_nan(predicted, measured)


def norm_ratio(predicted: np.ndarray, measured: np.ndarray) -> float:
    return safe_ratio(float(np.linalg.norm(predicted)), float(np.linalg.norm(measured)))


def relative_error(predicted: np.ndarray, measured: np.ndarray) -> float:
    denom = float(np.linalg.norm(measured))
    if denom <= NORM_EPS:
        return float("nan")
    return float(np.linalg.norm(predicted - measured) / denom)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= NORM_EPS:
        return float("nan")
    return float(numerator / denominator)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= NORM_EPS:
        return np.full_like(vector, np.nan)
    return vector / norm


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def serializable_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable_config(item) for item in value]
    return value


if __name__ == "__main__":
    main()
