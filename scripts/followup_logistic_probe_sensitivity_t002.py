"""CPU-only follow-up: refresh logistic-probe sensitivity at t=0.02.

This script uses saved activations only. It never loads the LLM and refuses to run
if the t=0, t=0.02, and t=0.04 activation stores are missing.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import (
    assert_label_mapping_equal,
    validate_activation_store_pair,
    validate_factor_balance,
)
from validate_logistic_probe_sensitivity import (
    EXPECTED_LABEL_MAPPINGS,
    NORM_EPS,
    TARGETS,
    analytical_beta_dot,
    cosine_or_nan,
    displacement_cosine,
    encoded_labels_by_layer,
    fit_fixed_scaler_probe,
    gradient_relative_residual,
    layer_matrix,
    norm_ratio,
    normalize,
    relative_error,
    rms,
    sample_loss_weights,
    strength_dir,
)

STEP = 0.02
REQUIRED_STRENGTHS = (0.0, 0.02, 0.04)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument(
        "--activation-root",
        default="results/logistic_probe_sensitivity/activations",
        help="Directory containing saved t_0p0000, t_0p0200, and t_0p0400 stores.",
    )
    parser.add_argument(
        "--zero-sensitivity-results",
        default="results/logistic_probe_sensitivity/sensitivity_step_results.csv",
        help="Existing zero-based sensitivity CSV used for endpoint comparison.",
    )
    parser.add_argument(
        "--output-root",
        default="results/logistic_probe_sensitivity_t002_refresh",
    )
    parser.add_argument("--expected-prompts", type=int, default=120)
    parser.add_argument("--cg-rtol", type=float, default=1e-9)
    parser.add_argument("--cg-atol", type=float, default=1e-11)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument(
        "--fit-tol",
        type=float,
        default=1e-8,
        help="Primary liblinear tolerance for the refreshed analysis.",
    )
    parser.add_argument("--fit-max-iter", type=int, default=5000)
    parser.add_argument(
        "--check-tol",
        type=float,
        default=1e-10,
        help="Stricter liblinear tolerance used only for optimizer sensitivity checks.",
    )
    parser.add_argument("--check-max-iter", type=int, default=10000)
    parser.add_argument("--force", action="store_true", help="Recompute existing layer rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_config = load_probe_config(args.probe_config)
    activation_root = resolve_repo_path(args.activation_root)
    output_root = resolve_repo_path(args.output_root)
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    stores = load_required_stores(activation_root)
    for strength in REQUIRED_STRENGTHS[1:]:
        validate_activation_store_pair(
            stores[0.0],
            stores[strength],
            expected_prompt_count=args.expected_prompts,
        )
    balance = validate_factor_balance(stores[0.0].metadata)

    zero_endpoint = load_zero_endpoint_comparison(resolve_repo_path(args.zero_sensitivity_results))
    result_path = output_root / "refreshed_t002_step_results.csv"
    probe_path = output_root / "probe_fit_diagnostics.csv"
    tight_path = output_root / "optimizer_tolerance_diagnostics.csv"

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "activation_root": str(activation_root),
        "output_root": str(output_root),
        "strengths": list(REQUIRED_STRENGTHS),
        "targets": list(TARGETS),
        "layer_0_excluded": True,
        "threads": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "logistic_objective": {
            "solver": "liblinear",
            "penalty": "l2",
            "C": 1.0,
            "fit_intercept": True,
            "intercept_penalized": True,
            "intercept_scaling": 1.0,
            "class_weight": probe_config.class_weight,
            "sample_weight": None,
            "tol": args.fit_tol,
            "max_iter": args.fit_max_iter,
            "random_state": probe_config.random_seed,
        },
        "cg": {"rtol": args.cg_rtol, "atol": args.cg_atol, "maxiter": args.cg_maxiter},
        "strict_fit_check": {"tol": args.check_tol, "max_iter": args.check_max_iter},
        "activation_derivative": "dz = (z(0.04) - z(0.02)) / 0.02 in fixed t=0 standardized coordinates",
        "balance": balance,
        "interpretation_note": (
            "Uses observed activation change over [0.02, 0.04]; this tests the "
            "linearized probe response to that change, not an independently predicted "
            "instantaneous activation derivative or cumulative integration."
        ),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    existing_results = pd.DataFrame() if args.force else read_existing(result_path)
    existing_keys = processed_keys(existing_results)
    result_rows = existing_results.to_dict("records")
    probe_rows = [] if args.force else read_existing(probe_path).to_dict("records")
    tight_rows = [] if args.force else read_existing(tight_path).to_dict("records")

    layers = [
        int(layer)
        for layer in sorted(stores[0.0].metadata["layer_index"].unique())
        if int(layer) != 0
    ]
    for target in TARGETS:
        print(f"refresh t=0.02 sensitivity: target={target}")
        y_by_layer, label_mapping = encoded_labels_by_layer(stores[0.0], target)
        for strength in REQUIRED_STRENGTHS[1:]:
            other_y_by_layer, other_mapping = encoded_labels_by_layer(stores[strength], target)
            assert_label_mapping_equal(label_mapping, other_mapping, target=target)
            for layer in layers:
                if not np.array_equal(y_by_layer[layer], other_y_by_layer[layer]):
                    raise ValueError(f"Labels differ for {target} layer {layer} at t={strength:g}.")
        if label_mapping != EXPECTED_LABEL_MAPPINGS[target]:
            raise ValueError(f"Unexpected label mapping for {target}: {label_mapping}")

        for layer in layers:
            key = (target, layer)
            if key in existing_keys and not args.force:
                continue
            print(f"  layer={layer}")
            rows = analyze_layer(
                stores=stores,
                target=target,
                layer=layer,
                y01=y_by_layer[layer],
                class_weight=probe_config.class_weight,
                max_iter=args.fit_max_iter,
                random_seed=probe_config.random_seed,
                fit_tol=args.fit_tol,
                cg_rtol=args.cg_rtol,
                cg_atol=args.cg_atol,
                cg_maxiter=args.cg_maxiter,
                check_tol=args.check_tol,
                check_max_iter=args.check_max_iter,
                zero_endpoint=zero_endpoint,
            )
            result_rows.append(rows["result"])
            probe_rows.extend(rows["probes"])
            tight_rows.extend(rows["tight"])
            write_csv_incremental(result_path, result_rows)
            write_csv_incremental(probe_path, probe_rows)
            write_csv_incremental(tight_path, tight_rows)

    results = pd.DataFrame(result_rows)
    probes = pd.DataFrame(probe_rows)
    tight = pd.DataFrame(tight_rows)
    make_error_plot(results, plots_root / "refreshed_vs_zero_endpoint_error_by_layer.png")
    write_report(
        output_root / "t002_refresh_report.md",
        results=results,
        probes=probes,
        tight=tight,
        metadata=metadata,
    )
    print("t002_refresh_followup_complete")
    print(f"results: {output_root}")


def load_required_stores(activation_root: Path) -> dict[float, Any]:
    stores = {}
    missing = []
    for strength in REQUIRED_STRENGTHS:
        path = strength_dir(activation_root, strength)
        if not (path / "manifest.json").exists():
            missing.append(str(path))
            continue
        stores[strength] = load_activation_store(path)
    if missing:
        raise FileNotFoundError(
            "Missing required saved activation stores; refusing to extract activations: "
            + ", ".join(missing)
        )
    return stores


def analyze_layer(
    *,
    stores: dict[float, Any],
    target: str,
    layer: int,
    y01: np.ndarray,
    class_weight: str | None,
    max_iter: int,
    random_seed: int,
    fit_tol: float,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
    check_tol: float,
    check_max_iter: int,
    zero_endpoint: pd.DataFrame,
) -> dict[str, Any]:
    x0 = layer_matrix(stores[0.0], layer)
    scaler = StandardScaler().fit(x0)
    z = {
        strength: scaler.transform(layer_matrix(stores[strength], layer).astype(np.float64))
        for strength in REQUIRED_STRENGTHS
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
            tol=fit_tol,
        )
        for strength in REQUIRED_STRENGTHS
    }
    dz = (z[0.04] - z[0.02]) / STEP
    beta_dot, solve_info = analytical_beta_dot(
        z0=z[0.02],
        y01=y01,
        beta0=probes[0.02].beta,
        dz_dt=dz,
        sample_weights=class_weights,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    w_dot = beta_dot[:-1] / scaler.scale_
    u02 = probes[0.02].unit_w
    w02 = probes[0.02].raw_w
    u_dot = (w_dot - u02 * float(u02 @ w_dot)) / probes[0.02].raw_norm
    result = compare_refreshed_prediction(
        target=target,
        layer=layer,
        w02=w02,
        u02=u02,
        w04=probes[0.04].raw_w,
        u04=probes[0.04].unit_w,
        w_dot=w_dot,
        u_dot=u_dot,
        dz=dz,
        solve_info=solve_info,
        zero_endpoint=zero_endpoint,
    )
    probe_rows = [
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
        for strength, probe in sorted(probes.items())
    ]
    tight_rows = tolerance_rows(
        target=target,
        layer=layer,
        z=z,
        y01=y01,
        scaler=scaler,
        class_weight=class_weight,
        random_seed=random_seed,
        default_probes=probes,
        fit_tol=fit_tol,
        check_tol=check_tol,
        check_max_iter=check_max_iter,
    )
    return {"result": result, "probes": probe_rows, "tight": tight_rows}


def compare_refreshed_prediction(
    *,
    target: str,
    layer: int,
    w02: np.ndarray,
    u02: np.ndarray,
    w04: np.ndarray,
    u04: np.ndarray,
    w_dot: np.ndarray,
    u_dot: np.ndarray,
    dz: np.ndarray,
    solve_info: dict[str, Any],
    zero_endpoint: pd.DataFrame,
) -> dict[str, Any]:
    measured_delta_w = w04 - w02
    predicted_delta_w = STEP * w_dot
    measured_delta_u = u04 - u02
    predicted_delta_u = STEP * u_dot
    refreshed_u04 = normalize(w02 + predicted_delta_w)
    zero_row = zero_endpoint[
        (zero_endpoint["target"] == target)
        & (zero_endpoint["layer"].astype(int) == layer)
        & (zero_endpoint["derivative_estimate"] == "forward_0p02")
        & np.isclose(zero_endpoint["strength_t"].astype(float), 0.04)
    ]
    if len(zero_row) == 1:
        zero_distance = float(zero_row.iloc[0]["normalized_finite_step_u_distance"])
        zero_cosine = float(zero_row.iloc[0]["normalized_finite_step_u_cosine"])
        zero_relative_error = float(zero_row.iloc[0]["u_relative_error"])
    else:
        zero_distance = np.nan
        zero_cosine = np.nan
        zero_relative_error = np.nan
    refreshed_distance = float(np.linalg.norm(refreshed_u04 - u04))
    return {
        "target": target,
        "layer": layer,
        "baseline_t": 0.02,
        "endpoint_t": 0.04,
        "step": STEP,
        "dz_dt_norm": float(np.linalg.norm(dz)),
        "dz_dt_rms": rms(dz),
        "w_displacement_cosine": displacement_cosine(predicted_delta_w, measured_delta_w),
        "w_norm_ratio_predicted_over_measured": norm_ratio(
            predicted_delta_w,
            measured_delta_w,
        ),
        "w_relative_error": relative_error(predicted_delta_w, measured_delta_w),
        "w_absolute_error": float(np.linalg.norm(predicted_delta_w - measured_delta_w)),
        "w_measured_delta_norm": float(np.linalg.norm(measured_delta_w)),
        "w_predicted_delta_norm": float(np.linalg.norm(predicted_delta_w)),
        "w_denominator_near_zero": bool(np.linalg.norm(measured_delta_w) <= NORM_EPS),
        "u_displacement_cosine": displacement_cosine(predicted_delta_u, measured_delta_u),
        "u_norm_ratio_predicted_over_measured": norm_ratio(
            predicted_delta_u,
            measured_delta_u,
        ),
        "u_relative_error": relative_error(predicted_delta_u, measured_delta_u),
        "u_absolute_error": float(np.linalg.norm(predicted_delta_u - measured_delta_u)),
        "u_measured_delta_norm": float(np.linalg.norm(measured_delta_u)),
        "u_predicted_delta_norm": float(np.linalg.norm(predicted_delta_u)),
        "u_denominator_near_zero": bool(np.linalg.norm(measured_delta_u) <= NORM_EPS),
        "refreshed_endpoint_u_distance_to_u04": refreshed_distance,
        "refreshed_endpoint_u_cosine_to_u04": cosine_or_nan(refreshed_u04, u04),
        "zero_extrapolated_forward_0p02_endpoint_u_distance_to_u04": zero_distance,
        "zero_extrapolated_forward_0p02_endpoint_u_cosine_to_u04": zero_cosine,
        "zero_extrapolated_forward_0p02_u_relative_error": zero_relative_error,
        "endpoint_distance_improvement": zero_distance - refreshed_distance
        if np.isfinite(zero_distance)
        else np.nan,
        "endpoint_distance_ratio_refreshed_over_zero": refreshed_distance / zero_distance
        if np.isfinite(zero_distance) and zero_distance > NORM_EPS
        else np.nan,
        "u_dot_orthogonality_abs": float(abs(u02 @ u_dot)),
        **solve_info,
    }


def tolerance_rows(
    *,
    target: str,
    layer: int,
    z: dict[float, np.ndarray],
    y01: np.ndarray,
    scaler: StandardScaler,
    class_weight: str | None,
    random_seed: int,
    default_probes: dict[float, Any],
    fit_tol: float,
    check_tol: float,
    check_max_iter: int,
) -> list[dict[str, Any]]:
    strict = {
        strength: fit_fixed_scaler_probe(
            z[strength],
            y01,
            scaler=scaler,
            max_iter=check_max_iter,
            random_seed=random_seed,
            class_weight=class_weight,
            tol=check_tol,
        )
        for strength in (0.02, 0.04)
    }
    rows = []
    for space, attr in [("w", "raw_w"), ("u", "unit_w")]:
        default_delta = getattr(default_probes[0.04], attr) - getattr(default_probes[0.02], attr)
        strict_delta = getattr(strict[0.04], attr) - getattr(strict[0.02], attr)
        rows.append(
            {
                "target": target,
                "layer": layer,
                "space": space,
                "primary_tol": fit_tol,
                "strict_tol": check_tol,
                "primary_delta_norm": float(np.linalg.norm(default_delta)),
                "strict_delta_norm": float(np.linalg.norm(strict_delta)),
                "delta_relative_difference": relative_error(strict_delta, default_delta),
                "delta_absolute_difference": float(np.linalg.norm(strict_delta - default_delta)),
                "delta_cosine": displacement_cosine(strict_delta, default_delta),
                "primary_gradient_residual_t0p02": default_probes[0.02].gradient_relative_residual,
                "primary_gradient_residual_t0p04": default_probes[0.04].gradient_relative_residual,
                "strict_gradient_residual_t0p02": gradient_relative_residual(
                    z=z[0.02],
                    y01=y01,
                    beta=strict[0.02].beta,
                    sample_weights=sample_loss_weights(y01, class_weight),
                ),
                "strict_gradient_residual_t0p04": gradient_relative_residual(
                    z=z[0.04],
                    y01=y01,
                    beta=strict[0.04].beta,
                    sample_weights=sample_loss_weights(y01, class_weight),
                ),
                "strict_convergence_warnings": len(strict[0.02].convergence_warnings)
                + len(strict[0.04].convergence_warnings),
            }
        )
    return rows


def load_zero_endpoint_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Zero-based sensitivity results missing: {path}", stacklevel=2)
        return pd.DataFrame(
            columns=[
                "target",
                "layer",
                "derivative_estimate",
                "strength_t",
                "normalized_finite_step_u_distance",
                "normalized_finite_step_u_cosine",
                "u_relative_error",
            ]
        )
    return pd.read_csv(path)


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def processed_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    if frame.empty:
        return set()
    return {
        (str(row.target), int(row.layer))
        for row in frame[["target", "layer"]].drop_duplicates().itertuples(index=False)
    }


def write_csv_incremental(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).sort_values(["target", "layer"]).to_csv(path, index=False)


def make_error_plot(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return
    plt.figure(figsize=(10, 5.5))
    for target, group in results.sort_values("layer").groupby("target", sort=True):
        plt.plot(
            group["layer"],
            group["refreshed_endpoint_u_distance_to_u04"],
            marker="o",
            label=f"{target} refreshed at t=0.02",
        )
        plt.plot(
            group["layer"],
            group["zero_extrapolated_forward_0p02_endpoint_u_distance_to_u04"],
            marker="s",
            linestyle="--",
            label=f"{target} extrapolated from t=0",
        )
    plt.xlabel("Layer")
    plt.ylabel("Endpoint distance to fitted u(0.04)")
    plt.title("Refreshed local prediction vs zero-based extrapolation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def write_report(
    path: Path,
    *,
    results: pd.DataFrame,
    probes: pd.DataFrame,
    tight: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# t=0.02 Refreshed Logistic-Probe Sensitivity",
        "",
        "This CPU-only follow-up uses saved activations at t=0, t=0.02, and t=0.04. "
        "The scaler is fit once at t=0 and reused at every strength.",
        "",
        "## Numerical Checks",
        "",
        f"- Probe convergence warnings: {int(probes['convergence_warning_count'].sum())}.",
        "- Median gradient relative residual: "
        f"{float(probes['gradient_relative_residual'].median()):.3e}; max "
        f"{float(probes['gradient_relative_residual'].max()):.3e}.",
        "- Median CG solve residual: "
        f"{float(results['solve_relative_residual'].median()):.3e}; max "
        f"{float(results['solve_relative_residual'].max()):.3e}.",
        "- Tight-tolerance median displacement relative difference: "
        f"{float(tight['delta_relative_difference'].median()):.3e}; max "
        f"{float(tight['delta_relative_difference'].max()):.3e}.",
        "",
        "## Accuracy Summary",
        "",
    ]
    for target in TARGETS:
        subset = results[results["target"] == target]
        improved = int((subset["endpoint_distance_improvement"] > 0.0).sum())
        total = int(subset["endpoint_distance_improvement"].notna().sum())
        worst = subset.sort_values("refreshed_endpoint_u_distance_to_u04", ascending=False).head(3)
        lines.append(
            f"- {target}: refreshed median endpoint distance "
            f"{subset['refreshed_endpoint_u_distance_to_u04'].median():.4f}; "
            f"zero-extrapolated median endpoint distance "
            f"{subset['zero_extrapolated_forward_0p02_endpoint_u_distance_to_u04'].median():.4f}; "
            f"improved {improved}/{total} layers."
        )
        lines.append(
            f"  Median u displacement relative error {subset['u_relative_error'].median():.3f}; "
            f"median displacement cosine {subset['u_displacement_cosine'].median():.3f}."
        )
        lines.append(
            "  Worst refreshed endpoint layers: "
            + ", ".join(
                f"L{int(row.layer)}={float(row.refreshed_endpoint_u_distance_to_u04):.4f}"
                for row in worst.itertuples()
            )
            + "."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This uses the observed activation change over [0.02, 0.04], so it tests "
            "the linearized probe response to that change. It does not independently "
            "validate the instantaneous activation derivative, distinguish model curvature "
            "from numerical noise, or validate cumulative integration.",
            "",
            "## Metadata",
            "",
            f"- Thread caps: {metadata['threads']}.",
            f"- CG settings: {metadata['cg']}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
