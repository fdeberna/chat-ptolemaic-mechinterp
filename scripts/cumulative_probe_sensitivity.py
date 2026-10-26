"""Cumulative probe-sensitivity integration along saved activation intervals.

This is CPU-only and uses saved activations at t=0, 0.01, 0.02, and 0.04.
It never loads the LLM or extracts activations.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import time
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

from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import assert_label_mapping_equal
from ptolemaic_mechinterp.probes.dataset import encode_binary_target
from validate_logistic_probe_sensitivity import (
    EXPECTED_LABEL_MAPPINGS,
    NORM_EPS,
    TARGETS,
    analytical_beta_dot,
    cosine_or_nan,
    fit_fixed_scaler_probe,
    gradient_relative_residual,
    relative_error,
    sample_loss_weights,
    strength_dir,
)

SAVED_STRENGTHS = (0.0, 0.01, 0.02, 0.04)
INTERVALS = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.04))
SUBSTEP_COUNTS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument(
        "--activation-root",
        default="results/logistic_probe_sensitivity/activations",
    )
    parser.add_argument(
        "--output-root",
        default="results/cumulative_probe_sensitivity",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["16"],
        help="Layer numbers or ranges like 1-28. Defaults to the safety pass layer 16.",
    )
    parser.add_argument("--all-layers", action="store_true", help="Run layers 1 through 28.")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--substeps", type=int, nargs="+", default=list(SUBSTEP_COUNTS))
    parser.add_argument("--expected-prompts", type=int, default=120)
    parser.add_argument("--fit-tol", type=float, default=1e-8)
    parser.add_argument("--fit-max-iter", type=int, default=5000)
    parser.add_argument("--cg-rtol", type=float, default=1e-9)
    parser.add_argument("--cg-atol", type=float, default=1e-11)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    peak_tracker = PeakMemoryTracker()

    probe_config = load_probe_config(args.probe_config)
    activation_root = resolve_repo_path(args.activation_root)
    output_root = resolve_repo_path(args.output_root)
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    for strength in SAVED_STRENGTHS:
        path = strength_dir(activation_root, strength)
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Missing saved activation store for t={strength:g}: {path}")

    layers = list(range(1, 29)) if args.all_layers else parse_layers(args.layers)
    bad_substeps = sorted(set(args.substeps).difference(SUBSTEP_COUNTS))
    if bad_substeps:
        raise ValueError(f"Supported substep counts are {SUBSTEP_COUNTS}; got {bad_substeps}")

    result_path = output_root / "cumulative_probe_sensitivity.csv"
    runtime_path = output_root / "runtime_summary.json"
    existing = pd.DataFrame() if args.force else read_existing(result_path)
    rows = existing.to_dict("records")
    completed = completed_runs(existing)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "activation_root": str(activation_root),
        "output_root": str(output_root),
        "saved_strengths": list(SAVED_STRENGTHS),
        "intervals": [list(item) for item in INTERVALS],
        "layers": layers,
        "targets": args.targets,
        "substep_counts": args.substeps,
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
        "objective": {
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
        "memory_note": (
            "Compressed NPZ activation files are decompressed when loaded. The script keeps "
            "only the requested layer matrices after each load and releases the full arrays."
        ),
        "interpretation_note": (
            "This tracks the probe optimum along a constructed piecewise-linear activation "
            "path. Substep refinement tests integration of that path, not the actual model "
            "trajectory between saved strengths."
        ),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for layer in layers:
        print(f"cumulative probe sensitivity: layer={layer}")
        layer_data = load_saved_layer_data(
            activation_root=activation_root,
            layer=layer,
            expected_prompts=args.expected_prompts,
        )
        peak_tracker.update()
        try:
            for target in args.targets:
                y01, label_mapping = labels_for_layer(layer_data, target)
                if label_mapping != EXPECTED_LABEL_MAPPINGS[target]:
                    raise ValueError(f"Unexpected label mapping for {target}: {label_mapping}")
                x0 = layer_data[0.0]["x"]
                scaler = StandardScaler().fit(x0)
                z = {
                    strength: scaler.transform(layer_data[strength]["x"]).astype(np.float64)
                    for strength in SAVED_STRENGTHS
                }
                references = {
                    strength: fit_fixed_scaler_probe(
                        z[strength],
                        y01,
                        scaler=scaler,
                        max_iter=args.fit_max_iter,
                        random_seed=probe_config.random_seed,
                        class_weight=probe_config.class_weight,
                        tol=args.fit_tol,
                    )
                    for strength in SAVED_STRENGTHS
                }
                peak_tracker.update()
                for substeps in args.substeps:
                    run_key = (target, layer, substeps)
                    if run_key in completed and not args.force:
                        continue
                    print(f"  target={target} substeps={substeps}")
                    run_rows = integrate_and_measure(
                        target=target,
                        layer=layer,
                        substeps=substeps,
                        z=z,
                        y01=y01,
                        scaler=scaler,
                        references=references,
                        class_weight=probe_config.class_weight,
                        cg_rtol=args.cg_rtol,
                        cg_atol=args.cg_atol,
                        cg_maxiter=args.cg_maxiter,
                    )
                    rows = [
                        row
                        for row in rows
                        if not (
                            row["target"] == target
                            and int(row["layer"]) == layer
                            and int(row["substeps_per_interval"]) == substeps
                        )
                    ]
                    rows.extend(run_rows)
                    write_results(result_path, rows)
                    peak_tracker.update()
        finally:
            del layer_data
            gc.collect()
            peak_tracker.update()

    results = pd.DataFrame(rows)
    make_final_error_plot(results, plots_root / "final_direction_error_by_layer.png")
    make_final_relative_error_plot(
        results,
        plots_root / "final_relative_direction_error_by_layer.png",
    )
    write_report(output_root / "cumulative_probe_sensitivity_report.md", results=results)
    runtime = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_ram_bytes": peak_tracker.peak_bytes,
        "peak_process_ram_mb": peak_tracker.peak_bytes / (1024**2),
        "current_process_ram_bytes": peak_tracker.current_bytes,
        "current_process_ram_mb": peak_tracker.current_bytes / (1024**2),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    print("cumulative_probe_sensitivity_complete")
    print(f"results: {output_root}")
    print(
        f"elapsed_seconds={runtime['elapsed_seconds']:.3f} "
        f"peak_process_ram_mb={runtime['peak_process_ram_mb']:.1f}"
    )


def load_saved_layer_data(
    *,
    activation_root: Path,
    layer: int,
    expected_prompts: int,
) -> dict[float, dict[str, Any]]:
    data: dict[float, dict[str, Any]] = {}
    reference_prompt_ids: list[str] | None = None
    for strength in SAVED_STRENGTHS:
        path = strength_dir(activation_root, strength)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        metadata = pd.read_csv(path / manifest.get("metadata_file", "metadata.csv"))
        layer_mask = metadata["layer_index"].to_numpy() == layer
        layer_metadata = metadata.loc[layer_mask].reset_index(drop=True)
        if len(layer_metadata) != expected_prompts:
            raise ValueError(
                f"Expected {expected_prompts} prompts for t={strength:g} layer {layer}; "
                f"found {len(layer_metadata)}."
            )
        prompt_ids = layer_metadata["prompt_id"].astype(str).tolist()
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif prompt_ids != reference_prompt_ids:
            raise ValueError(f"Prompt order differs at t={strength:g} layer {layer}.")
        with np.load(path / manifest.get("activation_file", "activations.npz")) as arrays:
            activations = arrays["activations"]
            if activations.shape[0] != len(metadata):
                raise ValueError(
                    f"Activation rows and metadata rows differ for t={strength:g}: "
                    f"{activations.shape[0]} vs {len(metadata)}."
                )
            x_layer = np.asarray(activations[layer_mask], dtype=np.float64).copy()
        data[strength] = {"x": x_layer, "metadata": layer_metadata}
        del metadata
        gc.collect()
    return data


def labels_for_layer(
    layer_data: dict[float, dict[str, Any]],
    target: str,
) -> tuple[np.ndarray, dict[str, int]]:
    y_ref, mapping_ref = encode_binary_target(layer_data[0.0]["metadata"], target)
    for strength in SAVED_STRENGTHS[1:]:
        y_other, mapping_other = encode_binary_target(layer_data[strength]["metadata"], target)
        assert_label_mapping_equal(mapping_ref, mapping_other, target=target)
        if not np.array_equal(y_ref, y_other):
            raise ValueError(f"Labels differ for {target} at t={strength:g}.")
    return y_ref, mapping_ref


def integrate_and_measure(
    *,
    target: str,
    layer: int,
    substeps: int,
    z: dict[float, np.ndarray],
    y01: np.ndarray,
    scaler: StandardScaler,
    references: dict[float, Any],
    class_weight: str | None,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> list[dict[str, Any]]:
    sample_weights = sample_loss_weights(y01, class_weight)
    beta = references[0.0].beta.copy()
    rows = [
        measurement_row(
            target=target,
            layer=layer,
            substeps=substeps,
            strength=0.0,
            beta=beta,
            scaler=scaler,
            z_t=z[0.0],
            y01=y01,
            reference=references[0.0],
            reference_u0=references[0.0].unit_w,
            reference_w0=references[0.0].raw_w,
            sample_weights=sample_weights,
            cg_info_max=0,
            cg_failure_count=0,
            solve_relative_residual_max=0.0,
            solve_relative_residual_last=0.0,
            integration_steps_completed=0,
        )
    ]
    cg_info_max = 0
    cg_failure_count = 0
    solve_relative_residual_max = 0.0
    solve_relative_residual_last = 0.0
    steps_completed = 0

    for start, end in INTERVALS:
        z_start = z[start]
        dz_dt = (z[end] - z_start) / (end - start)
        dt = (end - start) / substeps
        for step_index in range(substeps):
            alpha = step_index / substeps
            z_current = z_start + alpha * (z[end] - z_start)
            beta_dot, solve_info = analytical_beta_dot(
                z0=z_current,
                y01=y01,
                beta0=beta,
                dz_dt=dz_dt,
                sample_weights=sample_weights,
                cg_rtol=cg_rtol,
                cg_atol=cg_atol,
                cg_maxiter=cg_maxiter,
            )
            beta = beta + dt * beta_dot
            steps_completed += 1
            info = int(solve_info["cg_info"])
            cg_info_max = max(cg_info_max, info)
            if info != 0:
                cg_failure_count += 1
            solve_relative_residual_last = float(solve_info["solve_relative_residual"])
            solve_relative_residual_max = max(
                solve_relative_residual_max,
                solve_relative_residual_last,
            )
        rows.append(
            measurement_row(
                target=target,
                layer=layer,
                substeps=substeps,
                strength=end,
                beta=beta,
                scaler=scaler,
                z_t=z[end],
                y01=y01,
                reference=references[end],
                reference_u0=references[0.0].unit_w,
                reference_w0=references[0.0].raw_w,
                sample_weights=sample_weights,
                cg_info_max=cg_info_max,
                cg_failure_count=cg_failure_count,
                solve_relative_residual_max=solve_relative_residual_max,
                solve_relative_residual_last=solve_relative_residual_last,
                integration_steps_completed=steps_completed,
            )
        )
    return rows


def measurement_row(
    *,
    target: str,
    layer: int,
    substeps: int,
    strength: float,
    beta: np.ndarray,
    scaler: StandardScaler,
    z_t: np.ndarray,
    y01: np.ndarray,
    reference: Any,
    reference_u0: np.ndarray,
    reference_w0: np.ndarray,
    sample_weights: np.ndarray,
    cg_info_max: int,
    cg_failure_count: int,
    solve_relative_residual_max: float,
    solve_relative_residual_last: float,
    integration_steps_completed: int,
) -> dict[str, Any]:
    pred_w = beta[:-1] / scaler.scale_
    pred_u = normalize(pred_w)
    ref_w = reference.raw_w
    ref_u = reference.unit_w
    ref_move = ref_u - reference_u0
    ref_move_norm = float(np.linalg.norm(ref_move))
    raw_ref_norm = float(np.linalg.norm(ref_w))
    predicted_divergence = 1.0 - cosine_or_nan(reference_u0, pred_u)
    observed_divergence = 1.0 - cosine_or_nan(reference_u0, ref_u)
    return {
        "target": target,
        "layer": layer,
        "substeps_per_interval": substeps,
        "strength_t": strength,
        "direct_unit_direction_distance": float(np.linalg.norm(pred_u - ref_u)),
        "direction_error_over_reference_movement": (
            float(np.linalg.norm(pred_u - ref_u) / ref_move_norm)
            if ref_move_norm > NORM_EPS
            else np.nan
        ),
        "reference_movement_norm_from_t0": ref_move_norm,
        "reference_movement_denominator_near_zero": bool(ref_move_norm <= NORM_EPS),
        "relative_raw_weight_error": (
            float(np.linalg.norm(pred_w - ref_w) / raw_ref_norm)
            if raw_ref_norm > NORM_EPS
            else np.nan
        ),
        "raw_weight_error_denominator_near_zero": bool(raw_ref_norm <= NORM_EPS),
        "predicted_one_minus_cos_u0_ut": predicted_divergence,
        "observed_one_minus_cos_u0_ut": observed_divergence,
        "divergence_error": predicted_divergence - observed_divergence
        if np.isfinite(predicted_divergence) and np.isfinite(observed_divergence)
        else np.nan,
        "predicted_gradient_relative_residual": gradient_relative_residual(
            z=z_t,
            y01=y01,
            beta=beta,
            sample_weights=sample_weights,
        ),
        "reference_gradient_relative_residual": reference.gradient_relative_residual,
        "reference_raw_norm": reference.raw_norm,
        "predicted_raw_norm": float(np.linalg.norm(pred_w)),
        "cg_info_max": cg_info_max,
        "cg_failure_count": cg_failure_count,
        "solve_relative_residual_max": solve_relative_residual_max,
        "solve_relative_residual_last": solve_relative_residual_last,
        "integration_steps_completed": integration_steps_completed,
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= NORM_EPS:
        return np.full_like(vector, np.nan)
    return vector / norm


def parse_layers(values: list[str]) -> list[int]:
    layers: set[int] = set()
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(value))
    invalid = sorted(layer for layer in layers if layer < 1 or layer > 28)
    if invalid:
        raise ValueError(f"Layers must be in 1..28; got {invalid}")
    return sorted(layers)


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def completed_runs(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    if frame.empty:
        return set()
    needed = set(SAVED_STRENGTHS)
    done: set[tuple[str, int, int]] = set()
    for key, group in frame.groupby(["target", "layer", "substeps_per_interval"], sort=False):
        strengths = {round(float(item), 8) for item in group["strength_t"]}
        if needed.issubset(strengths):
            done.add((str(key[0]), int(key[1]), int(key[2])))
    return done


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).sort_values(
        ["target", "layer", "substeps_per_interval", "strength_t"]
    ).to_csv(path, index=False)


def make_final_error_plot(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return
    final = results[np.isclose(results["strength_t"].astype(float), 0.04)].copy()
    if final.empty:
        return
    plt.figure(figsize=(10, 5.5))
    styles = series_styles(final)
    for (target, substeps), group in final.sort_values("layer").groupby(
        ["target", "substeps_per_interval"],
        sort=True,
    ):
        style = styles[(target, int(substeps))]
        plt.plot(
            group["layer"],
            group["direct_unit_direction_distance"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"{target} {int(substeps)} substeps",
        )
    plt.xlabel("Layer")
    plt.ylabel("Final unit-direction distance at t=0.04")
    plt.title("Cumulative probe-sensitivity integration error")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def make_final_relative_error_plot(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return
    final = results[np.isclose(results["strength_t"].astype(float), 0.04)].copy()
    if final.empty:
        return
    plt.figure(figsize=(10, 5.5))
    styles = series_styles(final)
    for (target, substeps), group in final.sort_values("layer").groupby(
        ["target", "substeps_per_interval"],
        sort=True,
    ):
        style = styles[(target, int(substeps))]
        plt.plot(
            group["layer"],
            group["direction_error_over_reference_movement"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"{target} {int(substeps)} substeps",
        )
    plt.axhline(0.02, color="black", linewidth=1, linestyle=":", label="2% target")
    plt.xlabel("Layer")
    plt.ylabel("Final direction error / actual movement")
    plt.title("Cumulative integration relative direction error at t=0.04")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def series_styles(results: pd.DataFrame) -> dict[tuple[str, int], dict[str, str]]:
    """Return color-blind friendly styles with redundant marker/line encodings."""

    colors = {
        "stance": "#0072B2",
        "style": "#D55E00",
    }
    markers = {
        1: "o",
        2: "s",
        4: "^",
        8: "D",
    }
    linestyles = {
        1: "-",
        2: "--",
        4: "-.",
        8: ":",
    }
    styles: dict[tuple[str, int], dict[str, str]] = {}
    for target in sorted(results["target"].dropna().unique()):
        for substeps in sorted(int(item) for item in results["substeps_per_interval"].unique()):
            styles[(str(target), substeps)] = {
                "color": colors.get(str(target), "#000000"),
                "marker": markers.get(substeps, "o"),
                "linestyle": linestyles.get(substeps, "-"),
            }
    return styles


def write_report(path: Path, *, results: pd.DataFrame) -> None:
    lines = [
        "# Cumulative Probe-Sensitivity Integration",
        "",
        "This tests numerical tracking of the probe optimum along the constructed "
        "piecewise-linear activation path through saved strengths 0, 0.01, 0.02, and 0.04.",
        "",
        "## Final t=0.04 Error",
        "",
    ]
    final = results[np.isclose(results["strength_t"].astype(float), 0.04)].copy()
    if final.empty:
        lines.append("- No final-strength rows are available.")
    else:
        for target, group in final.groupby("target", sort=True):
            by_substep = group.groupby("substeps_per_interval")
            medians = by_substep["direct_unit_direction_distance"].median()
            relative_medians = by_substep["direction_error_over_reference_movement"].median()
            relative_max = by_substep["direction_error_over_reference_movement"].max()
            worst = group.sort_values(
                "direction_error_over_reference_movement",
                ascending=False,
            ).head(5)
            lines.append(
                f"- {target}: median final direction distance by substeps: "
                + ", ".join(f"{int(k)}={v:.4g}" for k, v in medians.items())
                + "."
            )
            lines.append(
                "  Median/max final relative direction error by substeps: "
                + ", ".join(
                    f"{int(k)}={relative_medians.loc[k]:.4g}/{relative_max.loc[k]:.4g}"
                    for k in relative_medians.index
                )
                + "."
            )
            lines.append(
                "  Worst current rows: "
                + ", ".join(
                    f"L{int(row.layer)} {int(row.substeps_per_interval)}x="
                    f"{float(row.direction_error_over_reference_movement):.4g}"
                    for row in worst.itertuples()
                )
                + "."
            )
            ordered = sorted(int(v) for v in medians.index)
            if len(ordered) >= 2:
                values = [float(relative_medians.loc[item]) for item in ordered]
                trend = "decrease" if all(b < a for a, b in zip(values, values[1:])) else (
                    "do not monotonically decrease"
                )
                lines.append(f"  Median relative errors {trend} with refinement on completed rows.")
            failures = int(group["cg_failure_count"].max())
            solve_max = float(group["solve_relative_residual_max"].max())
            near_zero = int(group["reference_movement_denominator_near_zero"].sum())
            lines.append(
                f"  CG failures: {failures}; max solve residual {solve_max:.3e}; "
                f"near-zero movement denominators: {near_zero}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Refining substeps improves integration of this constructed path when errors "
            "decrease. It does not establish the actual model trajectory between saved "
            "strengths and should not be described as an explanation of LoRA-induced rotation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PeakMemoryTracker:
    def __init__(self) -> None:
        self.current_bytes = 0
        self.peak_bytes = 0
        self.update()

    def update(self) -> None:
        current, peak = process_memory_bytes()
        self.current_bytes = current
        self.peak_bytes = max(self.peak_bytes, peak, current)


def process_memory_bytes() -> tuple[int, int]:
    if os.name == "nt":
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        return int(info.rss), int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return 0, 0


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


if __name__ == "__main__":
    main()
