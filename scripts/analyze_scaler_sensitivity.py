"""Endpoint scaler-sensitivity diagnostic for base-vs-LoRA linear probes."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ptolemaic_mechinterp.activations.storage import ActivationStore, load_activation_store
from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import (
    assert_label_mapping_equal,
    assert_same_group_splits,
    load_full_data_probe_artifact,
    validate_activation_store_pair,
    validate_factor_balance,
)
from ptolemaic_mechinterp.probes.dataset import encode_binary_target

TARGETS = ("stance", "style")
MODEL_CONDITIONS = ("base", "lora")
EXPECTED_LABEL_MAPPINGS = {
    "stance": {"geocentric": 0, "heliocentric": 1},
    "style": {"modern": 0, "premodern": 1},
}
NORM_EPS = 1e-12
SCORE_RTOL = 1e-10
SCORE_ATOL = 1e-10
ARTIFACT_RTOL = 1e-7
ARTIFACT_ATOL = 5e-6


@dataclass(frozen=True)
class ProbeFit:
    """Fitted logistic probe represented in raw activation coordinates."""

    condition: str
    scaler_source: str
    scaler: StandardScaler
    model: LogisticRegression
    raw_w: np.ndarray
    raw_b: float
    unit_w: np.ndarray
    raw_norm: float
    convergence_warnings: list[str]
    n_iter: int
    constant_features: int
    scale_min: float
    scale_median: float
    scale_max: float
    raw_score_max_abs_diff: float
    raw_score_allclose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument("--base-activations", default="results/activations/pilot_full/base")
    parser.add_argument("--lora-activations", default="results/activations/pilot_full/lora")
    parser.add_argument("--probe-results-root", default="results/probes/pilot_full")
    parser.add_argument("--output-root", default="results/scaler_sensitivity")
    parser.add_argument("--expected-prompts", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_probe_config(args.probe_config)
    base_store = load_activation_store(resolve_repo_path(args.base_activations))
    lora_store = load_activation_store(resolve_repo_path(args.lora_activations))
    structure = validate_activation_store_pair(
        base_store,
        lora_store,
        expected_prompt_count=args.expected_prompts,
    )
    balance = validate_factor_balance(base_store.metadata)
    validate_factor_balance(lora_store.metadata)

    output_root = resolve_repo_path(args.output_root)
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    probe_results_root = resolve_repo_path(args.probe_results_root)
    stores = {"base": base_store, "lora": lora_store}
    layers = sorted(int(layer) for layer in base_store.metadata["layer_index"].unique())

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_activations": str(resolve_repo_path(args.base_activations)),
        "lora_activations": str(resolve_repo_path(args.lora_activations)),
        "probe_results_root": str(probe_results_root),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "layers": layers,
        "probe_hyperparameters": {
            "standardizer": "sklearn.preprocessing.StandardScaler",
            "classifier": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "C": 1.0,
            "penalty": "l2",
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "random_state": config.random_seed,
            "sample_weight": None,
        },
        "structure": structure,
        "balance": balance,
        "score_validation": {"rtol": SCORE_RTOL, "atol": SCORE_ATOL},
        "artifact_comparison": {"rtol": ARTIFACT_RTOL, "atol": ARTIFACT_ATOL},
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    full_rows, fold_rows, check_rows = run_analysis(
        stores=stores,
        layers=layers,
        targets=TARGETS,
        grouping_column=config.grouping_column,
        n_folds=config.n_folds,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        probe_results_root=probe_results_root,
    )

    full_results = pd.DataFrame(full_rows)
    fold_results = pd.DataFrame(fold_rows)
    checks = pd.DataFrame(check_rows)
    full_results.to_csv(output_root / "scaler_sensitivity_full_data.csv", index=False)
    fold_results.to_csv(output_root / "scaler_sensitivity_grouped_cv_folds.csv", index=False)
    aggregate_fold_metrics(fold_results).to_csv(
        output_root / "scaler_sensitivity_grouped_cv_summary.csv",
        index=False,
    )
    checks.to_csv(output_root / "scaler_sensitivity_validation_checks.csv", index=False)

    make_plots(full_results, plots_root)
    write_report(
        output_root / "scaler_sensitivity_report.md",
        full_results=full_results,
        fold_results=fold_results,
        checks=checks,
        metadata=metadata,
    )

    print("scaler_sensitivity_analysis_complete")
    print(f"results: {output_root}")


def run_analysis(
    *,
    stores: dict[str, ActivationStore],
    layers: list[int],
    targets: tuple[str, ...],
    grouping_column: str,
    n_folds: int,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
    probe_results_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    prompt_ids = prompt_ids_for_layer(stores["base"], 0)

    for target in targets:
        print(f"scaler sensitivity: target={target}")
        y_by_layer, label_mapping = encoded_labels_by_layer(stores["base"], target)
        lora_y_by_layer, lora_label_mapping = encoded_labels_by_layer(stores["lora"], target)
        assert_label_mapping_equal(label_mapping, lora_label_mapping, target=target)
        expected = EXPECTED_LABEL_MAPPINGS[target]
        if label_mapping != expected:
            raise ValueError(
                f"Unexpected {target} label orientation: found {label_mapping}, "
                f"expected {expected}."
            )
        for layer in layers:
            if prompt_ids_for_layer(stores["base"], layer) != prompt_ids:
                raise ValueError(f"Base prompt order differs at layer {layer}.")
            if prompt_ids_for_layer(stores["lora"], layer) != prompt_ids:
                raise ValueError(f"LoRA prompt order differs at layer {layer}.")
            y = y_by_layer[layer]
            if not np.array_equal(y, lora_y_by_layer[layer]):
                raise ValueError(f"Base and LoRA labels differ for {target} layer {layer}.")
            x_base = layer_matrix(stores["base"], layer)
            x_lora = layer_matrix(stores["lora"], layer)

            fit_a = fit_probe(
                condition="A_base_base_scaler",
                scaler_source="base",
                x_train=x_base,
                y_train=y,
                x_score_check=x_base,
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=None,
            )
            fit_b = fit_probe(
                condition="B_lora_base_scaler",
                scaler_source="base",
                x_train=x_lora,
                y_train=y,
                x_score_check=x_lora,
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=fit_a.scaler,
            )
            fit_c = fit_probe(
                condition="C_lora_lora_scaler",
                scaler_source="lora",
                x_train=x_lora,
                y_train=y,
                x_score_check=x_lora,
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=None,
            )
            full_rows.append(
                build_full_data_row(
                    target=target,
                    layer=layer,
                    label_mapping=label_mapping,
                    fit_a=fit_a,
                    fit_b=fit_b,
                    fit_c=fit_c,
                    probe_results_root=probe_results_root,
                )
            )
            check_rows.extend(
                build_fit_check_rows(
                    target=target,
                    layer=layer,
                    fits=[fit_a, fit_b, fit_c],
                    probe_results_root=probe_results_root,
                )
            )

        fold_rows.extend(
            run_grouped_cv(
                stores=stores,
                target=target,
                label_mapping=label_mapping,
                grouping_column=grouping_column,
                n_folds=n_folds,
                random_seed=random_seed,
                max_iter=max_iter,
                class_weight=class_weight,
                probe_results_root=probe_results_root,
                layers=layers,
            )
        )
    return full_rows, fold_rows, check_rows


def fit_probe(
    *,
    condition: str,
    scaler_source: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_score_check: np.ndarray,
    max_iter: int,
    class_weight: str | None,
    random_seed: int,
    scaler: StandardScaler | None,
) -> ProbeFit:
    fitted_scaler = scaler if scaler is not None else StandardScaler().fit(x_train)
    x_scaled = fitted_scaler.transform(x_train)
    model = LogisticRegression(
        solver="liblinear",
        random_state=random_seed,
        max_iter=max_iter,
        class_weight=class_weight,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_scaled, y_train)
    convergence_warnings = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    raw_w = np.asarray(model.coef_, dtype=float).reshape(-1) / np.asarray(
        fitted_scaler.scale_,
        dtype=float,
    )
    raw_b = float(
        np.asarray(model.intercept_, dtype=float).reshape(-1)[0]
        - raw_w @ fitted_scaler.mean_
    )
    raw_norm = float(np.linalg.norm(raw_w))
    unit_w = raw_w / raw_norm if raw_norm > NORM_EPS else np.full_like(raw_w, np.nan)
    standardized_scores = model.decision_function(fitted_scaler.transform(x_score_check))
    raw_scores = x_score_check @ raw_w + raw_b
    score_diff = float(np.max(np.abs(raw_scores - standardized_scores)))
    scale = np.asarray(fitted_scaler.scale_, dtype=float)
    return ProbeFit(
        condition=condition,
        scaler_source=scaler_source,
        scaler=fitted_scaler,
        model=model,
        raw_w=raw_w,
        raw_b=raw_b,
        unit_w=unit_w,
        raw_norm=raw_norm,
        convergence_warnings=convergence_warnings,
        n_iter=int(np.max(model.n_iter_)),
        constant_features=int(np.sum(np.asarray(fitted_scaler.var_) <= NORM_EPS)),
        scale_min=float(np.min(scale)),
        scale_median=float(np.median(scale)),
        scale_max=float(np.max(scale)),
        raw_score_max_abs_diff=score_diff,
        raw_score_allclose=bool(
            np.allclose(raw_scores, standardized_scores, rtol=SCORE_RTOL, atol=SCORE_ATOL)
        ),
    )


def build_full_data_row(
    *,
    target: str,
    layer: int,
    label_mapping: dict[str, int],
    fit_a: ProbeFit,
    fit_b: ProbeFit,
    fit_c: ProbeFit,
    probe_results_root: Path,
) -> dict[str, Any]:
    cos_ab = cosine_or_nan(fit_a.unit_w, fit_b.unit_w)
    cos_ac = cosine_or_nan(fit_a.unit_w, fit_c.unit_w)
    cos_bc = cosine_or_nan(fit_b.unit_w, fit_c.unit_w)
    div_ab = 1.0 - cos_ab if np.isfinite(cos_ab) else np.nan
    div_ac = 1.0 - cos_ac if np.isfinite(cos_ac) else np.nan
    div_bc = 1.0 - cos_bc if np.isfinite(cos_bc) else np.nan
    base_artifact = artifact_comparison(probe_results_root, "base", target, layer, fit_a)
    lora_artifact = artifact_comparison(probe_results_root, "lora", target, layer, fit_c)
    return {
        "target": target,
        "layer": int(layer),
        "label_mapping": json.dumps(label_mapping, sort_keys=True),
        "positive_class": positive_class(label_mapping),
        "cos_A_base_base_scaler_vs_B_lora_base_scaler": cos_ab,
        "cos_A_base_base_scaler_vs_C_lora_lora_scaler": cos_ac,
        "cos_B_lora_base_scaler_vs_C_lora_lora_scaler": cos_bc,
        "divergence_A_vs_B_fixed_scaler": div_ab,
        "divergence_A_vs_C_refit_scaler": div_ac,
        "divergence_B_vs_C_lora_scaler_choice": div_bc,
        "divergence_refit_minus_fixed": div_ac - div_ab
        if np.isfinite(div_ac) and np.isfinite(div_ab)
        else np.nan,
        "A_raw_norm": fit_a.raw_norm,
        "B_raw_norm": fit_b.raw_norm,
        "C_raw_norm": fit_c.raw_norm,
        "A_near_zero_norm": fit_a.raw_norm <= NORM_EPS,
        "B_near_zero_norm": fit_b.raw_norm <= NORM_EPS,
        "C_near_zero_norm": fit_c.raw_norm <= NORM_EPS,
        "A_constant_features": fit_a.constant_features,
        "B_constant_features": fit_b.constant_features,
        "C_constant_features": fit_c.constant_features,
        "A_scale_min": fit_a.scale_min,
        "B_scale_min": fit_b.scale_min,
        "C_scale_min": fit_c.scale_min,
        "A_scale_median": fit_a.scale_median,
        "B_scale_median": fit_b.scale_median,
        "C_scale_median": fit_c.scale_median,
        "A_scale_max": fit_a.scale_max,
        "B_scale_max": fit_b.scale_max,
        "C_scale_max": fit_c.scale_max,
        "A_raw_score_max_abs_diff": fit_a.raw_score_max_abs_diff,
        "B_raw_score_max_abs_diff": fit_b.raw_score_max_abs_diff,
        "C_raw_score_max_abs_diff": fit_c.raw_score_max_abs_diff,
        "A_raw_score_allclose": fit_a.raw_score_allclose,
        "B_raw_score_allclose": fit_b.raw_score_allclose,
        "C_raw_score_allclose": fit_c.raw_score_allclose,
        "A_n_iter": fit_a.n_iter,
        "B_n_iter": fit_b.n_iter,
        "C_n_iter": fit_c.n_iter,
        "A_convergence_warning_count": len(fit_a.convergence_warnings),
        "B_convergence_warning_count": len(fit_b.convergence_warnings),
        "C_convergence_warning_count": len(fit_c.convergence_warnings),
        "A_matches_existing_base_artifact": base_artifact["matches"],
        "A_existing_artifact_cosine": base_artifact["cosine"],
        "A_existing_artifact_max_abs_raw_w_diff": base_artifact["max_abs_raw_w_diff"],
        "A_existing_artifact_raw_b_diff": base_artifact["raw_b_diff"],
        "C_matches_existing_lora_artifact": lora_artifact["matches"],
        "C_existing_artifact_cosine": lora_artifact["cosine"],
        "C_existing_artifact_max_abs_raw_w_diff": lora_artifact["max_abs_raw_w_diff"],
        "C_existing_artifact_raw_b_diff": lora_artifact["raw_b_diff"],
    }


def build_fit_check_rows(
    *,
    target: str,
    layer: int,
    fits: list[ProbeFit],
    probe_results_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for fit in fits:
        row = {
            "target": target,
            "layer": int(layer),
            "condition": fit.condition,
            "scaler_source": fit.scaler_source,
            "raw_norm": fit.raw_norm,
            "near_zero_norm": fit.raw_norm <= NORM_EPS,
            "raw_score_max_abs_diff": fit.raw_score_max_abs_diff,
            "raw_score_allclose": fit.raw_score_allclose,
            "constant_features": fit.constant_features,
            "scale_min": fit.scale_min,
            "scale_median": fit.scale_median,
            "scale_max": fit.scale_max,
            "n_iter": fit.n_iter,
            "convergence_warning_count": len(fit.convergence_warnings),
            "convergence_warnings": json.dumps(fit.convergence_warnings),
        }
        if fit.condition == "A_base_base_scaler":
            row.update(
                {f"existing_artifact_{key}": value for key, value in artifact_comparison(
                    probe_results_root,
                    "base",
                    target,
                    layer,
                    fit,
                ).items()}
            )
        elif fit.condition == "C_lora_lora_scaler":
            row.update(
                {f"existing_artifact_{key}": value for key, value in artifact_comparison(
                    probe_results_root,
                    "lora",
                    target,
                    layer,
                    fit,
                ).items()}
            )
        rows.append(row)
    return rows


def artifact_comparison(
    results_root: Path,
    model_name: str,
    target: str,
    layer: int,
    fit: ProbeFit,
) -> dict[str, Any]:
    path = (
        results_root
        / model_name
        / target
        / "full_data_coefficients"
        / f"layer_{layer:03d}_full_data.npz"
    )
    if not path.exists():
        return {
            "available": False,
            "matches": False,
            "cosine": np.nan,
            "max_abs_raw_w_diff": np.nan,
            "raw_b_diff": np.nan,
        }
    artifact = load_full_data_probe_artifact(path)
    raw_w = np.asarray(artifact["raw_direction"], dtype=float).reshape(-1)
    data = np.load(path, allow_pickle=False)
    scaled_intercept = float(np.asarray(data["intercept"], dtype=float).reshape(-1)[0])
    scaler_mean = np.asarray(data["scaler_mean"], dtype=float).reshape(-1)
    artifact_raw_b = scaled_intercept - float(raw_w @ scaler_mean)
    cosine = cosine_or_nan(normalize(raw_w), fit.unit_w)
    max_abs_diff = float(np.max(np.abs(raw_w - fit.raw_w)))
    raw_b_diff = float(abs(artifact_raw_b - fit.raw_b))
    matches = bool(
        np.allclose(raw_w, fit.raw_w, rtol=ARTIFACT_RTOL, atol=ARTIFACT_ATOL)
        and np.isclose(artifact_raw_b, fit.raw_b, rtol=ARTIFACT_RTOL, atol=ARTIFACT_ATOL)
    )
    return {
        "available": True,
        "matches": matches,
        "cosine": cosine,
        "max_abs_raw_w_diff": max_abs_diff,
        "raw_b_diff": raw_b_diff,
    }


def run_grouped_cv(
    *,
    stores: dict[str, ActivationStore],
    target: str,
    label_mapping: dict[str, int],
    grouping_column: str,
    n_folds: int,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
    probe_results_root: Path,
    layers: list[int],
) -> list[dict[str, Any]]:
    del n_folds
    fold_metrics = load_existing_fold_metrics(probe_results_root, target)
    rows: list[dict[str, Any]] = []
    base_groups = prompt_metadata(stores["base"])[grouping_column].astype(str).to_numpy()
    y_by_layer, _ = encoded_labels_by_layer(stores["base"], target)
    for layer in layers:
        x_base = layer_matrix(stores["base"], layer)
        x_lora = layer_matrix(stores["lora"], layer)
        y = y_by_layer[layer]
        layer_splits = split_indices_from_fold_metrics(
            fold_metrics,
            layer=layer,
            groups=base_groups,
        )
        for fold, train_index, test_index, train_families, test_families in layer_splits:
            fit_a = fit_probe(
                condition="A_base_base_scaler",
                scaler_source="base",
                x_train=x_base[train_index],
                y_train=y[train_index],
                x_score_check=x_base[test_index],
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=None,
            )
            fit_b = fit_probe(
                condition="B_lora_base_scaler",
                scaler_source="base",
                x_train=x_lora[train_index],
                y_train=y[train_index],
                x_score_check=x_lora[test_index],
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=fit_a.scaler,
            )
            fit_c = fit_probe(
                condition="C_lora_lora_scaler",
                scaler_source="lora",
                x_train=x_lora[train_index],
                y_train=y[train_index],
                x_score_check=x_lora[test_index],
                max_iter=max_iter,
                class_weight=class_weight,
                random_seed=random_seed,
                scaler=None,
            )
            for fit, x_eval in [
                (fit_a, x_base[test_index]),
                (fit_b, x_lora[test_index]),
                (fit_c, x_lora[test_index]),
            ]:
                metrics = evaluate_raw_scores(fit, x_eval, y[test_index])
                rows.append(
                    {
                        "target": target,
                        "layer": int(layer),
                        "fold": int(fold),
                        "condition": fit.condition,
                        "scaler_source": fit.scaler_source,
                        "label_mapping": json.dumps(label_mapping, sort_keys=True),
                        "positive_class": positive_class(label_mapping),
                        "n_train": int(len(train_index)),
                        "n_test": int(len(test_index)),
                        "train_template_families": json.dumps(train_families),
                        "test_template_families": json.dumps(test_families),
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "roc_auc": metrics["roc_auc"],
                        "accuracy": metrics["accuracy"],
                        "raw_score_max_abs_diff": fit.raw_score_max_abs_diff,
                        "raw_score_allclose": fit.raw_score_allclose,
                        "constant_features": fit.constant_features,
                        "raw_norm": fit.raw_norm,
                        "near_zero_norm": fit.raw_norm <= NORM_EPS,
                        "n_iter": fit.n_iter,
                        "convergence_warning_count": len(fit.convergence_warnings),
                    }
                )
    return rows


def load_existing_fold_metrics(results_root: Path, target: str) -> pd.DataFrame:
    base_path = results_root / "base" / target / "fold_metrics.csv"
    lora_path = results_root / "lora" / target / "fold_metrics.csv"
    if not base_path.exists() or not lora_path.exists():
        raise FileNotFoundError(
            f"Missing existing grouped fold metrics for {target}: {base_path}, {lora_path}"
        )
    base = pd.read_csv(base_path)
    lora = pd.read_csv(lora_path)
    assert_same_group_splits(base, lora)
    return base


def split_indices_from_fold_metrics(
    fold_metrics: pd.DataFrame,
    *,
    layer: int,
    groups: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray, list[str], list[str]]]:
    rows = fold_metrics[fold_metrics["layer_index"].astype(int) == int(layer)]
    if rows.empty:
        raise ValueError(f"No saved fold metrics found for layer {layer}.")
    splits = []
    for _, row in rows.sort_values("fold").iterrows():
        train_families = json.loads(row["train_template_families"])
        test_families = json.loads(row["test_template_families"])
        if set(train_families).intersection(test_families):
            raise ValueError(f"Saved fold has train/test family overlap: layer {layer}.")
        train_index = np.flatnonzero(np.isin(groups, train_families))
        test_index = np.flatnonzero(np.isin(groups, test_families))
        if len(np.unique(groups[train_index])) != len(train_families):
            raise ValueError(f"Could not reconstruct all train families for layer {layer}.")
        if len(np.unique(groups[test_index])) != len(test_families):
            raise ValueError(f"Could not reconstruct all test families for layer {layer}.")
        splits.append((int(row["fold"]), train_index, test_index, train_families, test_families))
    return splits


def aggregate_fold_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in folds.groupby(["target", "layer", "condition", "scaler_source"], sort=True):
        target, layer, condition, scaler_source = keys
        rows.append(
            {
                "target": target,
                "layer": int(layer),
                "condition": condition,
                "scaler_source": scaler_source,
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "std_balanced_accuracy": float(group["balanced_accuracy"].std(ddof=1)),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "std_roc_auc": float(group["roc_auc"].std(ddof=1)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "n_folds": int(group["fold"].nunique()),
                "n_evaluated_prompts": int(group["n_test"].sum()),
                "convergence_warning_count": int(group["convergence_warning_count"].sum()),
                "near_zero_norm_count": int(group["near_zero_norm"].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_plots(full_results: pd.DataFrame, plots_root: Path) -> None:
    for target in TARGETS:
        rows = full_results[full_results["target"] == target].sort_values("layer")
        plt.figure(figsize=(9, 5))
        plt.plot(
            rows["layer"],
            rows["divergence_A_vs_B_fixed_scaler"],
            marker="o",
            label="Base vs LoRA, fixed base scaler",
        )
        plt.plot(
            rows["layer"],
            rows["divergence_A_vs_C_refit_scaler"],
            marker="o",
            label="Base vs LoRA, refit LoRA scaler",
        )
        plt.xlabel("Layer")
        plt.ylabel("Angular divergence (1 - cosine)")
        plt.title(f"{target.capitalize()} base-to-LoRA divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_root / f"{target}_base_lora_fixed_vs_refit_divergence.png", dpi=170)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(
            rows["layer"],
            rows["divergence_B_vs_C_lora_scaler_choice"],
            marker="o",
            color="tab:green",
        )
        plt.xlabel("Layer")
        plt.ylabel("Angular divergence (1 - cosine)")
        plt.title(f"{target.capitalize()} direct LoRA direction divergence: fixed vs refit scaler")
        plt.tight_layout()
        plt.savefig(plots_root / f"{target}_lora_fixed_vs_refit_direct_divergence.png", dpi=170)
        plt.close()


def write_report(
    path: Path,
    *,
    full_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    checks: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    cv_summary = aggregate_fold_metrics(fold_results)
    lines = [
        "# Scaler Sensitivity Diagnostic",
        "",
        "This endpoint diagnostic uses only saved alpha=0 base and alpha=1 LoRA activations. "
        "It does not extract new activations, modify model weights, perform steering, or test "
        "the analytical derivative.",
        "",
        "## Method Checks",
        "",
        f"- Prompt-level rows: {metadata['structure']['prompt_count']}.",
        f"- Template families: {metadata['balance']['template_families']}.",
        "- Existing family-grouped fold definitions were loaded from saved fold metrics "
        "and reused.",
        "- Label orientation was required to be stance geocentric=0/heliocentric=1 and style "
        "modern=0/premodern=1.",
        "- Raw-coordinate scores `w h + b` were checked against standardized-pipeline decision "
        "scores for every full-data and grouped-CV fit.",
        "- Full-data A and C fits were checked against existing saved full-data probe artifacts.",
        "",
        "## Full-Data Geometry",
        "",
    ]
    for target in TARGETS:
        rows = full_results[full_results["target"] == target]
        lines.extend(target_geometry_lines(target, rows))
    lines.extend(
        [
            "",
            "## Predictive Performance",
            "",
        ]
    )
    for target in TARGETS:
        rows = cv_summary[cv_summary["target"] == target]
        lines.extend(target_performance_lines(target, rows))
    lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            f"- Raw-score checks failed: {int((~checks['raw_score_allclose']).sum())}.",
            "- Near-zero full-data norms: "
            f"{int(full_results.filter(like='near_zero_norm').sum().sum())}.",
            "- Convergence warnings are recorded per fit in the CSV outputs.",
            "- Fixed scaling still imposes an anisotropic raw-coordinate penalty; this diagnostic "
            "only holds that penalty fixed between base and LoRA within each layer.",
            "- Endpoint agreement does not establish agreement at intermediate alpha.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_geometry_lines(target: str, rows: pd.DataFrame) -> list[str]:
    nonzero = rows[rows["layer"] != 0]
    largest_effect = rows.reindex(
        rows["divergence_refit_minus_fixed"].abs().sort_values(ascending=False).index
    ).iloc[0]
    fixed_mean = nonzero["divergence_A_vs_B_fixed_scaler"].mean()
    refit_mean = nonzero["divergence_A_vs_C_refit_scaler"].mean()
    direct_mean = nonzero["divergence_B_vs_C_lora_scaler_choice"].mean()
    diff_mean = nonzero["divergence_refit_minus_fixed"].mean()
    early = nonzero[nonzero["layer"].between(1, 8)]["divergence_A_vs_B_fixed_scaler"].mean()
    late = nonzero[nonzero["layer"].between(20, 28)]["divergence_A_vs_B_fixed_scaler"].mean()
    return [
        f"- {target}: mean fixed-scaler base-to-LoRA divergence {fixed_mean:.3f}; "
        f"mean refit-scaler divergence {refit_mean:.3f}; mean direct B-vs-C divergence "
        f"{direct_mean:.3f}; mean refit-minus-fixed divergence {diff_mean:.3f}.",
        f"- {target}: scaler choice mattered most at layer {int(largest_effect['layer'])}, "
        f"where refit-minus-fixed divergence was "
        f"{float(largest_effect['divergence_refit_minus_fixed']):.3f}.",
        f"- {target}: fixed-scaler depth trend persists descriptively: early-layer mean "
        f"divergence {early:.3f}, late-layer mean divergence {late:.3f}.",
    ]


def target_performance_lines(target: str, rows: pd.DataFrame) -> list[str]:
    lines = []
    for condition in [
        "A_base_base_scaler",
        "B_lora_base_scaler",
        "C_lora_lora_scaler",
    ]:
        condition_rows = rows[rows["condition"] == condition]
        lines.append(
            f"- {target} {condition}: mean held-out balanced accuracy "
            f"{condition_rows['mean_balanced_accuracy'].mean():.3f}; mean ROC AUC "
            f"{condition_rows['mean_roc_auc'].mean():.3f} across layers."
        )
    return lines


def prompt_metadata(store: ActivationStore) -> pd.DataFrame:
    return store.metadata[store.metadata["layer_index"] == 0].reset_index(drop=True)


def prompt_ids_for_layer(store: ActivationStore, layer: int) -> list[str]:
    return (
        store.metadata.loc[store.metadata["layer_index"] == layer, "prompt_id"]
        .astype(str)
        .reset_index(drop=True)
        .tolist()
    )


def layer_matrix(store: ActivationStore, layer: int) -> np.ndarray:
    layer_mask = store.metadata["layer_index"].to_numpy() == layer
    return np.asarray(store.activations[layer_mask], dtype=float)


def encoded_labels_by_layer(
    store: ActivationStore,
    target: str,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    y_all, mapping = encode_binary_target(store.metadata, target)
    labels = {}
    for layer in sorted(int(item) for item in store.metadata["layer_index"].unique()):
        labels[layer] = y_all[store.metadata["layer_index"].to_numpy() == layer]
    return labels, mapping


def positive_class(label_mapping: dict[str, int]) -> str:
    positives = [label for label, encoded in label_mapping.items() if encoded == 1]
    if len(positives) != 1:
        raise ValueError(f"Expected one positive class, found {label_mapping}.")
    return positives[0]


def cosine_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= NORM_EPS or right_norm <= NORM_EPS:
        return float("nan")
    return float(left @ right / (left_norm * right_norm))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= NORM_EPS:
        return np.full_like(vector, np.nan)
    return vector / norm


def evaluate_raw_scores(fit: ProbeFit, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    scores = x_eval @ fit.raw_w + fit.raw_b
    predictions = (scores >= 0.0).astype(int)
    return {
        "accuracy": float(np.mean(predictions == y_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, predictions)),
        "roc_auc": float(roc_auc_score(y_eval, scores)),
    }


if __name__ == "__main__":
    main()
