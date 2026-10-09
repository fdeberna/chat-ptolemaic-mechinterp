"""Analyze bootstrap stability of raw-space linear probe directions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptolemaic_mechinterp.activations.storage import ActivationStore, load_activation_store
from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import (
    load_full_data_probe_artifact,
    validate_activation_store_pair,
    validate_factor_balance,
)
from ptolemaic_mechinterp.probes.evaluation import binary_classification_metrics
from ptolemaic_mechinterp.probes.stability import (
    assert_expected_label_orientation,
    bootstrap_samples_frame,
    cosine_distribution_summary,
    draw_family_bootstrap_samples,
    fit_normalized_raw_direction,
    matched_direction_cosines,
    normalize_vector,
    pairwise_direction_cosines,
    prompt_rows,
)

TARGETS = ("stance", "style")
MODEL_CONDITIONS = ("base", "lora")
SENSITIVITY_LAYERS = {"stance": (16, 17, 18, 19), "style": (6, 7, 8)}
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument("--base-activations", default="results/activations/pilot_full/base")
    parser.add_argument("--lora-activations", default="results/activations/pilot_full/lora")
    parser.add_argument("--probe-results-root", default="results/probes/pilot_full")
    parser.add_argument("--output-root", default="results/probe_stability")
    parser.add_argument("--expected-prompts", type=int, default=120)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--sensitivity-replicates", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20261006)
    parser.add_argument("--primary-c", type=float, default=1.0)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_C_GRID),
        help="Regularization C values for sensitivity analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_probe_config(args.probe_config)
    base_store = load_activation_store(resolve_repo_path(args.base_activations))
    lora_store = load_activation_store(resolve_repo_path(args.lora_activations))
    stores = {"base": base_store, "lora": lora_store}
    structure = validate_activation_store_pair(
        base_store,
        lora_store,
        expected_prompt_count=args.expected_prompts,
    )
    balance = validate_factor_balance(base_store.metadata)
    validate_factor_balance(lora_store.metadata)

    output_root = resolve_repo_path(args.output_root)
    directions_root = output_root / "directions"
    plots_root = output_root / "plots"
    samples_root = output_root / "bootstrap_samples"
    for path in [output_root, directions_root, plots_root, samples_root]:
        path.mkdir(parents=True, exist_ok=True)

    layers = [
        int(layer)
        for layer in sorted(base_store.metadata["layer_index"].unique())
        if int(layer) != 0
    ]
    if not layers:
        raise ValueError("No nonzero layers are available for probe stability analysis.")

    run_metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_activations": str(resolve_repo_path(args.base_activations)),
        "lora_activations": str(resolve_repo_path(args.lora_activations)),
        "probe_results_root": str(resolve_repo_path(args.probe_results_root)),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "layers_analyzed": layers,
        "excluded_layers": [0],
        "bootstrap_replicates": args.bootstrap_replicates,
        "sensitivity_replicates": args.sensitivity_replicates,
        "random_seed": args.random_seed,
        "primary_c": args.primary_c,
        "c_grid": list(args.c_grid),
        "probe_hyperparameters": {
            "standardizer": "sklearn.preprocessing.StandardScaler",
            "classifier": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "C": args.primary_c,
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "random_state": config.random_seed,
        },
        "structure": structure,
        "balance": balance,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    within_rows, matched_rows, full_rows, alignment_rows, rejection_rows = run_primary_bootstrap(
        stores=stores,
        layers=layers,
        grouping_column=config.grouping_column,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_seed=args.random_seed,
        logistic_random_seed=config.random_seed,
        n_replicates=args.bootstrap_replicates,
        c_value=args.primary_c,
        probe_results_root=resolve_repo_path(args.probe_results_root),
        directions_root=directions_root,
        samples_root=samples_root,
    )

    within = pd.DataFrame(within_rows)
    matched = pd.DataFrame(matched_rows)
    full_reference = pd.DataFrame(full_rows)
    alignment = pd.DataFrame(alignment_rows)
    rejections = pd.DataFrame(
        rejection_rows,
        columns=["target", "model_condition", "layer", "replicate", "reason"],
    )
    within.to_csv(output_root / "bootstrap_within_summary.csv", index=False)
    matched.to_csv(output_root / "matched_base_lora_summary.csv", index=False)
    full_reference.to_csv(output_root / "full_data_reference_stability.csv", index=False)
    alignment.to_csv(output_root / "alignment_gap_summary.csv", index=False)
    rejections.to_csv(output_root / "bootstrap_fit_rejections.csv", index=False)

    sensitivity = run_regularization_sensitivity(
        stores=stores,
        selected_layers=SENSITIVITY_LAYERS,
        grouping_column=config.grouping_column,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_seed=args.random_seed + 10_000,
        logistic_random_seed=config.random_seed,
        n_replicates=args.sensitivity_replicates,
        c_grid=args.c_grid,
    )
    sensitivity.to_csv(output_root / "regularization_sensitivity.csv", index=False)

    redundancy = compute_feature_redundancy_diagnostics(
        stores=stores,
        selected_layers=sorted(set(sum((list(v) for v in SENSITIVITY_LAYERS.values()), []))),
    )
    redundancy.to_csv(output_root / "feature_redundancy_diagnostics.csv", index=False)

    make_plots(
        alignment=alignment,
        full_reference=full_reference,
        sensitivity=sensitivity,
        plots_root=plots_root,
    )
    write_report(
        path=output_root / "probe_stability_report.md",
        alignment=alignment,
        full_reference=full_reference,
        sensitivity=sensitivity,
        redundancy=redundancy,
        metadata=run_metadata,
    )

    print("probe_stability_analysis_complete")
    print(f"results: {output_root}")


def run_primary_bootstrap(
    *,
    stores: dict[str, ActivationStore],
    layers: list[int],
    grouping_column: str,
    max_iter: int,
    class_weight: str | None,
    random_seed: int,
    logistic_random_seed: int,
    n_replicates: int,
    c_value: float,
    probe_results_root: Path,
    directions_root: Path,
    samples_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    within_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    base_prompts = prompt_rows(stores["base"].metadata)
    lora_prompts = prompt_rows(stores["lora"].metadata)
    assert_prompt_alignment(base_prompts, lora_prompts)

    for target_index, target in enumerate(TARGETS):
        y_base, label_mapping = assert_expected_label_orientation(base_prompts, target)
        y_lora, lora_mapping = assert_expected_label_orientation(lora_prompts, target)
        if label_mapping != lora_mapping or not np.array_equal(y_base, y_lora):
            raise ValueError(f"Base and LoRA labels differ for target {target}.")
        samples = draw_family_bootstrap_samples(
            base_prompts,
            grouping_column=grouping_column,
            n_replicates=n_replicates,
            random_seed=random_seed + target_index,
        )
        bootstrap_samples_frame(samples).to_csv(
            samples_root / f"{target}_family_bootstrap_samples.csv",
            index=False,
        )
        directions_by_model: dict[str, np.ndarray] = {}
        for model_name in MODEL_CONDITIONS:
            print(f"primary bootstrap: target={target} model={model_name}")
            directions = np.full(
                (n_replicates, len(layers), stores[model_name].activations.shape[1]),
                np.nan,
                dtype=np.float32,
            )
            for layer_position, layer in enumerate(layers):
                x_layer = layer_matrix(stores[model_name], layer, base_prompts["prompt_id"])
                for replicate, prompt_index in enumerate(samples.prompt_indices):
                    if len(np.unique(y_base[prompt_index])) != 2:
                        rejection_rows.append(
                            rejection_row(target, model_name, layer, replicate, "single_class")
                        )
                        continue
                    try:
                        direction, _ = fit_normalized_raw_direction(
                            x_layer[prompt_index],
                            y_base[prompt_index],
                            random_seed=logistic_random_seed,
                            max_iter=max_iter,
                            class_weight=class_weight,
                            c_value=c_value,
                        )
                    except ValueError as exc:
                        rejection_rows.append(
                            rejection_row(target, model_name, layer, replicate, str(exc))
                        )
                        continue
                    directions[replicate, layer_position] = direction.astype(np.float32)
            np.savez_compressed(
                directions_root / f"{model_name}_{target}_bootstrap_directions.npz",
                directions=directions,
                layers=np.asarray(layers, dtype=int),
                target=np.asarray(target),
                model_condition=np.asarray(model_name),
                c_value=np.asarray(c_value, dtype=float),
            )
            directions_by_model[model_name] = directions

            for layer_position, layer in enumerate(layers):
                stats = prefixed_summary(
                    pairwise_direction_cosines(directions[:, layer_position, :])
                )
                within_rows.append(
                    {
                        "target": target,
                        "layer": layer,
                        "model_condition": model_name,
                        "comparison": f"{model_name}_within",
                        **stats,
                    }
                )
                full_direction = full_data_direction(probe_results_root, model_name, target, layer)
                full_stats = prefixed_summary(directions[:, layer_position, :] @ full_direction)
                full_rows.append(
                    {
                        "target": target,
                        "layer": layer,
                        "model_condition": model_name,
                        "comparison": f"{model_name}_bootstrap_vs_full_data",
                        **full_stats,
                    }
                )

        for layer_position, layer in enumerate(layers):
            matched_values = matched_direction_cosines(
                directions_by_model["base"][:, layer_position, :],
                directions_by_model["lora"][:, layer_position, :],
            )
            matched_stats = prefixed_summary(matched_values)
            matched_rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "comparison": "matched_base_lora",
                    **matched_stats,
                }
            )
            base_stats = prefixed_summary(
                pairwise_direction_cosines(directions_by_model["base"][:, layer_position, :])
            )
            lora_stats = prefixed_summary(
                pairwise_direction_cosines(directions_by_model["lora"][:, layer_position, :])
            )
            original = original_full_data_base_lora_cosine(probe_results_root, target, layer)
            within_reference = (base_stats["mean"] + lora_stats["mean"]) / 2.0
            alignment_rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "base_within_mean_cosine": base_stats["mean"],
                    "base_within_median_cosine": base_stats["median"],
                    "base_within_ci_low": base_stats["p2_5"],
                    "base_within_ci_high": base_stats["p97_5"],
                    "lora_within_mean_cosine": lora_stats["mean"],
                    "lora_within_median_cosine": lora_stats["median"],
                    "lora_within_ci_low": lora_stats["p2_5"],
                    "lora_within_ci_high": lora_stats["p97_5"],
                    "base_lora_mean_cosine": matched_stats["mean"],
                    "base_lora_median_cosine": matched_stats["median"],
                    "base_lora_ci_low": matched_stats["p2_5"],
                    "base_lora_ci_high": matched_stats["p97_5"],
                    "original_full_data_base_lora_cosine": original,
                    "within_reference": within_reference,
                    "alignment_gap": within_reference - matched_stats["mean"],
                }
            )
    return within_rows, matched_rows, full_rows, alignment_rows, rejection_rows


def run_regularization_sensitivity(
    *,
    stores: dict[str, ActivationStore],
    selected_layers: dict[str, tuple[int, ...]],
    grouping_column: str,
    max_iter: int,
    class_weight: str | None,
    random_seed: int,
    logistic_random_seed: int,
    n_replicates: int,
    c_grid: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_prompts = prompt_rows(stores["base"].metadata)
    lora_prompts = prompt_rows(stores["lora"].metadata)
    assert_prompt_alignment(base_prompts, lora_prompts)

    for target_index, target in enumerate(TARGETS):
        y_prompt, _ = assert_expected_label_orientation(base_prompts, target)
        samples = draw_family_bootstrap_samples(
            base_prompts,
            grouping_column=grouping_column,
            n_replicates=n_replicates,
            random_seed=random_seed + target_index,
        )
        groups = base_prompts[grouping_column].astype(str).to_numpy()
        for layer in selected_layers[target]:
            directions_by_c_model: dict[tuple[float, str], np.ndarray] = {}
            for c_value in c_grid:
                for model_name in MODEL_CONDITIONS:
                    print(
                        "sensitivity: "
                        f"target={target} layer={layer} C={c_value:g} model={model_name}"
                    )
                    x_layer = layer_matrix(stores[model_name], layer, base_prompts["prompt_id"])
                    directions = np.full(
                        (n_replicates, stores[model_name].activations.shape[1]),
                        np.nan,
                        dtype=np.float32,
                    )
                    train_metrics: list[dict[str, float]] = []
                    oob_metrics: list[dict[str, float]] = []
                    for replicate, prompt_index in enumerate(samples.prompt_indices):
                        direction, probe = fit_normalized_raw_direction(
                            x_layer[prompt_index],
                            y_prompt[prompt_index],
                            random_seed=logistic_random_seed,
                            max_iter=max_iter,
                            class_weight=class_weight,
                            c_value=c_value,
                        )
                        directions[replicate] = direction.astype(np.float32)
                        train_metrics.append(
                            evaluate_probe(probe, x_layer[prompt_index], y_prompt[prompt_index])
                        )
                        selected_families = set(samples.sampled_families[replicate])
                        oob_index = np.flatnonzero(
                            np.asarray([family not in selected_families for family in groups])
                        )
                        if len(oob_index) > 0 and len(np.unique(y_prompt[oob_index])) == 2:
                            oob_metrics.append(
                                evaluate_probe(probe, x_layer[oob_index], y_prompt[oob_index])
                            )
                    directions_by_c_model[(float(c_value), model_name)] = directions
                    direction_stats = prefixed_summary(pairwise_direction_cosines(directions))
                    row = {
                        "target": target,
                        "layer": layer,
                        "C": float(c_value),
                        "model_condition": model_name,
                        "n_replicates": n_replicates,
                        "direction_mean_cosine": direction_stats["mean"],
                        "direction_median_cosine": direction_stats["median"],
                        "direction_ci_low": direction_stats["p2_5"],
                        "direction_ci_high": direction_stats["p97_5"],
                    }
                    row.update(metric_summary(train_metrics, prefix="train"))
                    row.update(metric_summary(oob_metrics, prefix="oob"))
                    rows.append(row)
                matched_stats = prefixed_summary(
                    matched_direction_cosines(
                        directions_by_c_model[(float(c_value), "base")],
                        directions_by_c_model[(float(c_value), "lora")],
                    )
                )
                for row in rows:
                    if (
                        row["target"] == target
                        and row["layer"] == layer
                        and row["C"] == float(c_value)
                    ):
                        row["matched_base_lora_mean_cosine"] = matched_stats["mean"]
                        row["matched_base_lora_median_cosine"] = matched_stats["median"]
                        row["matched_base_lora_ci_low"] = matched_stats["p2_5"]
                        row["matched_base_lora_ci_high"] = matched_stats["p97_5"]
    return pd.DataFrame(rows)


def compute_feature_redundancy_diagnostics(
    *,
    stores: dict[str, ActivationStore],
    selected_layers: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_prompts = prompt_rows(stores["base"].metadata)["prompt_id"]
    for model_name in MODEL_CONDITIONS:
        for layer in selected_layers:
            x_layer = layer_matrix(stores[model_name], layer, reference_prompts)
            centered = x_layer - x_layer.mean(axis=0, keepdims=True)
            singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
            variance = singular_values**2
            total = float(variance.sum())
            if total <= 0.0:
                explained = np.full_like(variance, np.nan, dtype=float)
                effective_rank = float("nan")
            else:
                explained = variance / total
                positive = explained[explained > 0.0]
                effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
            cumulative = np.cumsum(explained)
            rows.append(
                {
                    "model_condition": model_name,
                    "layer": layer,
                    "n_prompts": int(x_layer.shape[0]),
                    "hidden_size": int(x_layer.shape[1]),
                    "matrix_rank": int(np.linalg.matrix_rank(centered)),
                    "effective_rank": effective_rank,
                    "pc50": pcs_for_threshold(cumulative, 0.50),
                    "pc80": pcs_for_threshold(cumulative, 0.80),
                    "pc90": pcs_for_threshold(cumulative, 0.90),
                    "pc95": pcs_for_threshold(cumulative, 0.95),
                    "top1_explained_variance": float(explained[0]),
                    "top5_explained_variance": float(np.nansum(explained[:5])),
                    "top10_explained_variance": float(np.nansum(explained[:10])),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    *,
    alignment: pd.DataFrame,
    full_reference: pd.DataFrame,
    sensitivity: pd.DataFrame,
    plots_root: Path,
) -> None:
    for target in TARGETS:
        target_rows = alignment[alignment["target"] == target].sort_values("layer")
        plt.figure(figsize=(9, 5))
        plot_band(
            target_rows,
            "layer",
            "base_within_mean_cosine",
            "base_within_ci_low",
            "base_within_ci_high",
            "Base within",
        )
        plot_band(
            target_rows,
            "layer",
            "lora_within_mean_cosine",
            "lora_within_ci_low",
            "lora_within_ci_high",
            "LoRA within",
        )
        plot_band(
            target_rows,
            "layer",
            "base_lora_mean_cosine",
            "base_lora_ci_low",
            "base_lora_ci_high",
            "Base vs LoRA",
        )
        plt.xlabel("Layer")
        plt.ylabel("Cosine similarity")
        plt.ylim(0.0, 1.05)
        plt.title(f"{target.capitalize()} direction stability by layer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_root / f"{target}_direction_stability_by_layer.png", dpi=170)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        plt.plot(target_rows["layer"], target_rows["alignment_gap"], marker="o")
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        plt.xlabel("Layer")
        plt.ylabel("Mean within-model stability - mean base-vs-LoRA alignment")
        plt.title(f"{target.capitalize()} alignment gap by layer")
        plt.tight_layout()
        plt.savefig(plots_root / f"{target}_alignment_gap_by_layer.png", dpi=170)
        plt.close()

        plt.figure(figsize=(9, 5))
        target_full = full_reference[full_reference["target"] == target].sort_values("layer")
        for model_name in MODEL_CONDITIONS:
            rows = target_full[target_full["model_condition"] == model_name]
            plt.plot(rows["layer"], rows["mean"], marker="o", label=model_name)
            plt.fill_between(rows["layer"], rows["p2_5"], rows["p97_5"], alpha=0.18)
        plt.xlabel("Layer")
        plt.ylabel("Cosine with full-data direction")
        plt.ylim(0.0, 1.05)
        plt.title(f"{target.capitalize()} bootstrap-to-full-data direction similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_root / f"{target}_bootstrap_to_full_data_similarity.png", dpi=170)
        plt.close()

        plot_regularization_sensitivity(
            sensitivity[sensitivity["target"] == target],
            target=target,
            output_path=plots_root / f"{target}_regularization_sensitivity.png",
        )


def plot_regularization_sensitivity(rows: pd.DataFrame, *, target: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for (model_name, layer), group in rows.groupby(["model_condition", "layer"], sort=True):
        group = group.sort_values("C")
        label = f"{model_name} L{int(layer)}"
        axes[0].plot(group["C"], group["mean_oob_balanced_accuracy"], marker="o", label=label)
        axes[1].plot(group["C"], group["direction_mean_cosine"], marker="o", label=label)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("C")
    axes[1].set_xlabel("C")
    axes[0].set_ylabel("Mean OOB balanced accuracy")
    axes[1].set_ylabel("Mean within-model direction cosine")
    axes[0].set_title("Predictive performance")
    axes[1].set_title("Direction stability")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{target.capitalize()} regularization sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_report(
    *,
    path: Path,
    alignment: pd.DataFrame,
    full_reference: pd.DataFrame,
    sensitivity: pd.DataFrame,
    redundancy: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Probe Direction Stability",
        "",
        "This analysis uses saved activations only. It does not rerun activation extraction, "
        "modify model weights, run activation steering, or generate new model text.",
        "",
        "## Method Checks",
        "",
        f"- Prompt-level rows: {metadata['structure']['prompt_count']}.",
        f"- Template families: {metadata['balance']['template_families']}.",
        "- Base and LoRA prompt IDs, labels, template families, and layer indices were required "
        "to align exactly before analysis.",
        "- Bootstrap resampling is at the template-family level with replacement; duplicated "
        "families are expanded to duplicated prompt rows.",
        "- Probe directions are converted from standardized coefficients to raw activation "
        "coordinates before L2 normalization.",
        "- Label orientation is required to be stance geocentric=0/heliocentric=1 and style "
        "modern=0/premodern=1.",
        "- Layer 0 is excluded from aggregate conclusions.",
        "",
        "## Primary Bootstrap Summary",
        "",
    ]
    for target in TARGETS:
        rows = alignment[alignment["target"] == target]
        lines.extend(target_summary_lines(target, rows))
    lines.extend(
        [
            "",
            "## Full-Data Probe Reference",
            "",
        ]
    )
    for target in TARGETS:
        for model_name in MODEL_CONDITIONS:
            rows = full_reference[
                (full_reference["target"] == target)
                & (full_reference["model_condition"] == model_name)
            ]
            lines.append(
                f"- {target} {model_name}: mean bootstrap-vs-full cosine "
                f"{rows['mean'].mean():.3f} across nonzero layers "
                f"(median layer value {rows['median'].median():.3f})."
            )
    lines.extend(
        [
            "",
            "## Regularization Sensitivity",
            "",
        ]
    )
    for target in TARGETS:
        rows = sensitivity[sensitivity["target"] == target]
        lines.append(
            f"- {target}: OOB balanced accuracy ranged from "
            f"{rows['mean_oob_balanced_accuracy'].min():.3f} to "
            f"{rows['mean_oob_balanced_accuracy'].max():.3f}; within-model direction "
            f"mean cosine ranged from {rows['direction_mean_cosine'].min():.3f} to "
            f"{rows['direction_mean_cosine'].max():.3f} over the selected layers and C grid."
        )
    lines.extend(
        [
            "",
            "## Feature-Correlation Context",
            "",
            (
                "- Candidate-layer activation matrices are prompt-by-hidden-state matrices. "
                "Effective rank and PCA counts are descriptive redundancy diagnostics only; "
                "they do not identify causal model features."
            ),
        ]
    )
    lines.append(
        "- Effective rank range across candidate layers and model conditions: "
        f"{redundancy['effective_rank'].min():.1f} to {redundancy['effective_rank'].max():.1f}."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use `alignment_gap = mean(within-model direction stability) - "
            "mean(base-vs-LoRA alignment)` descriptively. Values near 0 mean the between-model "
            "direction difference is comparable to ordinary probe refitting variability. Larger "
            "positive values mean base and LoRA directions differ more than within-model bootstrap "
            "refits. This is not a formal statistical test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_summary_lines(target: str, rows: pd.DataFrame) -> list[str]:
    base_mean = rows["base_within_mean_cosine"].mean()
    lora_mean = rows["lora_within_mean_cosine"].mean()
    matched_mean = rows["base_lora_mean_cosine"].mean()
    gap_mean = rows["alignment_gap"].mean()
    max_gap = rows.sort_values("alignment_gap", ascending=False).iloc[0]
    min_gap = rows.sort_values("alignment_gap", ascending=True).iloc[0]
    return [
        f"- {target}: mean base-within cosine {base_mean:.3f}; mean LoRA-within cosine "
        f"{lora_mean:.3f}; mean matched base-vs-LoRA cosine {matched_mean:.3f}; "
        f"mean alignment gap {gap_mean:.3f}.",
        f"- {target}: largest gap at layer {int(max_gap['layer'])} "
        f"({float(max_gap['alignment_gap']):.3f}); smallest gap at layer "
        f"{int(min_gap['layer'])} ({float(min_gap['alignment_gap']):.3f}).",
    ]


def prefixed_summary(values: np.ndarray) -> dict[str, float]:
    return cosine_distribution_summary(values)


def rejection_row(
    target: str,
    model_name: str,
    layer: int,
    replicate: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "target": target,
        "model_condition": model_name,
        "layer": layer,
        "replicate": replicate,
        "reason": reason,
    }


def assert_prompt_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["prompt_id", "stance", "style", "template_family"]
    for column in columns:
        if (
            left[column].fillna("").astype(str).tolist()
            != right[column].fillna("").astype(str).tolist()
        ):
            raise ValueError(f"Prompt-level base/LoRA metadata differ for column: {column}.")


def layer_matrix(store: ActivationStore, layer: int, reference_prompt_ids: pd.Series) -> np.ndarray:
    mask = store.metadata["layer_index"].to_numpy() == layer
    rows = store.metadata.loc[mask].reset_index(drop=True)
    prompt_ids = rows["prompt_id"].astype(str).tolist()
    expected = reference_prompt_ids.astype(str).tolist()
    if prompt_ids != expected:
        raise ValueError(f"Prompt order differs at layer {layer}.")
    return np.asarray(store.activations[mask], dtype=float)


def full_data_direction(results_root: Path, model_name: str, target: str, layer: int) -> np.ndarray:
    path = (
        results_root
        / model_name
        / target
        / "full_data_coefficients"
        / f"layer_{layer:03d}_full_data.npz"
    )
    artifact = load_full_data_probe_artifact(path)
    return normalize_vector(artifact["raw_direction"])


def original_full_data_base_lora_cosine(results_root: Path, target: str, layer: int) -> float:
    base = full_data_direction(results_root, "base", target, layer)
    lora = full_data_direction(results_root, "lora", target, layer)
    return float(np.dot(base, lora))


def evaluate_probe(probe: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = probe.predict(x)
    scores = probe.predict_proba(x)[:, 1]
    return binary_classification_metrics(y, predictions, scores)


def metric_summary(rows: list[dict[str, float]], *, prefix: str) -> dict[str, float]:
    if not rows:
        return {
            f"mean_{prefix}_accuracy": float("nan"),
            f"mean_{prefix}_balanced_accuracy": float("nan"),
            f"mean_{prefix}_roc_auc": float("nan"),
            f"std_{prefix}_balanced_accuracy": float("nan"),
            f"n_{prefix}_evaluations": 0,
        }
    frame = pd.DataFrame(rows)
    return {
        f"mean_{prefix}_accuracy": float(frame["accuracy"].mean()),
        f"mean_{prefix}_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        f"mean_{prefix}_roc_auc": float(frame["roc_auc"].mean()),
        f"std_{prefix}_balanced_accuracy": float(frame["balanced_accuracy"].std(ddof=1)),
        f"n_{prefix}_evaluations": int(len(frame)),
    }


def pcs_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    if not np.isfinite(cumulative).all():
        return 0
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def plot_band(
    rows: pd.DataFrame,
    x_column: str,
    mean_column: str,
    low_column: str,
    high_column: str,
    label: str,
) -> None:
    plt.plot(rows[x_column], rows[mean_column], marker="o", label=label)
    plt.fill_between(rows[x_column], rows[low_column], rows[high_column], alpha=0.18)


if __name__ == "__main__":
    main()
