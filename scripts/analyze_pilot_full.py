"""Aggregate full pilot probes and run base/LoRA comparison analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import (
    aggregate_layer_metrics,
    aggregate_transfer_layers,
    assert_label_mapping_equal,
    assert_same_group_splits,
    build_audited_direction_similarity,
    cosine_similarity,
    cross_model_probe_transfer,
    label_mapping_from_metrics,
    train_full_data_probe_directions,
    validate_activation_store_pair,
    validate_factor_balance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", default="configs/probes.yaml")
    parser.add_argument("--base-activations", default="results/activations/pilot_full/base")
    parser.add_argument("--lora-activations", default="results/activations/pilot_full/lora")
    parser.add_argument("--results-root", default="results/probes/pilot_full")
    parser.add_argument("--comparison-root", default="results/comparisons/pilot_full")
    parser.add_argument("--figure-root", default="results/figures/pilot_full")
    parser.add_argument("--summary-md", default="results/pilot_full_summary.md")
    parser.add_argument("--expected-prompts", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_probe_config(args.probe_config)
    base_store = load_activation_store(resolve_repo_path(args.base_activations))
    lora_store = load_activation_store(resolve_repo_path(args.lora_activations))
    structure = validate_activation_store_pair(
        base_store, lora_store, expected_prompt_count=args.expected_prompts
    )
    balance = validate_factor_balance(base_store.metadata)
    validate_factor_balance(lora_store.metadata)

    results_root = resolve_repo_path(args.results_root)
    comparison_root = resolve_repo_path(args.comparison_root)
    figure_root = resolve_repo_path(args.figure_root)
    comparison_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    fold_metrics = load_and_aggregate_probe_metrics(results_root)
    for target in ["stance", "style"]:
        assert_same_group_splits(fold_metrics[("base", target)], fold_metrics[("lora", target)])

    direction_rows = []
    direction_vectors: dict[tuple[str, str], dict[int, Any]] = {}
    for model_name, store in [("base", base_store), ("lora", lora_store)]:
        for target in ["stance", "style"]:
            output_dir = results_root / model_name / target / "full_data_coefficients"
            metadata, vectors = train_full_data_probe_directions(
                store,
                target=target,
                random_seed=config.random_seed,
                max_iter=config.max_iter,
                class_weight=config.class_weight,
                output_dir=output_dir,
            )
            metadata.to_csv(
                results_root / model_name / target / "full_data_probe_metadata.csv",
                index=False,
            )
            direction_vectors[(model_name, target)] = vectors
            direction_rows.append(metadata.assign(model_condition=model_name))

    pd.concat(direction_rows, ignore_index=True).to_csv(
        comparison_root / "full_data_probe_metadata.csv", index=False
    )
    direction_similarity = build_direction_similarity(fold_metrics, direction_vectors)
    direction_similarity.to_csv(
        comparison_root / "probe_direction_similarity.csv", index=False
    )
    build_audited_direction_similarity(results_root=results_root).to_csv(
        comparison_root / "probe_direction_similarity_audited.csv", index=False
    )

    transfer_folds = run_transfer_analyses(
        base_store=base_store,
        lora_store=lora_store,
        config=config,
    )
    transfer_folds.to_csv(
        comparison_root / "cross_model_probe_transfer_folds.csv", index=False
    )
    transfer_layers = aggregate_transfer_layers(transfer_folds)
    transfer_layers.to_csv(
        comparison_root / "cross_model_probe_transfer_layers.csv", index=False
    )

    make_metric_plots(results_root, figure_root)
    summary_table = build_stance_style_summary(
        results_root=results_root,
        direction_similarity=direction_similarity,
        transfer_layers=transfer_layers,
    )
    summary_table.to_csv(comparison_root / "stance_style_summary.csv", index=False)
    write_summary(
        path=resolve_repo_path(args.summary_md),
        structure=structure,
        balance=balance,
        results_root=results_root,
        direction_similarity=direction_similarity,
        transfer_layers=transfer_layers,
        summary_table=summary_table,
    )

    print("pilot_full_analysis_complete")
    print(json.dumps(structure, indent=2, sort_keys=True))
    print(f"summary: {resolve_repo_path(args.summary_md)}")


def load_and_aggregate_probe_metrics(results_root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    metrics: dict[tuple[str, str], pd.DataFrame] = {}
    for model_name in ["base", "lora"]:
        for target in ["stance", "style"]:
            target_dir = results_root / model_name / target
            folds = pd.read_csv(target_dir / "fold_metrics.csv")
            folds.to_csv(target_dir / "fold_metrics.csv", index=False)
            aggregate_layer_metrics(folds).to_csv(target_dir / "layer_metrics.csv", index=False)
            label_mapping_from_metrics(folds)
            metrics[(model_name, target)] = folds
    return metrics


def build_direction_similarity(
    fold_metrics: dict[tuple[str, str], pd.DataFrame],
    direction_vectors: dict[tuple[str, str], dict[int, Any]],
) -> pd.DataFrame:
    rows = []
    for target in ["stance", "style"]:
        base_mapping = label_mapping_from_metrics(fold_metrics[("base", target)])
        lora_mapping = label_mapping_from_metrics(fold_metrics[("lora", target)])
        assert_label_mapping_equal(base_mapping, lora_mapping, target=target)
    layers = sorted(direction_vectors[("base", "stance")])
    for layer in layers:
        row = {"layer": layer}
        for target in ["stance", "style"]:
            base_direction = direction_vectors[("base", target)][layer]
            lora_direction = direction_vectors[("lora", target)][layer]
            base_norm = float((base_direction @ base_direction) ** 0.5)
            lora_norm = float((lora_direction @ lora_direction) ** 0.5)
            row[f"{target}_cosine"] = (
                np.nan
                if base_norm <= 0.0 or lora_norm <= 0.0
                else cosine_similarity(base_direction, lora_direction)
            )
            row[f"base_{target}_norm"] = base_norm
            row[f"lora_{target}_norm"] = lora_norm
        rows.append(row)
    return pd.DataFrame(rows)


def run_transfer_analyses(*, base_store: Any, lora_store: Any, config: Any) -> pd.DataFrame:
    rows = []
    stores = {"base": base_store, "lora": lora_store}
    for target in ["stance", "style"]:
        for source_name, target_name in [
            ("base", "base"),
            ("lora", "lora"),
            ("base", "lora"),
            ("lora", "base"),
        ]:
            rows.append(
                cross_model_probe_transfer(
                    stores[source_name],
                    stores[target_name],
                    source_model=source_name,
                    target_model=target_name,
                    target_variable=target,
                    grouping_column=config.grouping_column,
                    n_folds=config.n_folds,
                    random_seed=config.random_seed,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                )
            )
    return pd.concat(rows, ignore_index=True)


def make_metric_plots(results_root: Path, figure_root: Path) -> None:
    for target in ["stance", "style"]:
        for metric in ["balanced_accuracy", "roc_auc"]:
            plt.figure(figsize=(8, 4.5))
            for model_name in ["base", "lora"]:
                metrics = pd.read_csv(results_root / model_name / target / "layer_metrics.csv")
                plt.plot(metrics["layer"], metrics[f"mean_{metric}"], marker="o", label=model_name)
                plt.fill_between(
                    metrics["layer"],
                    metrics[f"mean_{metric}"] - metrics[f"std_{metric}"],
                    metrics[f"mean_{metric}"] + metrics[f"std_{metric}"],
                    alpha=0.2,
                )
            plt.xlabel("Layer index")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"{target.capitalize()} probe {metric.replace('_', ' ')}")
            plt.ylim(0.0, 1.05)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_root / f"{target}_{metric}_base_lora.png", dpi=160)
            plt.close()


def build_stance_style_summary(
    *,
    results_root: Path,
    direction_similarity: pd.DataFrame,
    transfer_layers: pd.DataFrame,
) -> pd.DataFrame:
    base_stance = metric_series(results_root, "base", "stance", "balanced_accuracy")
    lora_stance = metric_series(results_root, "lora", "stance", "balanced_accuracy")
    base_style = metric_series(results_root, "base", "style", "balanced_accuracy")
    lora_style = metric_series(results_root, "lora", "style", "balanced_accuracy")
    rows = []
    for layer in sorted(base_stance):
        row = {
            "layer": layer,
            "base_stance_bal_acc": base_stance[layer],
            "lora_stance_bal_acc": lora_stance[layer],
            "base_style_bal_acc": base_style[layer],
            "lora_style_bal_acc": lora_style[layer],
            "stance_probe_cosine": value_at(direction_similarity, layer, "stance_cosine"),
            "style_probe_cosine": value_at(direction_similarity, layer, "style_cosine"),
            "base_to_lora_stance_transfer": transfer_value(
                transfer_layers, layer, "stance", "base", "lora"
            ),
            "lora_to_base_stance_transfer": transfer_value(
                transfer_layers, layer, "stance", "lora", "base"
            ),
            "base_to_lora_style_transfer": transfer_value(
                transfer_layers, layer, "style", "base", "lora"
            ),
            "lora_to_base_style_transfer": transfer_value(
                transfer_layers, layer, "style", "lora", "base"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def metric_series(
    results_root: Path, model_name: str, target: str, metric: str
) -> dict[int, float]:
    metrics = pd.read_csv(results_root / model_name / target / "layer_metrics.csv")
    return dict(zip(metrics["layer"].astype(int), metrics[f"mean_{metric}"], strict=True))


def value_at(frame: pd.DataFrame, layer: int, column: str) -> float:
    values = frame.loc[frame["layer"] == layer, column]
    if len(values) != 1:
        raise ValueError(f"Expected one value for layer {layer}, column {column}.")
    return float(values.iloc[0])


def transfer_value(
    frame: pd.DataFrame, layer: int, target: str, source_model: str, target_model: str
) -> float:
    rows = frame[
        (frame["layer"] == layer)
        & (frame["target_variable"] == target)
        & (frame["source_model"] == source_model)
        & (frame["target_model"] == target_model)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one transfer value for {layer}, {target}, {source_model}->{target_model}."
        )
    return float(rows["mean_balanced_accuracy"].iloc[0])


def write_summary(
    *,
    path: Path,
    structure: dict[str, Any],
    balance: dict[str, Any],
    results_root: Path,
    direction_similarity: pd.DataFrame,
    transfer_layers: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = {
        (model, target): best_layer(results_root, model, target)
        for model in ["base", "lora"]
        for target in ["stance", "style"]
    }
    weakest_transfer = transfer_layers[
        transfer_layers["source_model"] != transfer_layers["target_model"]
    ].sort_values("mean_balanced_accuracy").head(1)
    strongest_transfer = transfer_layers[
        transfer_layers["source_model"] != transfer_layers["target_model"]
    ].sort_values("mean_balanced_accuracy", ascending=False).head(1)
    lines = [
        "# Pilot Full Probe Summary",
        "",
        "This is a linear-probe diagnostic. Linear decodability means the target label is "
        "recoverable by a linear separator from hidden states; it does not establish causal "
        "use by the language model during generation. Steering and activation patching are "
        "separate later experiments.",
        "",
        "## Dataset and Activations",
        "",
        f"- Dataset size: {structure['prompt_count']} prompts.",
        f"- Template families: {balance['template_families']}.",
        "- Model conditions: base Qwen and LoRA-enabled Qwen.",
        (
            f"- Activation shape: {structure['layer_count_including_embedding']} layers including "
            f"layer 0 embedding output, {structure['transformer_block_count']} transformer blocks, "
            f"hidden dimension {structure['hidden_size']}."
        ),
        "- No generated text is involved in this stage.",
        "",
        "## Cross-Validation",
        "",
        "- Grouped stratified cross-validation by `template_family`.",
        "- Train/test family overlap was checked and must be zero for every fold.",
        "- The same held-out template families are used for base and LoRA metrics.",
        "",
        "## Best Decoding Layers",
        "",
    ]
    for model, target in [
        ("base", "stance"),
        ("lora", "stance"),
        ("base", "style"),
        ("lora", "style"),
    ]:
        item = best[(model, target)]
        cosine_column = f"{target}_cosine"
        cosine = value_at(direction_similarity, item["layer"], cosine_column)
        lines.append(
            f"- {model} {target}: layer {item['layer']}, balanced accuracy "
            f"{item['mean_balanced_accuracy']:.3f}, ROC AUC {item['mean_roc_auc']:.3f}, "
            f"base/LoRA direction cosine at that layer {cosine:.3f}."
        )
    lines.extend(
        [
            "",
            "## Transfer",
            "",
            f"- Strongest cross-model transfer: {format_transfer_row(strongest_transfer.iloc[0])}.",
            f"- Weakest cross-model transfer: {format_transfer_row(weakest_transfer.iloc[0])}.",
            "",
            "## Stance vs Style",
            "",
            (
                "- Diagnostic comparison table: "
                "`results/comparisons/pilot_full/stance_style_summary.csv`."
            ),
            f"- Mean stance direction cosine: {direction_similarity['stance_cosine'].mean():.3f}.",
            f"- Mean style direction cosine: {direction_similarity['style_cosine'].mean():.3f}.",
            "",
            "## Warnings",
            "",
            "- Layer 0 is the embedding output and should be interpreted cautiously.",
            "- Transformer layers should not be treated as independent statistical samples.",
        ]
    )
    if summary_table.isna().any().any():
        lines.append(
            "- Some retention ratios are NaN where within-model performance was near chance."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_layer(results_root: Path, model: str, target: str) -> dict[str, Any]:
    metrics = pd.read_csv(results_root / model / target / "layer_metrics.csv")
    row = metrics.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
    return {
        "layer": int(row["layer"]),
        "mean_balanced_accuracy": float(row["mean_balanced_accuracy"]),
        "mean_roc_auc": float(row["mean_roc_auc"]),
    }


def format_transfer_row(row: pd.Series) -> str:
    return (
        f"{row['target_variable']} layer {int(row['layer'])} "
        f"{row['source_model']}->{row['target_model']} balanced accuracy "
        f"{float(row['mean_balanced_accuracy']):.3f}, ROC AUC "
        f"{float(row['mean_roc_auc']):.3f}"
    )


if __name__ == "__main__":
    main()
