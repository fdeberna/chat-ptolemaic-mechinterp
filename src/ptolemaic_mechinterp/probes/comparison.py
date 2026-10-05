"""Cross-model probe comparisons and pilot-full diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ptolemaic_mechinterp.activations.storage import ActivationStore
from ptolemaic_mechinterp.probes.dataset import (
    encode_binary_target,
    grouped_stratified_splits,
)
from ptolemaic_mechinterp.probes.evaluation import binary_classification_metrics
from ptolemaic_mechinterp.probes.linear import build_probe, extract_probe_direction

METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]


def validate_activation_store_pair(
    base: ActivationStore,
    lora: ActivationStore,
    *,
    expected_prompt_count: int | None = None,
) -> dict[str, Any]:
    """Fail if base and LoRA activation stores are not structurally aligned."""

    if base.activations.shape != lora.activations.shape:
        raise ValueError(
            "Activation shapes differ: "
            f"base={base.activations.shape}, lora={lora.activations.shape}."
        )
    if not np.isfinite(base.activations).all():
        raise ValueError("Base activations contain NaN or infinite values.")
    if not np.isfinite(lora.activations).all():
        raise ValueError("LoRA activations contain NaN or infinite values.")
    if base.manifest.get("layer_zero") != "embedding_output":
        raise ValueError("Base manifest does not document layer 0 as embedding_output.")
    if lora.manifest.get("layer_zero") != "embedding_output":
        raise ValueError("LoRA manifest does not document layer 0 as embedding_output.")

    base_layers = list(base.manifest.get("layer_indices", []))
    lora_layers = list(lora.manifest.get("layer_indices", []))
    if base_layers != lora_layers:
        raise ValueError(f"Layer indices differ: base={base_layers}, lora={lora_layers}.")
    if int(base.manifest.get("hidden_size", base.activations.shape[-1])) != int(
        lora.manifest.get("hidden_size", lora.activations.shape[-1])
    ):
        raise ValueError("Hidden dimensions differ between base and LoRA manifests.")

    base_prompt_ids = prompt_ids_in_order(base.metadata)
    lora_prompt_ids = prompt_ids_in_order(lora.metadata)
    if expected_prompt_count is not None and len(base_prompt_ids) != expected_prompt_count:
        raise ValueError(
            f"Expected {expected_prompt_count} prompts, found {len(base_prompt_ids)} in base."
        )
    if expected_prompt_count is not None and len(lora_prompt_ids) != expected_prompt_count:
        raise ValueError(
            f"Expected {expected_prompt_count} prompts, found {len(lora_prompt_ids)} in LoRA."
        )
    if base_prompt_ids != lora_prompt_ids:
        raise ValueError("Base and LoRA prompt IDs are not aligned in the same order.")

    comparable_columns = [
        "prompt_id",
        "layer_index",
        "stance",
        "style",
        "template_family",
        "framework",
        "attribution",
        "prompt_text",
        "topic",
        "proposition",
    ]
    for column in comparable_columns:
        if column in base.metadata.columns and column in lora.metadata.columns:
            left = base.metadata[column].fillna("").astype(str).tolist()
            right = lora.metadata[column].fillna("").astype(str).tolist()
            if left != right:
                raise ValueError(f"Base and LoRA metadata differ for column: {column}.")

    return {
        "prompt_count": len(base_prompt_ids),
        "layer_count_including_embedding": len(base_layers),
        "transformer_block_count": len(base_layers) - 1,
        "hidden_size": int(base.activations.shape[1]),
        "layer_zero": "embedding_output",
    }


def prompt_ids_in_order(metadata: pd.DataFrame) -> list[str]:
    """Return prompt IDs from layer 0, preserving activation-store order."""

    return (
        metadata.loc[metadata["layer_index"] == 0, "prompt_id"]
        .astype(str)
        .reset_index(drop=True)
        .tolist()
    )


def validate_factor_balance(metadata: pd.DataFrame) -> dict[str, Any]:
    """Validate stance/style balance and 2x2 coverage on prompt-level rows."""

    prompt_rows = metadata[metadata["layer_index"] == 0].copy()
    stance_counts = prompt_rows["stance"].astype(str).value_counts().sort_index().to_dict()
    style_counts = prompt_rows["style"].astype(str).value_counts().sort_index().to_dict()
    combo_counts = (
        prompt_rows.groupby(["stance", "style"], dropna=False).size().sort_index().to_dict()
    )
    if len(set(stance_counts.values())) != 1:
        raise ValueError(f"Stance labels are not balanced: {stance_counts}.")
    if len(set(style_counts.values())) != 1:
        raise ValueError(f"Style labels are not balanced: {style_counts}.")
    if len(combo_counts) != 4 or len(set(combo_counts.values())) != 1:
        raise ValueError(f"Stance x style combinations are not balanced: {combo_counts}.")
    return {
        "stance_counts": stance_counts,
        "style_counts": style_counts,
        "stance_style_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in combo_counts.items()},
        "template_families": int(prompt_rows["template_family"].nunique()),
    }


def aggregate_layer_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level probe metrics to one row per layer."""

    rows: list[dict[str, Any]] = []
    for layer_index, layer_rows in fold_metrics.groupby("layer_index", sort=True):
        row: dict[str, Any] = {"layer_index": int(layer_index), "layer": int(layer_index)}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(layer_rows[metric].mean())
            row[f"std_{metric}"] = float(layer_rows[metric].std(ddof=1))
        row["n_folds"] = int(layer_rows["fold"].nunique())
        row["n_evaluated_prompts"] = int(layer_rows["n_test"].sum())
        row["n_template_families"] = count_test_families(layer_rows)
        rows.append(row)
    return pd.DataFrame(rows)


def count_test_families(rows: pd.DataFrame) -> int:
    families: set[str] = set()
    for value in rows["test_template_families"]:
        families.update(json.loads(value))
    return len(families)


def assert_same_group_splits(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Fail if two fold-metric tables used different held-out groups."""

    keys = ["layer_index", "fold"]
    left_sorted = left.sort_values(keys).reset_index(drop=True)
    right_sorted = right.sort_values(keys).reset_index(drop=True)
    for column in ["train_template_families", "test_template_families"]:
        if left_sorted[column].tolist() != right_sorted[column].tolist():
            raise ValueError(f"Group splits differ for column: {column}.")


def parse_label_mapping(value: Any) -> dict[str, int]:
    """Parse a label mapping stored in fold metrics."""

    if isinstance(value, dict):
        return {str(key): int(item) for key, item in value.items()}
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label mapping is not valid JSON: {text}") from exc
    return {str(key): int(item) for key, item in parsed.items()}


def label_mapping_from_metrics(metrics: pd.DataFrame) -> dict[str, int]:
    values = metrics["label_mapping"].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one label mapping, found {values}.")
    return parse_label_mapping(values[0])


def assert_label_mapping_equal(left: dict[str, int], right: dict[str, int], *, target: str) -> None:
    if left != right:
        raise ValueError(f"Label encoding differs for {target}: base={left}, lora={right}.")


def positive_class(label_mapping: dict[str, int]) -> str:
    positives = [label for label, encoded in label_mapping.items() if encoded == 1]
    if len(positives) != 1:
        raise ValueError(f"Expected exactly one positive class in {label_mapping}.")
    return positives[0]


def train_full_data_probe_directions(
    store: ActivationStore,
    *,
    target: str,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Train final full-data probes for direction analysis only."""

    y_all, label_mapping = encode_binary_target(store.metadata, target)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    directions: dict[int, np.ndarray] = {}
    for layer_index in sorted(store.metadata["layer_index"].unique()):
        mask = store.metadata["layer_index"].to_numpy() == layer_index
        probe = build_probe(random_seed=random_seed, max_iter=max_iter, class_weight=class_weight)
        probe.fit(store.activations[mask], y_all[mask])
        model = probe.named_steps["logistic_regression"]
        scaler = probe.named_steps["standardize"]
        direction = extract_probe_direction(probe)
        norm = float(np.linalg.norm(direction))
        directions[int(layer_index)] = direction
        np.savez_compressed(
            output_dir / f"layer_{int(layer_index):03d}_full_data.npz",
            coef=model.coef_,
            intercept=model.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            activation_space_direction=direction,
            label_mapping=json.dumps(label_mapping, sort_keys=True),
            positive_class=positive_class(label_mapping),
        )
        rows.append(
            {
                "layer": int(layer_index),
                "target": target,
                "label_mapping": json.dumps(label_mapping, sort_keys=True),
                "positive_class": positive_class(label_mapping),
                "direction_norm": norm,
                "coefficient_file": str(output_dir / f"layer_{int(layer_index):03d}_full_data.npz"),
            }
        )
    return pd.DataFrame(rows), directions


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero-norm vector.")
    return float(np.dot(left, right) / (left_norm * right_norm))


def cross_model_probe_transfer(
    source: ActivationStore,
    target: ActivationStore,
    *,
    source_model: str,
    target_model: str,
    target_variable: str,
    grouping_column: str,
    n_folds: int,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
) -> pd.DataFrame:
    """Train on source activations and evaluate on target activations using source scaling."""

    validate_same_prompt_layer_labels(source, target, target_variable, grouping_column)
    source_y_all, source_mapping = encode_binary_target(source.metadata, target_variable)
    target_y_all, target_mapping = encode_binary_target(target.metadata, target_variable)
    assert_label_mapping_equal(source_mapping, target_mapping, target=target_variable)
    if not np.array_equal(source_y_all, target_y_all):
        raise ValueError("Source and target encoded labels differ after alignment checks.")

    rows: list[dict[str, Any]] = []
    layer0_mask = source.metadata["layer_index"].to_numpy() == 0
    base_groups = source.metadata.loc[layer0_mask, grouping_column].astype(str).to_numpy()
    base_y = source_y_all[layer0_mask]
    splits = grouped_stratified_splits(
        base_y, base_groups, n_folds=n_folds, random_seed=random_seed
    )

    for layer_index in sorted(source.metadata["layer_index"].unique()):
        source_mask = source.metadata["layer_index"].to_numpy() == layer_index
        target_mask = target.metadata["layer_index"].to_numpy() == layer_index
        x_source = source.activations[source_mask]
        x_target = target.activations[target_mask]
        y_source = source_y_all[source_mask]
        y_target = target_y_all[target_mask]
        groups = source.metadata.loc[source_mask, grouping_column].astype(str).to_numpy()
        for fold_index, (train_index, test_index) in enumerate(splits):
            train_families = sorted(set(groups[train_index]))
            test_families = sorted(set(groups[test_index]))
            if set(train_families).intersection(test_families):
                raise RuntimeError(f"Group leakage detected in transfer fold {fold_index}.")
            probe = build_probe(
                random_seed=random_seed, max_iter=max_iter, class_weight=class_weight
            )
            probe.fit(x_source[train_index], y_source[train_index])
            predictions = probe.predict(x_target[test_index])
            scores = probe.predict_proba(x_target[test_index])[:, 1]
            metrics = binary_classification_metrics(y_target[test_index], predictions, scores)
            row: dict[str, Any] = {
                "layer": int(layer_index),
                "target_variable": target_variable,
                "source_model": source_model,
                "target_model": target_model,
                "fold": int(fold_index),
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "train_family_ids": json.dumps(train_families),
                "test_family_ids": json.dumps(test_families),
                "label_mapping": json.dumps(source_mapping, sort_keys=True),
                "positive_class": positive_class(source_mapping),
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def validate_same_prompt_layer_labels(
    source: ActivationStore,
    target: ActivationStore,
    target_variable: str,
    grouping_column: str,
) -> None:
    for column in ["prompt_id", "layer_index", target_variable, grouping_column]:
        if source.metadata[column].fillna("").astype(str).tolist() != target.metadata[
            column
        ].fillna("").astype(str).tolist():
            raise ValueError(f"Source and target stores are misaligned for column: {column}.")


def aggregate_transfer_layers(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["layer", "target_variable", "source_model", "target_model"]
    for keys, group in folds.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
        row["n_folds"] = int(group["fold"].nunique())
        row["n_evaluated_prompts"] = int(group["n_test"].sum())
        row["n_template_families"] = count_transfer_test_families(group)
        rows.append(row)
    aggregated = pd.DataFrame(rows)
    return add_transfer_retention(aggregated)


def count_transfer_test_families(rows: pd.DataFrame) -> int:
    families: set[str] = set()
    for value in rows["test_family_ids"]:
        families.update(json.loads(value))
    return len(families)


def add_transfer_retention(aggregated: pd.DataFrame) -> pd.DataFrame:
    result = aggregated.copy()
    result["balanced_accuracy_transfer_ratio"] = np.nan
    result["roc_auc_transfer_ratio"] = np.nan
    result["balanced_accuracy_chance_corrected_retention"] = np.nan
    for index, row in result.iterrows():
        if row["source_model"] == row["target_model"]:
            continue
        baseline = result[
            (result["layer"] == row["layer"])
            & (result["target_variable"] == row["target_variable"])
            & (result["source_model"] == row["source_model"])
            & (result["target_model"] == row["source_model"])
        ]
        if len(baseline) != 1:
            raise ValueError(f"Missing within-model baseline for transfer row: {row.to_dict()}")
        base_row = baseline.iloc[0]
        for metric, output in [
            ("balanced_accuracy", "balanced_accuracy_transfer_ratio"),
            ("roc_auc", "roc_auc_transfer_ratio"),
        ]:
            denom = float(base_row[f"mean_{metric}"])
            if abs(denom) > 1e-12 and abs(denom - 0.5) > 1e-6:
                result.at[index, output] = float(row[f"mean_{metric}"]) / denom
        within_minus_chance = float(base_row["mean_balanced_accuracy"]) - 0.5
        if within_minus_chance > 1e-6:
            result.at[index, "balanced_accuracy_chance_corrected_retention"] = (
                float(row["mean_balanced_accuracy"]) - 0.5
            ) / within_minus_chance
    return result
