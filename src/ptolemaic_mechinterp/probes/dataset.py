"""Dataset preparation and grouped split validation for probes."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def encode_binary_target(metadata: pd.DataFrame, target: str) -> tuple[np.ndarray, dict[str, int]]:
    """Encode a binary string target column as 0/1 labels."""

    if target not in metadata.columns:
        raise KeyError(f"Target column not found in metadata: {target}")
    values = metadata[target]
    if values.isna().any():
        raise ValueError(f"Target column '{target}' contains missing labels.")
    classes = sorted(values.astype(str).unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Target '{target}' must contain exactly two classes, found {classes}.")
    mapping = {label: index for index, label in enumerate(classes)}
    return values.astype(str).map(mapping).to_numpy(dtype=int), mapping


def grouped_stratified_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create grouped stratified folds and validate class coverage."""

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_folds:
        raise ValueError(
            "Need at least n_folds distinct groups; "
            f"got {len(unique_groups)} groups for {n_folds} folds."
        )
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    splits = list(splitter.split(np.zeros_like(y), y, groups))
    for fold_index, (train_index, test_index) in enumerate(splits):
        train_classes = np.unique(y[train_index])
        test_classes = np.unique(y[test_index])
        if len(train_classes) < 2 or len(test_classes) < 2:
            raise ValueError(
                "Grouped split produced a single-class fold at "
                f"fold {fold_index}: train={train_classes.tolist()}, test={test_classes.tolist()}."
            )
        if set(groups[train_index]).intersection(set(groups[test_index])):
            raise RuntimeError(f"Group leakage detected at fold {fold_index}.")
    return splits


def class_balance(y: np.ndarray, label_mapping: dict[str, int]) -> dict[str, int]:
    """Count samples per original class label."""

    inverse = {value: key for key, value in label_mapping.items()}
    counts = np.bincount(y, minlength=len(label_mapping))
    return {inverse[index]: int(count) for index, count in enumerate(counts)}


def fold_diagnostics(
    metadata: pd.DataFrame,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    y: np.ndarray,
    label_mapping: dict[str, int],
    grouping_column: str,
    topic_column: str = "topic",
) -> dict[str, Any]:
    """Build leakage and distribution diagnostics for one grouped fold."""

    groups = metadata[grouping_column].astype(str).to_numpy()
    train_groups = sorted(set(groups[train_index]))
    test_groups = sorted(set(groups[test_index]))
    intersection = sorted(set(train_groups).intersection(test_groups))
    if intersection:
        raise RuntimeError(f"Group leakage detected: {intersection}.")

    inverse = {value: key for key, value in label_mapping.items()}
    train_class_counts = Counter(inverse[int(label)] for label in y[train_index])
    test_class_counts = Counter(inverse[int(label)] for label in y[test_index])
    diagnostics: dict[str, Any] = {
        "train_template_families": json.dumps(train_groups),
        "test_template_families": json.dumps(test_groups),
        "group_intersection_size": len(intersection),
        "train_class_distribution": json.dumps(dict(sorted(train_class_counts.items()))),
        "test_class_distribution": json.dumps(dict(sorted(test_class_counts.items()))),
    }

    if topic_column in metadata.columns:
        topics = metadata[topic_column].fillna("<missing>").astype(str).to_numpy()
        train_topic_counts = Counter(topics[train_index])
        test_topic_counts = Counter(topics[test_index])
        diagnostics["train_topic_distribution"] = json.dumps(
            dict(sorted(train_topic_counts.items()))
        )
        diagnostics["test_topic_distribution"] = json.dumps(
            dict(sorted(test_topic_counts.items()))
        )
    return diagnostics


def duplicate_text_diagnostics(
    metadata: pd.DataFrame,
    *,
    text_column: str = "prompt_text",
) -> dict[str, int]:
    """Count exact and normalized duplicate prompt texts after de-duplicating layers."""

    if text_column not in metadata.columns or "prompt_id" not in metadata.columns:
        return {"exact_duplicate_texts": 0, "normalized_duplicate_texts": 0}

    prompt_rows = metadata[["prompt_id", text_column]].drop_duplicates("prompt_id")
    exact_counts = Counter(prompt_rows[text_column].fillna("").astype(str))
    normalized_counts = Counter(normalize_text(text) for text in exact_counts.keys())
    return {
        "exact_duplicate_texts": sum(1 for count in exact_counts.values() if count > 1),
        "normalized_duplicate_texts": sum(
            1 for count in normalized_counts.values() if count > 1
        ),
    }


def normalize_text(text: str) -> str:
    """Normalize text for duplicate diagnostics."""

    return re.sub(r"\s+", " ", text.strip().lower())
