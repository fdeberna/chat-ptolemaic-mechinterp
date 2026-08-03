"""Dataset preparation and grouped split validation for probes."""

from __future__ import annotations

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
