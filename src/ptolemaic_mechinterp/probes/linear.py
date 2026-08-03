"""Layer-wise logistic-regression probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ptolemaic_mechinterp.activations.storage import ActivationStore
from ptolemaic_mechinterp.probes.dataset import (
    class_balance,
    encode_binary_target,
    grouped_stratified_splits,
)
from ptolemaic_mechinterp.probes.evaluation import binary_classification_metrics


def train_layerwise_logistic_probes(
    store: ActivationStore,
    *,
    target: str,
    grouping_column: str,
    n_folds: int,
    random_seed: int,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    coefficient_output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate one logistic-regression probe per layer."""

    metadata = store.metadata
    if "layer_index" not in metadata.columns:
        raise KeyError("Activation metadata must contain 'layer_index'.")
    if grouping_column not in metadata.columns:
        raise KeyError(f"Grouping column not found in metadata: {grouping_column}")

    y_all, label_mapping = encode_binary_target(metadata, target)
    rows: list[dict[str, Any]] = []
    coefficient_dir = Path(coefficient_output_dir) if coefficient_output_dir else None
    if coefficient_dir:
        coefficient_dir.mkdir(parents=True, exist_ok=True)

    for layer_index in sorted(metadata["layer_index"].unique()):
        layer_mask = metadata["layer_index"].to_numpy() == layer_index
        x_layer = store.activations[layer_mask]
        y_layer = y_all[layer_mask]
        groups = metadata.loc[layer_mask, grouping_column].astype(str).to_numpy()
        splits = grouped_stratified_splits(
            y_layer,
            groups,
            n_folds=n_folds,
            random_seed=random_seed,
        )
        balance = class_balance(y_layer, label_mapping)
        for fold_index, (train_index, test_index) in enumerate(splits):
            probe = build_probe(
                random_seed=random_seed,
                max_iter=max_iter,
                class_weight=class_weight,
            )
            probe.fit(x_layer[train_index], y_layer[train_index])
            predictions = probe.predict(x_layer[test_index])
            scores = probe.predict_proba(x_layer[test_index])[:, 1]
            metrics = binary_classification_metrics(y_layer[test_index], predictions, scores)
            row: dict[str, Any] = {
                "layer_index": int(layer_index),
                "fold": fold_index,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "target": target,
                "grouping_column": grouping_column,
                "label_mapping": label_mapping,
            }
            row.update({f"class_count_{label}": count for label, count in balance.items()})
            row.update(metrics)
            rows.append(row)
            if coefficient_dir:
                save_probe_coefficients(coefficient_dir, int(layer_index), fold_index, probe)

    return pd.DataFrame(rows)


def build_probe(
    *,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
) -> Pipeline:
    """Construct the standardized logistic-regression probe pipeline."""

    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    solver="liblinear",
                    random_state=random_seed,
                    max_iter=max_iter,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def save_probe_coefficients(
    output_dir: Path,
    layer_index: int,
    fold_index: int,
    probe: Pipeline,
) -> None:
    """Save fitted logistic-regression coefficients for a layer/fold."""

    model = probe.named_steps["logistic_regression"]
    np.savez_compressed(
        output_dir / f"layer_{layer_index:03d}_fold_{fold_index:02d}.npz",
        coef=model.coef_,
        intercept=model.intercept_,
    )


def extract_probe_direction(probe: Pipeline) -> np.ndarray:
    """Return the learned linear direction from a fitted probe."""

    model = probe.named_steps["logistic_regression"]
    return np.asarray(model.coef_).reshape(-1)


def train_on_source_evaluate_on_target(*args: object, **kwargs: object) -> pd.DataFrame:
    """Train probes on one model condition and evaluate on another.

    This is the intended entry point for base-to-LoRA and LoRA-to-base transfer experiments. It
    needs explicit alignment checks for prompt IDs, layer numbering, and target labels before use.
    """

    raise NotImplementedError("Cross-model probe transfer is not implemented in the minimal slice.")
