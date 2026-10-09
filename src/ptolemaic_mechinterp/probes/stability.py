"""Bootstrap diagnostics for linear-probe direction stability."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ptolemaic_mechinterp.probes.dataset import encode_binary_target
from ptolemaic_mechinterp.probes.linear import raw_activation_direction_from_scaled_coef

EXPECTED_LABEL_MAPPINGS = {
    "stance": {"geocentric": 0, "heliocentric": 1},
    "style": {"modern": 0, "premodern": 1},
}


@dataclass(frozen=True)
class BootstrapSamples:
    """Family-level bootstrap samples represented as prompt-row indices."""

    sampled_families: list[list[str]]
    prompt_indices: list[np.ndarray]
    family_count: int


def prompt_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one row per prompt using layer 0 as the prompt-level reference."""

    if "layer_index" not in metadata.columns:
        raise KeyError("Activation metadata must contain 'layer_index'.")
    rows = metadata[metadata["layer_index"] == 0].reset_index(drop=True)
    if rows.empty:
        raise ValueError("No layer-0 prompt rows found in activation metadata.")
    return rows


def assert_expected_label_orientation(
    metadata: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode a target and fail if its 0/1 orientation differs from the study convention."""

    y, mapping = encode_binary_target(metadata, target)
    expected = EXPECTED_LABEL_MAPPINGS.get(target)
    if expected is None:
        raise ValueError(f"No expected label orientation is configured for target: {target}")
    if mapping != expected:
        raise ValueError(
            f"Unexpected {target} label orientation: found {mapping}, expected {expected}."
        )
    return y, mapping


def draw_family_bootstrap_samples(
    metadata: pd.DataFrame,
    *,
    grouping_column: str,
    n_replicates: int,
    random_seed: int,
) -> BootstrapSamples:
    """Sample template families with replacement and expand to prompt-row indices."""

    if grouping_column not in metadata.columns:
        raise KeyError(f"Grouping column not found in metadata: {grouping_column}")
    groups = metadata[grouping_column].astype(str).to_numpy()
    families = sorted(pd.unique(groups).tolist())
    if not families:
        raise ValueError("Cannot bootstrap an empty set of template families.")
    family_to_indices = {
        family: np.flatnonzero(groups == family).astype(int) for family in families
    }
    rng = np.random.default_rng(random_seed)
    sampled_families: list[list[str]] = []
    prompt_indices: list[np.ndarray] = []
    for _ in range(n_replicates):
        sample = rng.choice(families, size=len(families), replace=True).astype(str).tolist()
        sampled_families.append(sample)
        prompt_indices.append(
            np.concatenate([family_to_indices[family] for family in sample]).astype(int)
        )
    return BootstrapSamples(
        sampled_families=sampled_families,
        prompt_indices=prompt_indices,
        family_count=len(families),
    )


def bootstrap_samples_frame(samples: BootstrapSamples) -> pd.DataFrame:
    """Describe bootstrap samples, including multiplicity of duplicated families."""

    rows: list[dict[str, Any]] = []
    for replicate, families in enumerate(samples.sampled_families):
        counts = {family: families.count(family) for family in sorted(set(families))}
        rows.append(
            {
                "replicate": replicate,
                "n_sampled_families": len(families),
                "n_unique_families": len(counts),
                "sampled_template_families": json.dumps(families),
                "template_family_multiplicities": json.dumps(counts, sort_keys=True),
                "n_prompt_rows": int(len(samples.prompt_indices[replicate])),
            }
        )
    return pd.DataFrame(rows)


def build_probe_with_c(
    *,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
    c_value: float,
) -> Pipeline:
    """Construct the probe pipeline with the same defaults as the original analysis."""

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
                    C=float(c_value),
                ),
            ),
        ]
    )


def fit_normalized_raw_direction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
    class_weight: str | None,
    c_value: float = 1.0,
) -> tuple[np.ndarray, Pipeline]:
    """Fit a standardized logistic probe and return its L2-normalized raw-space direction."""

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Probe fit requires two classes, found {classes.tolist()}.")
    probe = build_probe_with_c(
        random_seed=random_seed,
        max_iter=max_iter,
        class_weight=class_weight,
        c_value=c_value,
    )
    probe.fit(x, y)
    model = probe.named_steps["logistic_regression"]
    scaler = probe.named_steps["standardize"]
    raw_direction = raw_activation_direction_from_scaled_coef(model.coef_, scaler.scale_)
    norm = float(np.linalg.norm(raw_direction))
    if norm <= 0.0:
        raise ValueError("Fitted probe has a zero-norm raw activation direction.")
    return raw_direction / norm, probe


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a one-dimensional L2-normalized vector."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-norm vector.")
    return flat / norm


def cosine_distribution_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Summarize a cosine-similarity distribution."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p2_5": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p97_5": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p2_5": float(np.percentile(array, 2.5)),
        "p25": float(np.percentile(array, 25.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p97_5": float(np.percentile(array, 97.5)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def pairwise_direction_cosines(directions: np.ndarray) -> np.ndarray:
    """Return upper-triangle pairwise cosines among normalized direction rows."""

    matrix = np.asarray(directions, dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    if len(matrix) < 2:
        return np.array([], dtype=float)
    cosines = matrix @ matrix.T
    indices = np.triu_indices(len(matrix), k=1)
    return cosines[indices]


def matched_direction_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosines for matched normalized direction arrays."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Matched direction arrays have different shapes: {left_array.shape} vs "
            f"{right_array.shape}."
        )
    valid = np.isfinite(left_array).all(axis=1) & np.isfinite(right_array).all(axis=1)
    return np.sum(left_array[valid] * right_array[valid], axis=1)
