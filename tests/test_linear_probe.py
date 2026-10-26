import numpy as np
import pandas as pd
import pytest

from ptolemaic_mechinterp.activations.storage import ActivationStore
from ptolemaic_mechinterp.probes.dataset import encode_binary_target, grouped_stratified_splits
from ptolemaic_mechinterp.probes.linear import (
    build_probe,
    extract_probe_direction,
    raw_activation_direction_from_scaled_coef,
    train_layerwise_logistic_probes,
)


def test_grouped_splits_prevent_template_family_leakage() -> None:
    y = np.array([0, 1] * 6)
    groups = np.repeat([f"family_{index}" for index in range(6)], 2)

    splits = grouped_stratified_splits(y, groups, n_folds=3, random_seed=0)

    assert len(splits) == 3
    for train_index, test_index in splits:
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
        assert set(np.unique(y[train_index])) == {0, 1}
        assert set(np.unique(y[test_index])) == {0, 1}


def test_grouped_splits_fail_on_single_class_fold() -> None:
    y = np.array([0, 0, 1, 1, 0, 0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    with pytest.raises(ValueError, match="single-class fold"):
        grouped_stratified_splits(y, groups, n_folds=3, random_seed=0)


def test_encode_binary_target_rejects_non_binary_target() -> None:
    metadata = pd.DataFrame({"stance": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="exactly two"):
        encode_binary_target(metadata, "stance")


def test_layerwise_probe_detects_synthetic_signal() -> None:
    store = synthetic_activation_store()

    metrics = train_layerwise_logistic_probes(
        store,
        target="stance",
        grouping_column="template_family",
        n_folds=3,
        random_seed=0,
        max_iter=200,
        class_weight="balanced",
    )

    layer_summary = metrics.groupby("layer_index")["balanced_accuracy"].mean()
    assert layer_summary.loc[1] > 0.95
    assert set(metrics["layer_index"]) == {0, 1, 2}


def test_layerwise_probe_rejects_template_leakage_by_construction() -> None:
    store = synthetic_activation_store()
    metadata = store.metadata
    layer_mask = metadata["layer_index"].to_numpy() == 0
    groups = metadata.loc[layer_mask, "template_family"].to_numpy()
    y = np.array(
        [0 if label == "geocentric" else 1 for label in metadata.loc[layer_mask, "stance"]]
    )

    train_index, test_index = grouped_stratified_splits(y, groups, n_folds=3, random_seed=0)[0]

    assert set(groups[train_index]).isdisjoint(set(groups[test_index]))


def test_raw_activation_direction_from_scaled_coef_undoes_standard_scaler_scale() -> None:
    raw_direction = np.array([1.0, 1.0])
    scale_a = np.array([1.0, 100.0])
    scale_b = np.array([100.0, 1.0])
    scaled_direction_a = raw_direction * scale_a
    scaled_direction_b = raw_direction * scale_b

    scaled_cosine = float(
        np.dot(scaled_direction_a, scaled_direction_b)
        / (np.linalg.norm(scaled_direction_a) * np.linalg.norm(scaled_direction_b))
    )
    recovered_a = raw_activation_direction_from_scaled_coef(scaled_direction_a, scale_a)
    recovered_b = raw_activation_direction_from_scaled_coef(scaled_direction_b, scale_b)
    raw_cosine = float(
        np.dot(recovered_a, recovered_b)
        / (np.linalg.norm(recovered_a) * np.linalg.norm(recovered_b))
    )

    assert scaled_cosine < 0.03
    assert np.allclose(recovered_a, raw_direction)
    assert np.allclose(recovered_b, raw_direction)
    assert raw_cosine == pytest.approx(1.0)


@pytest.mark.parametrize(
    "scale",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
    ],
)
def test_raw_activation_direction_rejects_invalid_scaler_scale(scale: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Scaler scale_"):
        raw_activation_direction_from_scaled_coef(np.array([1.0, 2.0]), scale)


def test_extract_probe_direction_returns_raw_activation_space_direction() -> None:
    x = np.array(
        [
            [-10.0, -1.0],
            [-9.0, 1.0],
            [9.0, -1.0],
            [10.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    probe = build_probe(random_seed=0, max_iter=200, class_weight=None)
    probe.fit(x, y)

    model = probe.named_steps["logistic_regression"]
    scaler = probe.named_steps["standardize"]
    expected = np.asarray(model.coef_).reshape(-1) / scaler.scale_

    assert np.allclose(extract_probe_direction(probe), expected)


def synthetic_activation_store() -> ActivationStore:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    activations: list[np.ndarray] = []
    layer_count = 3
    hidden_size = 5
    prompt_index = 0
    for family_index in range(12):
        for stance, label in [("geocentric", 0), ("heliocentric", 1)]:
            for layer_index in range(layer_count):
                vector = rng.normal(scale=0.05, size=hidden_size)
                if layer_index == 1:
                    vector[0] = -10.0 if label == 0 else 10.0
                rows.append(
                    {
                        "prompt_id": f"p{prompt_index}",
                        "layer_index": layer_index,
                        "model_condition": "base",
                        "stance": stance,
                        "template_family": f"family_{family_index}",
                    }
                )
                activations.append(vector)
            prompt_index += 1
    return ActivationStore(
        activations=np.asarray(activations, dtype=np.float32),
        metadata=pd.DataFrame(rows),
        manifest={"layer_indices": [0, 1, 2]},
    )
