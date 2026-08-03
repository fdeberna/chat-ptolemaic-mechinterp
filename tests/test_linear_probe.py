import numpy as np
import pandas as pd
import pytest

from ptolemaic_mechinterp.activations.storage import ActivationStore
from ptolemaic_mechinterp.probes.dataset import encode_binary_target, grouped_stratified_splits
from ptolemaic_mechinterp.probes.linear import train_layerwise_logistic_probes


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
