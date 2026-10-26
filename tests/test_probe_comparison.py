import json

import numpy as np
import pandas as pd
import pytest

from ptolemaic_mechinterp.activations.storage import ActivationStore
from ptolemaic_mechinterp.probes.comparison import (
    add_transfer_retention,
    assert_same_group_splits,
    audited_probe_direction_similarity,
    build_audited_direction_similarity,
    cosine_similarity,
    cross_model_probe_transfer,
    parse_label_mapping,
    validate_activation_store_pair,
)


def test_cosine_similarity_uses_normalized_vectors() -> None:
    assert cosine_similarity(np.array([2.0, 0.0]), np.array([10.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 3.0])) == pytest.approx(0.0)


def test_parse_label_mapping_requires_json() -> None:
    assert parse_label_mapping('{"geocentric": 0, "heliocentric": 1}') == {
        "geocentric": 0,
        "heliocentric": 1,
    }
    with pytest.raises(ValueError, match="valid JSON"):
        parse_label_mapping("{'geocentric': 0, 'heliocentric': 1}")


def test_cross_model_transfer_reuses_source_scaler() -> None:
    source = synthetic_store(condition="base", target_shift=0.0)
    target = synthetic_store(condition="lora", target_shift=100.0)

    metrics = cross_model_probe_transfer(
        source,
        target,
        source_model="base",
        target_model="lora",
        target_variable="stance",
        grouping_column="template_family",
        n_folds=3,
        random_seed=0,
        max_iter=200,
        class_weight="balanced",
    )

    assert metrics["balanced_accuracy"].mean() < 0.75


def test_same_group_splits_detects_mismatch() -> None:
    left = pd.DataFrame(
        {
            "layer_index": [0],
            "fold": [0],
            "train_template_families": [json.dumps(["a"])],
            "test_template_families": [json.dumps(["b"])],
        }
    )
    right = left.copy()
    right["test_template_families"] = [json.dumps(["c"])]

    with pytest.raises(ValueError, match="Group splits differ"):
        assert_same_group_splits(left, right)


def test_retention_omits_near_chance_denominator() -> None:
    frame = pd.DataFrame(
        [
            {
                "layer": 0,
                "target_variable": "stance",
                "source_model": "base",
                "target_model": "base",
                "mean_balanced_accuracy": 0.5,
                "mean_roc_auc": 0.5,
            },
            {
                "layer": 0,
                "target_variable": "stance",
                "source_model": "base",
                "target_model": "lora",
                "mean_balanced_accuracy": 0.6,
                "mean_roc_auc": 0.6,
            },
        ]
    )

    result = add_transfer_retention(frame)
    transfer = result[result["target_model"] == "lora"].iloc[0]

    assert np.isnan(transfer["balanced_accuracy_transfer_ratio"])
    assert np.isnan(transfer["balanced_accuracy_chance_corrected_retention"])


def test_activation_pair_fails_on_misaligned_prompt_ids() -> None:
    base = synthetic_store(condition="base", target_shift=0.0)
    lora = synthetic_store(condition="lora", target_shift=0.0)
    lora.metadata.loc[lora.metadata["prompt_id"] == "p0", "prompt_id"] = "different"

    with pytest.raises(ValueError, match="prompt IDs"):
        validate_activation_store_pair(base, lora)


def test_audited_probe_direction_similarity_reports_scaled_and_raw_spaces(tmp_path) -> None:
    results_root = tmp_path / "probes"
    raw_direction = np.array([1.0, 1.0])
    write_probe_artifact(
        results_root,
        model_name="base",
        target="stance",
        layer=0,
        scaled_direction=raw_direction * np.array([1.0, 100.0]),
        scaler_scale=np.array([1.0, 100.0]),
        label_mapping={"geocentric": 0, "heliocentric": 1},
    )
    write_probe_artifact(
        results_root,
        model_name="lora",
        target="stance",
        layer=0,
        scaled_direction=raw_direction * np.array([100.0, 1.0]),
        scaler_scale=np.array([100.0, 1.0]),
        label_mapping={"geocentric": 0, "heliocentric": 1},
    )

    row = audited_probe_direction_similarity(results_root=results_root, target="stance", layer=0)

    assert row["scaled_space_cosine"] < 0.03
    assert row["raw_activation_space_cosine"] == pytest.approx(1.0)
    assert row["base_positive_class"] == "heliocentric"
    assert row["lora_positive_class"] == "heliocentric"


def test_audited_probe_direction_similarity_detects_opposite_label_encoding(tmp_path) -> None:
    results_root = tmp_path / "probes"
    write_probe_artifact(
        results_root,
        model_name="base",
        target="stance",
        layer=0,
        scaled_direction=np.array([1.0, 2.0]),
        scaler_scale=np.array([1.0, 1.0]),
        label_mapping={"geocentric": 0, "heliocentric": 1},
    )
    write_probe_artifact(
        results_root,
        model_name="lora",
        target="stance",
        layer=0,
        scaled_direction=np.array([1.0, 2.0]),
        scaler_scale=np.array([1.0, 1.0]),
        label_mapping={"geocentric": 1, "heliocentric": 0},
    )

    with pytest.raises(ValueError, match="Label encoding differs"):
        audited_probe_direction_similarity(results_root=results_root, target="stance", layer=0)


def test_build_audited_direction_similarity_from_artifacts(tmp_path) -> None:
    results_root = tmp_path / "probes"
    for target, label_mapping in [
        ("stance", {"geocentric": 0, "heliocentric": 1}),
        ("style", {"modern": 0, "premodern": 1}),
    ]:
        for model_name in ["base", "lora"]:
            write_probe_artifact(
                results_root,
                model_name=model_name,
                target=target,
                layer=0,
                scaled_direction=np.array([1.0, 0.0]),
                scaler_scale=np.array([1.0, 1.0]),
                label_mapping=label_mapping,
            )

    audited = build_audited_direction_similarity(results_root=results_root)

    assert audited[["layer", "target"]].to_dict("records") == [
        {"layer": 0, "target": "stance"},
        {"layer": 0, "target": "style"},
    ]
    assert audited["raw_activation_space_cosine"].tolist() == pytest.approx([1.0, 1.0])


def synthetic_store(*, condition: str, target_shift: float) -> ActivationStore:
    rng = np.random.default_rng(0)
    rows = []
    activations = []
    prompt_index = 0
    for family_index in range(12):
        for stance, label in [("geocentric", 0), ("heliocentric", 1)]:
            prompt_id = f"p{prompt_index}"
            prompt_index += 1
            for layer_index in [0, 1]:
                x0 = -2.0 if label == 0 else 2.0
                if condition == "lora":
                    x0 = -x0 + target_shift
                vector = np.array([x0, rng.normal(scale=0.01)], dtype=np.float32)
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "layer_index": layer_index,
                        "model_condition": condition,
                        "stance": stance,
                        "style": "modern" if label == 0 else "premodern",
                        "template_family": f"family_{family_index}",
                        "prompt_text": f"text {prompt_id}",
                    }
                )
                activations.append(vector)
    return ActivationStore(
        activations=np.asarray(activations),
        metadata=pd.DataFrame(rows),
        manifest={"layer_indices": [0, 1], "hidden_size": 2, "layer_zero": "embedding_output"},
    )


def write_probe_artifact(
    results_root,
    *,
    model_name: str,
    target: str,
    layer: int,
    scaled_direction: np.ndarray,
    scaler_scale: np.ndarray,
    label_mapping: dict[str, int],
) -> None:
    output_dir = results_root / model_name / target / "full_data_coefficients"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_direction = np.asarray(scaled_direction, dtype=float) / np.asarray(
        scaler_scale,
        dtype=float,
    )
    positive = [label for label, encoded in label_mapping.items() if encoded == 1][0]
    np.savez_compressed(
        output_dir / f"layer_{layer:03d}_full_data.npz",
        coef=np.asarray(scaled_direction, dtype=float).reshape(1, -1),
        intercept=np.array([0.0]),
        scaler_mean=np.zeros_like(scaler_scale, dtype=float),
        scaler_scale=np.asarray(scaler_scale, dtype=float),
        activation_space_direction=raw_direction,
        label_mapping=json.dumps(label_mapping, sort_keys=True),
        positive_class=positive,
    )
