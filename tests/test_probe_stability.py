from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptolemaic_mechinterp.probes.stability import (
    assert_expected_label_orientation,
    draw_family_bootstrap_samples,
    fit_normalized_raw_direction,
)


def test_family_bootstrap_preserves_duplicate_family_multiplicity() -> None:
    metadata = pd.DataFrame(
        {
            "template_family": ["a", "a", "b", "b"],
            "prompt_id": ["a1", "a2", "b1", "b2"],
        }
    )
    samples = draw_family_bootstrap_samples(
        metadata,
        grouping_column="template_family",
        n_replicates=1,
        random_seed=0,
    )

    sampled = samples.sampled_families[0]
    indices = samples.prompt_indices[0]

    assert len(sampled) == 2
    assert len(indices) == 4
    for family in set(sampled):
        family_rows = np.flatnonzero(metadata["template_family"].to_numpy() == family)
        expected_count = sampled.count(family)
        for row_index in family_rows:
            assert int(np.sum(indices == row_index)) == expected_count


def test_expected_label_orientation_accepts_study_convention() -> None:
    metadata = pd.DataFrame({"stance": ["geocentric", "heliocentric"]})

    y, mapping = assert_expected_label_orientation(metadata, "stance")

    assert mapping == {"geocentric": 0, "heliocentric": 1}
    assert y.tolist() == [0, 1]


def test_expected_label_orientation_rejects_wrong_target_labels() -> None:
    metadata = pd.DataFrame({"style": ["ancient", "modern"]})

    with pytest.raises(ValueError, match="Unexpected style label orientation"):
        assert_expected_label_orientation(metadata, "style")


def test_fit_normalized_raw_direction_returns_unit_raw_space_direction() -> None:
    x = np.array(
        [
            [-4.0, -0.1],
            [-3.0, 0.1],
            [3.0, -0.1],
            [4.0, 0.1],
        ]
    )
    y = np.array([0, 0, 1, 1])

    direction, probe = fit_normalized_raw_direction(
        x,
        y,
        random_seed=0,
        max_iter=200,
        class_weight=None,
        c_value=1.0,
    )
    model = probe.named_steps["logistic_regression"]
    scaler = probe.named_steps["standardize"]
    expected_raw = model.coef_.reshape(-1) / scaler.scale_
    expected_raw = expected_raw / np.linalg.norm(expected_raw)

    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert np.allclose(direction, expected_raw)
