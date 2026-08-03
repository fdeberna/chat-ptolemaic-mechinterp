from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ptolemaic_mechinterp.activations.extraction import (
    ExtractedActivations,
    final_non_padding_token_indices,
)
from ptolemaic_mechinterp.activations.storage import load_activation_store, save_activation_store


def test_final_non_padding_token_indices() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )

    indices = final_non_padding_token_indices(attention_mask)

    assert indices.tolist() == [2, 0, 1]


def test_final_non_padding_token_indices_rejects_empty_prompt() -> None:
    attention_mask = torch.tensor([[0, 0]])

    with pytest.raises(ValueError, match="at least one"):
        final_non_padding_token_indices(attention_mask)


def test_activation_save_load_round_trip(tmp_path: Path) -> None:
    activations = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2"],
            "layer_index": [0, 1, 0],
            "model_condition": ["base", "base", "base"],
            "stance": ["geocentric", "geocentric", "heliocentric"],
            "template_family": ["a", "a", "b"],
        }
    )
    extracted = ExtractedActivations(
        activations=activations,
        metadata=metadata,
        layer_indices=[0, 1],
        hidden_size=4,
    )

    save_activation_store(tmp_path, extracted, extraction_config={"condition": "base"})
    loaded = load_activation_store(tmp_path)

    np.testing.assert_array_equal(loaded.activations, activations)
    pd.testing.assert_frame_equal(loaded.metadata, metadata)
    assert loaded.manifest["layer_zero"] == "embedding_output"
    assert loaded.manifest["extraction_config"] == {"condition": "base"}

