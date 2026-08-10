"""Residual-stream activation extraction using returned hidden states."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from ptolemaic_mechinterp.data.schemas import PromptRecord


@dataclass(frozen=True)
class ExtractedActivations:
    """Flattened layer-wise prompt activations and aligned metadata."""

    activations: np.ndarray
    metadata: pd.DataFrame
    layer_indices: list[int]
    hidden_size: int
    layer_zero: str = "embedding_output"


def final_non_padding_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final non-padding token index for each sequence."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence].")
    lengths = attention_mask.long().sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("Every prompt must contain at least one non-padding token.")
    return lengths - 1


@torch.inference_mode()
def extract_final_token_hidden_states(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[PromptRecord],
    *,
    condition: str,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    max_length: int | None = None,
    max_batches: int | None = None,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
) -> ExtractedActivations:
    """Extract every returned hidden-state layer at each prompt's final non-padding token.

    Layer 0 is the embedding output returned by Hugging Face models. Transformer block outputs
    are layers 1..N.
    """

    if not prompts:
        raise ValueError("At least one prompt is required for activation extraction.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1 when provided.")

    model_device = torch.device(device) if device else infer_model_device(model)
    rows: list[dict[str, Any]] = []
    activation_chunks: list[np.ndarray] = []
    expected_layer_count: int | None = None
    hidden_size: int | None = None

    config_num_hidden_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    batch_count = 0

    for start in range(0, len(prompts), batch_size):
        if max_batches is not None and batch_count >= max_batches:
            break
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            [record.text for record in batch],
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        if expected_layer_count is None:
            expected_layer_count = len(hidden_states)
            if (
                config_num_hidden_layers is not None
                and expected_layer_count != int(config_num_hidden_layers) + 1
            ):
                raise RuntimeError(
                    "Model returned an unexpected number of hidden-state tensors: "
                    f"got {expected_layer_count}, expected {int(config_num_hidden_layers) + 1} "
                    "including embedding output layer 0."
                )
        elif len(hidden_states) != expected_layer_count:
            raise RuntimeError("Model returned an inconsistent number of hidden-state layers.")

        token_indices = final_non_padding_token_indices(encoded["attention_mask"])
        per_layer: list[torch.Tensor] = []
        for hidden in hidden_states:
            if hidden.ndim != 3:
                raise RuntimeError(
                    "Each hidden-state tensor must have shape [batch, sequence, hidden]."
                )
            if hidden.shape[0] != len(batch):
                raise RuntimeError("Hidden-state batch size does not match prompt batch size.")
            if hidden_size is not None and int(hidden.shape[-1]) != hidden_size:
                raise RuntimeError("Hidden-state size changed across batches or layers.")
            selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), token_indices, :]
            per_layer.append(selected.detach().float().cpu())

        stacked = torch.stack(per_layer, dim=1).numpy()
        current_hidden_size = int(stacked.shape[-1])
        if hidden_size is None:
            hidden_size = current_hidden_size
        elif current_hidden_size != hidden_size:
            raise RuntimeError("Activation hidden size changed across batches.")
        activation_chunks.append(stacked.reshape(-1, hidden_size))

        for record in batch:
            for layer_index in range(expected_layer_count):
                rows.append(
                    {
                        "prompt_id": record.prompt_id,
                        "layer_index": layer_index,
                        "model_condition": condition,
                        "model_name_or_path": model_name_or_path,
                        "adapter_path": adapter_path,
                        "stance": record.stance,
                        "template_family": record.template_family,
                        "style": record.style,
                        "framework": record.framework,
                        "attribution": record.attribution,
                        "prompt_text": record.text,
                        "topic": record.metadata.get("topic"),
                        "proposition": record.metadata.get("proposition"),
                    }
                )
        batch_count += 1
        del encoded, outputs, hidden_states, per_layer, stacked

    if expected_layer_count is None or hidden_size is None:
        raise RuntimeError("No activations were extracted.")
    activations = np.concatenate(activation_chunks, axis=0)
    metadata = pd.DataFrame(rows)
    if len(metadata) != activations.shape[0]:
        raise RuntimeError("Activation rows and metadata rows are misaligned.")
    return ExtractedActivations(
        activations=activations,
        metadata=metadata,
        layer_indices=list(range(expected_layer_count)),
        hidden_size=hidden_size,
    )


def infer_model_device(model: torch.nn.Module) -> torch.device:
    """Infer the device of a model from its first parameter."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
