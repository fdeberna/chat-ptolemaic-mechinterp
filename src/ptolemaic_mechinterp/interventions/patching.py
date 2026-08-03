"""Activation patching interfaces."""

from __future__ import annotations

import torch


def patch_activation(
    target_hidden_state: torch.Tensor,
    source_hidden_state: torch.Tensor,
    token_index: int,
) -> torch.Tensor:
    """Return a copy of target hidden states with one token patched from a source run."""

    if target_hidden_state.shape != source_hidden_state.shape:
        raise ValueError("Source and target hidden states must have identical shapes.")
    patched = target_hidden_state.clone()
    patched[:, token_index, :] = source_hidden_state[:, token_index, :]
    return patched


def run_cross_model_activation_patching(*args: object, **kwargs: object) -> None:
    """Patch activations across base and LoRA model conditions."""

    raise NotImplementedError("Cross-model activation patching is not implemented yet.")

