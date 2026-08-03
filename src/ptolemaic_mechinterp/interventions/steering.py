"""Activation steering interfaces."""

from __future__ import annotations

import torch


def apply_activation_steering(
    hidden_state: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Return a hidden state shifted along a steering direction."""

    if hidden_state.shape[-1] != direction.shape[-1]:
        raise ValueError("hidden_state and direction must have the same final dimension.")
    return hidden_state + coefficient * direction


def run_steered_generation(*args: object, **kwargs: object) -> None:
    """Run generation with steering hooks.

    This requires model-specific hook placement and is intentionally left for the steering
    experiment phase.
    """

    raise NotImplementedError("Steered generation is not implemented in the minimal slice.")

