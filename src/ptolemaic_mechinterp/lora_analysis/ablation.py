"""Selective LoRA adapter ablation interfaces."""

from __future__ import annotations

import torch


def disable_lora_modules_by_name(model: torch.nn.Module, module_name_patterns: list[str]) -> None:
    """Disable selected LoRA modules by name pattern.

    PEFT adapter internals vary across versions and target modules, so the concrete ablation
    policy is deferred until the first adapter-specific experiment.
    """

    raise NotImplementedError(
        "Selective LoRA module ablation needs adapter-specific PEFT module inspection."
    )

