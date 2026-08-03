"""LoRA weight analysis interfaces."""

from __future__ import annotations

import numpy as np
import torch


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare two probe or LoRA-derived directions."""

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError("Cannot compute cosine similarity with a zero vector.")
    return float(np.dot(a, b) / denominator)


def project_lora_output_onto_probe_direction(
    lora_output: torch.Tensor,
    probe_direction: torch.Tensor,
) -> torch.Tensor:
    """Project LoRA module outputs onto a probe direction."""

    if lora_output.shape[-1] != probe_direction.shape[-1]:
        raise ValueError("LoRA output and probe direction dimensions do not match.")
    unit_direction = probe_direction / probe_direction.norm()
    return lora_output @ unit_direction

