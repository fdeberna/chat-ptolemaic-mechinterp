"""Hook utilities reserved for experiments that require lower-level intervention."""

from __future__ import annotations

from collections.abc import Callable

import torch


def register_residual_hook(
    model: torch.nn.Module,
    module_name: str,
    hook_fn: Callable[
        [torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor | None
    ],
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook by module name.

    The initial vertical slice uses returned hidden states instead. This helper exists for later
    activation patching or steering experiments that need module-level hooks.
    """

    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module not found: {module_name}")
    return modules[module_name].register_forward_hook(hook_fn)
