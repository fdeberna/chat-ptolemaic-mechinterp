"""Activation steering interfaces."""

from __future__ import annotations

import torch
from contextlib import contextmanager


def apply_activation_steering(
    hidden_state: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Return a hidden state shifted along a steering direction."""

    if hidden_state.shape[-1] != direction.shape[-1]:
        raise ValueError("hidden_state and direction must have the same final dimension.")
    return hidden_state + coefficient * direction


@contextmanager
def residual_steering_hook(module, direction, coefficient, prompt_token_index, observer=None):
    """Steer a block output: final prompt token once, then cached single tokens.

    Batch size one is required. Caller must start a fresh cached forward sequence
    inside each context. Tensor and tuple module outputs are supported.
    An optional observer receives the before/after tensors for targeted checks.
    """
    state = {'calls': 0, 'prefill_calls': 0, 'decode_calls': 0,
             'modified_tokens': 0, 'ratios': [], 'dtypes': []}
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError('Steering requires [1, sequence, hidden] output')
        if direction.numel() != tensor.shape[-1]:
            raise ValueError('Direction dimension differs from residual dimension')
        prefill = state['calls'] == 0
        index = prompt_token_index if prefill else 0
        if not prefill and tensor.shape[1] != 1:
            raise ValueError('Cached decoding must process exactly one new token')
        if not 0 <= index < tensor.shape[1]:
            raise ValueError('Prompt token index outside prefill tensor')
        # Form the requested perturbation in float32, then round the result to
        # the actual residual dtype. Never mutate the original module output.
        delta = direction.to(device=tensor.device, dtype=torch.float32) * coefficient
        changed = tensor.clone()
        changed[0,index,:] = (tensor[0,index,:].float() + delta).to(tensor.dtype)
        actual = changed[0,index,:].float() - tensor[0,index,:].float()
        state['calls'] += 1
        state['prefill_calls' if prefill else 'decode_calls'] += 1
        state['modified_tokens'] += 1
        state['ratios'].append(float(actual.norm()/tensor[0,index,:].float().norm()))
        state['dtypes'].append(str(tensor.dtype))
        if observer is not None:
            observer(tensor, changed, index, delta)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def run_steered_generation(module, direction, coefficient, prompt_token_index, generate, observer=None):
    """Run a caller-supplied cached generation with an exception-safe hook."""
    with residual_steering_hook(module, direction, coefficient, prompt_token_index, observer) as state:
        result = generate()
    return result, state

