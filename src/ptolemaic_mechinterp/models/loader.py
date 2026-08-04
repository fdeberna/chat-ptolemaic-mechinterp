"""Hugging Face and PEFT model loading."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A loaded causal LM, tokenizer, and condition label."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    condition: str


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: str | None = None,
    device: str | None = None,
    dtype: str = "auto",
    load_in_4bit: bool = False,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> LoadedModel:
    """Load a base causal LM and optionally attach a PEFT LoRA adapter."""

    if load_in_4bit and importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "load_in_4bit=True requires bitsandbytes. Install the optional quantization "
            "dependencies or disable 4-bit loading."
        )

    torch_dtype = parse_dtype(dtype)
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    tokenizer_source = choose_tokenizer_source(model_name_or_path, adapter_path)
    LOGGER.info("Loading tokenizer from %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"output_hidden_states": True}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    LOGGER.info("Loading base model from %s", model_name_or_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load base model from {model_name_or_path}: {exc}") from exc

    condition = "base"
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("adapter_path was provided, but peft is not installed.") from exc
        LOGGER.info("Attaching PEFT adapter from %s", adapter_path)
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to attach LoRA adapter from {adapter_path}: {exc}") from exc
        condition = "lora"

    if device:
        model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedModel(model=model, tokenizer=tokenizer, condition=condition)


def choose_tokenizer_source(model_name_or_path: str, adapter_path: str | None) -> str:
    """Use adapter-local tokenizer files when the LoRA export includes them."""

    if not adapter_path:
        return model_name_or_path

    adapter_dir = Path(adapter_path)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any((adapter_dir / filename).exists() for filename in tokenizer_files):
        return str(adapter_dir)
    return model_name_or_path


def parse_dtype(dtype: str) -> torch.dtype | str:
    """Parse a user-facing dtype string."""

    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use auto, float16, bfloat16, or float32.")
    return mapping[normalized]


def enable_adapter(model: torch.nn.Module) -> None:
    """Enable PEFT adapter layers when the loaded model supports it."""

    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
        return
    raise TypeError("This model does not expose PEFT adapter enable controls.")


def disable_adapter(model: torch.nn.Module) -> None:
    """Disable PEFT adapter layers when the loaded model supports it."""

    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        return
    raise TypeError("This model does not expose PEFT adapter disable controls.")
