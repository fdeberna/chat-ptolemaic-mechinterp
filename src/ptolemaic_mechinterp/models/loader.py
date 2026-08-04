"""Hugging Face and PEFT model loading."""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A loaded causal LM, tokenizer, and condition label."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    condition: str
    model_name_or_path: str
    tokenizer_source: str
    adapter_path: str | None = None
    adapter_active: bool = False


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
    LOGGER.info("Loading tokenizer from %s", safe_log_identifier(tokenizer_source))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"output_hidden_states": True}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    LOGGER.info("Loading base model from %s", safe_log_identifier(model_name_or_path))
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
        LOGGER.info("Attaching PEFT adapter from %s", safe_log_identifier(adapter_path))
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to attach LoRA adapter from {adapter_path}: {exc}") from exc
        if not has_peft_adapter(model):
            raise RuntimeError("LoRA adapter load returned a model with no PEFT adapter config.")
        if not adapter_is_active(model):
            raise RuntimeError("LoRA adapter is attached but does not appear to be active.")
        condition = "lora"

    if device:
        model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        condition=condition,
        model_name_or_path=model_name_or_path,
        tokenizer_source=tokenizer_source,
        adapter_path=adapter_path,
        adapter_active=adapter_is_active(model),
    )


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


def has_peft_adapter(model: torch.nn.Module) -> bool:
    """Return whether a model exposes at least one PEFT adapter config."""

    peft_config = getattr(model, "peft_config", None)
    return isinstance(peft_config, dict) and bool(peft_config)


def adapter_is_active(model: torch.nn.Module) -> bool:
    """Return whether a PEFT model reports an active adapter."""

    if not has_peft_adapter(model):
        return False

    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    if isinstance(active_adapters, (list, tuple, set)):
        return bool(active_adapters)
    if isinstance(active_adapters, str):
        return bool(active_adapters)

    active_adapter = getattr(model, "active_adapter", None)
    if callable(active_adapter):
        active_adapter = active_adapter()
    if isinstance(active_adapter, (list, tuple, set)):
        return bool(active_adapter)
    if isinstance(active_adapter, str):
        return bool(active_adapter)

    return True


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


def safe_log_identifier(value: str | None) -> str | None:
    """Return a model/path identifier with common local secret material redacted."""

    if value is None:
        return None
    text = str(value)
    if is_local_path_like(text):
        try:
            text = str(Path(text).expanduser().resolve()).replace(str(Path.home()), "~")
        except OSError:
            pass
    for secret_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        secret = os.environ.get(secret_name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"${secret_name}")
    return text


def is_local_path_like(value: str) -> bool:
    """Return whether a string looks like a filesystem path rather than an HF model ID."""

    normalized = value.replace("\\", "/")
    return (
        Path(value).is_absolute()
        or normalized.startswith((".", "~", "/"))
        or "\\" in value
        or Path(value).exists()
    )


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
