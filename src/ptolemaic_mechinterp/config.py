"""Typed YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Model and tokenizer loading options."""

    model_name_or_path: str
    adapter_path: str | None = None
    device: str | None = None
    dtype: str = "auto"
    load_in_4bit: bool = False
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionConfig:
    """Activation extraction options."""

    model: ModelConfig
    prompt_dataset_path: Path
    activation_output_dir: Path
    batch_size: int = 1
    max_length: int | None = None
    random_seed: int = 0


@dataclass(frozen=True)
class ProbeConfig:
    """Layer-wise linear probe training options."""

    target: str
    grouping_column: str
    n_folds: int = 4
    random_seed: int = 0
    output_csv: Path = Path("results/probes/metrics.csv")
    save_coefficients: bool = False
    coefficient_output_dir: Path | None = None
    max_iter: int = 1000
    class_weight: str | None = "balanced"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(data).__name__}.")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    """Load model configuration from YAML."""

    data = load_yaml(path)
    return parse_model_config(data)


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    """Load extraction configuration from YAML."""

    data = load_yaml(path)
    model_data = data.get("model", {})
    if not isinstance(model_data, dict):
        raise ValueError("'model' must be a mapping.")
    return ExtractionConfig(
        model=parse_model_config(model_data),
        prompt_dataset_path=resolve_repo_path(required(data, "prompt_dataset_path")),
        activation_output_dir=resolve_repo_path(required(data, "activation_output_dir")),
        batch_size=int(data.get("batch_size", 1)),
        max_length=optional_int(data.get("max_length")),
        random_seed=int(data.get("random_seed", 0)),
    )


def load_probe_config(path: str | Path) -> ProbeConfig:
    """Load probe configuration from YAML."""

    data = load_yaml(path)
    coefficient_output_dir = data.get("coefficient_output_dir")
    return ProbeConfig(
        target=str(required(data, "target")),
        grouping_column=str(data.get("grouping_column", "template_family")),
        n_folds=int(data.get("n_folds", 4)),
        random_seed=int(data.get("random_seed", 0)),
        output_csv=resolve_repo_path(data.get("output_csv", "results/probes/metrics.csv")),
        save_coefficients=bool(data.get("save_coefficients", False)),
        coefficient_output_dir=(
            resolve_repo_path(coefficient_output_dir) if coefficient_output_dir else None
        ),
        max_iter=int(data.get("max_iter", 1000)),
        class_weight=data.get("class_weight", "balanced"),
    )


def parse_model_config(data: dict[str, Any]) -> ModelConfig:
    """Parse a model configuration mapping."""

    return ModelConfig(
        model_name_or_path=str(required(data, "model_name_or_path")),
        adapter_path=optional_model_artifact_path(data.get("adapter_path")),
        device=optional_str(data.get("device")),
        dtype=str(data.get("dtype", "auto")),
        load_in_4bit=bool(data.get("load_in_4bit", False)),
        tokenizer_kwargs=dict(data.get("tokenizer_kwargs") or {}),
    )


def repository_root() -> Path:
    """Return the repository root for resolving project-local config paths."""

    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return current.parents[2]


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repository_root() / path).resolve()


def required(data: dict[str, Any], key: str) -> Any:
    """Return a required configuration value."""

    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required configuration key: {key}")
    return value


def optional_str(value: Any) -> str | None:
    """Convert null-like values to None, otherwise string."""

    if value is None or value == "":
        return None
    return str(value)


def optional_model_artifact_path(value: Any) -> str | None:
    """Resolve local model artifact paths while preserving Hugging Face identifiers."""

    raw = optional_str(value)
    if raw is None:
        return None
    normalized = raw.replace("\\", "/")
    if normalized.startswith((".", "~", "/")) or "\\" in raw:
        return str(resolve_repo_path(raw))
    first_component = normalized.split("/", maxsplit=1)[0]
    if first_component in {"adapters", "checkpoints", "models", "outputs", "results"}:
        return str(resolve_repo_path(raw))
    return raw


def optional_int(value: Any) -> int | None:
    """Convert null-like values to None, otherwise int."""

    if value is None or value == "":
        return None
    return int(value)
