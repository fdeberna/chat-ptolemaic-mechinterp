"""Extract final-token hidden-state activations for a prompt dataset."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ptolemaic_mechinterp.activations.extraction import extract_final_token_hidden_states
from ptolemaic_mechinterp.activations.storage import (
    MANIFEST_FILE,
    load_activation_store,
    save_activation_store,
)
from ptolemaic_mechinterp.config import load_extraction_config, repository_root, resolve_repo_path
from ptolemaic_mechinterp.data.prompts import read_prompt_jsonl
from ptolemaic_mechinterp.data.schemas import PromptRecord
from ptolemaic_mechinterp.data.validation import validate_prompt_records
from ptolemaic_mechinterp.models.loader import choose_tokenizer_source, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to extraction YAML config.")
    parser.add_argument("--condition", choices=["base", "lora"], required=True)
    parser.add_argument("--model-name-or-path", help="Override model path/name from config.")
    parser.add_argument("--adapter-path", help="Override adapter path from config.")
    parser.add_argument("--output-dir", help="Override activation output directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without model loading.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Select complete template families up to this count.",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        help="Select the first N complete template families.",
    )
    parser.add_argument("--max-batches", type=int, help="Stop extraction after N batches.")
    parser.add_argument(
        "--load-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load tokenizer during dry-run to report tokenized prompt lengths.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = load_extraction_config(args.config)
    model_config = config.model
    model_name_or_path = args.model_name_or_path or model_config.model_name_or_path
    adapter_path = (
        str(resolve_repo_path(args.adapter_path))
        if args.adapter_path is not None
        else model_config.adapter_path
    )
    if args.condition == "base":
        adapter_path = None
    elif args.condition == "lora" and not adapter_path:
        raise ValueError("--condition lora requires adapter_path in config or --adapter-path.")

    prompts = read_prompt_jsonl(config.prompt_dataset_path)
    validation = validate_prompt_records(prompts, path=config.prompt_dataset_path)
    if not validation.is_valid:
        raise ValueError("Prompt validation failed:\n" + "\n".join(validation.errors))
    selected_prompts = select_prompt_subset(
        prompts,
        max_prompts=args.max_prompts,
        max_families=args.max_families,
    )
    output_dir = resolve_output_dir(args.output_dir, config.activation_output_dir, args.condition)
    ensure_no_cross_condition_overwrite(output_dir, args.condition)
    print_selection_summary(args.condition, selected_prompts, output_dir)

    if args.dry_run:
        run_dry_run(
            prompts=selected_prompts,
            condition=args.condition,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            tokenizer_kwargs=model_config.tokenizer_kwargs,
            output_dir=output_dir,
            load_tokenizer=args.load_tokenizer,
            max_length=config.max_length,
        )
        return

    loaded = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        device=model_config.device,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
        tokenizer_kwargs=model_config.tokenizer_kwargs,
    )
    if loaded.condition != args.condition:
        raise RuntimeError(
            f"Requested condition {args.condition}, but loader returned {loaded.condition}."
        )
    if args.condition == "lora" and not loaded.adapter_active:
        raise RuntimeError("Requested LoRA extraction, but no active adapter was loaded.")
    if loaded.model.training:
        raise RuntimeError("Model is not in evaluation mode.")
    if any(parameter.requires_grad for parameter in loaded.model.parameters()):
        raise RuntimeError("Model parameters still require gradients.")

    extracted = extract_final_token_hidden_states(
        loaded.model,
        loaded.tokenizer,
        selected_prompts,
        condition=args.condition,
        batch_size=config.batch_size,
        device=model_config.device,
        max_length=config.max_length,
        max_batches=args.max_batches,
        model_name_or_path=sanitize_identifier(model_name_or_path),
        adapter_path=sanitize_identifier(adapter_path),
    )
    validate_extracted_alignment(extracted.metadata, selected_prompts)
    manifest_config = serializable_config(asdict(config))
    manifest_config["model"]["model_name_or_path"] = sanitize_identifier(model_name_or_path)
    manifest_config["model"]["adapter_path"] = sanitize_identifier(adapter_path)
    save_activation_store(
        output_dir,
        extracted,
        extraction_config=serializable_config(
            manifest_config
            | {
                "condition": args.condition,
                "selected_prompt_ids": selected_prompt_ids_from_metadata(extracted.metadata),
                "model_name_or_path": sanitize_identifier(model_name_or_path),
                "adapter_path": sanitize_identifier(adapter_path),
                "tokenizer_source": sanitize_identifier(loaded.tokenizer_source),
                "adapter_active": loaded.adapter_active,
                "layer_zero": extracted.layer_zero,
                "max_prompts": args.max_prompts,
                "max_families": args.max_families,
                "max_batches": args.max_batches,
            }
        ),
    )
    round_trip = verify_saved_round_trip(output_dir, extracted)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_extraction_summary(output_dir, extracted, round_trip)
    logging.info("Saved %s activation rows to %s", extracted.activations.shape[0], output_dir)


def select_prompt_subset(
    prompts: list[PromptRecord],
    *,
    max_prompts: int | None,
    max_families: int | None,
) -> list[PromptRecord]:
    """Select prompts deterministically without splitting template families."""

    if max_prompts is not None and max_prompts < 1:
        raise ValueError("--max-prompts must be >= 1.")
    if max_families is not None and max_families < 1:
        raise ValueError("--max-families must be >= 1.")
    if max_prompts is None and max_families is None:
        return list(prompts)

    families: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.template_family not in seen:
            seen.add(prompt.template_family)
            families.append(prompt.template_family)

    selected_families = families[:max_families] if max_families is not None else families
    selected: list[PromptRecord] = [
        prompt for prompt in prompts if prompt.template_family in set(selected_families)
    ]
    if max_prompts is None:
        return selected

    family_limited: list[PromptRecord] = []
    for family in selected_families:
        family_rows = [prompt for prompt in prompts if prompt.template_family == family]
        if len(family_limited) + len(family_rows) > max_prompts:
            break
        family_limited.extend(family_rows)

    if len(family_limited) != max_prompts:
        raise ValueError(
            "--max-prompts must land on a complete template-family boundary; "
            f"selected {len(family_limited)} prompts before exceeding {max_prompts}. "
            "Use --max-families for explicit family-aware selection."
        )
    return family_limited


def resolve_output_dir(
    output_dir: str | None,
    activation_output_dir: Path,
    condition: str,
) -> Path:
    if output_dir:
        return resolve_repo_path(output_dir)
    return activation_output_dir / condition


def ensure_no_cross_condition_overwrite(output_dir: Path, condition: str) -> None:
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    existing_condition = manifest.get("extraction_config", {}).get("condition")
    if existing_condition and existing_condition != condition:
        raise RuntimeError(
            f"Refusing to write {condition} activations over existing "
            f"{existing_condition} output at {output_dir}."
        )


def print_selection_summary(
    condition: str,
    prompts: list[PromptRecord],
    output_dir: Path,
) -> None:
    families = []
    seen = set()
    for prompt in prompts:
        if prompt.template_family not in seen:
            seen.add(prompt.template_family)
            families.append(prompt.template_family)
    print(f"condition: {condition}")
    print(f"selected_prompts: {len(prompts)}")
    print(f"selected_template_families: {families}")
    print(f"selected_prompt_ids: {[prompt.prompt_id for prompt in prompts]}")
    print(f"expected_output_dir: {output_dir}")


def run_dry_run(
    *,
    prompts: list[PromptRecord],
    condition: str,
    model_name_or_path: str,
    adapter_path: str | None,
    tokenizer_kwargs: dict[str, Any],
    output_dir: Path,
    load_tokenizer: bool,
    max_length: int | None,
) -> None:
    print("dry_run: true")
    print(f"model_condition: {condition}")
    print(f"prompt_count: {len(prompts)}")
    print(f"model_name_or_path: {sanitize_identifier(model_name_or_path)}")
    print(f"adapter_path: {sanitize_identifier(adapter_path)}")
    print(f"expected_output_dir: {output_dir}")
    if not load_tokenizer:
        print("tokenizer: skipped")
        return

    from transformers import AutoTokenizer

    tokenizer_source = choose_tokenizer_source(model_name_or_path, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    print(f"tokenizer_source: {sanitize_identifier(tokenizer_source)}")
    for prompt in prompts[:2]:
        encoded = tokenizer(
            prompt.text,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        print(f"tokenized_length[{prompt.prompt_id}]: {int(encoded['input_ids'].shape[-1])}")


def validate_extracted_alignment(metadata: pd.DataFrame, prompts: list[PromptRecord]) -> None:
    layer_zero = metadata[metadata["layer_index"] == 0].reset_index(drop=True)
    extracted_prompt_ids = layer_zero["prompt_id"].astype(str).tolist()
    expected_prompt_ids = [prompt.prompt_id for prompt in prompts[: len(extracted_prompt_ids)]]
    if extracted_prompt_ids != expected_prompt_ids:
        raise RuntimeError("Prompt IDs are not aligned with activation metadata rows.")
    for column, values in {
        "stance": [prompt.stance for prompt in prompts[: len(extracted_prompt_ids)]],
        "style": [prompt.style for prompt in prompts[: len(extracted_prompt_ids)]],
        "template_family": [
            prompt.template_family for prompt in prompts[: len(extracted_prompt_ids)]
        ],
    }.items():
        if layer_zero[column].tolist() != values:
            raise RuntimeError(f"{column} labels are not aligned with activation metadata rows.")


def verify_saved_round_trip(output_dir: Path, extracted: Any) -> Any:
    store = load_activation_store(output_dir)
    if store.activations.shape != extracted.activations.shape:
        raise RuntimeError("Reloaded activation shape does not match saved activation shape.")
    for column in ["prompt_id", "layer_index", "stance", "style", "template_family"]:
        loaded_values = store.metadata[column].fillna("").tolist()
        saved_values = extracted.metadata[column].fillna("").tolist()
        if loaded_values != saved_values:
            raise RuntimeError(f"Reloaded metadata column does not match saved values: {column}.")
    if store.manifest.get("layer_zero") != "embedding_output":
        raise RuntimeError("Manifest does not document embedding output as layer 0.")
    return store


def selected_prompt_ids_from_metadata(metadata: pd.DataFrame) -> list[str]:
    return metadata[metadata["layer_index"] == 0]["prompt_id"].astype(str).tolist()


def print_extraction_summary(output_dir: Path, extracted: Any, store: Any) -> None:
    print("extraction_summary:")
    print(f"  model_condition: {extracted.metadata['model_condition'].iloc[0]}")
    print(f"  prompts: {len(selected_prompt_ids_from_metadata(extracted.metadata))}")
    print(f"  layers_including_embedding: {len(extracted.layer_indices)}")
    print(f"  transformer_layers_excluding_embedding: {len(extracted.layer_indices) - 1}")
    print(f"  hidden_dimension: {extracted.hidden_size}")
    print(f"  activation_shape: {tuple(extracted.activations.shape)}")
    print(f"  dtype: {extracted.activations.dtype}")
    print(f"  activation_file: {output_dir / 'activations.npz'}")
    print(f"  metadata_file: {output_dir / 'metadata.csv'}")
    print(f"  manifest_file: {output_dir / 'manifest.json'}")
    print("metadata_preview:")
    print(store.metadata.head().to_string(index=False))
    print(f"layer_zero: {store.manifest.get('layer_zero')}")


def serializable_config(value: Any) -> Any:
    """Convert Path-containing dataclass dictionaries into JSON-compatible values."""

    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, str):
        return display_path(value)
    if isinstance(value, dict):
        return {key: serializable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable_config(item) for item in value]
    return value


def sanitize_identifier(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    for secret_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        secret = __import__("os").environ.get(secret_name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"${secret_name}")
    return display_path(text)


def display_path(value: str | Path) -> str:
    text = str(value)
    path = Path(text)
    if not path.is_absolute():
        return text
    root = repository_root()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        relative_to_models = path.relative_to(root.parent)
        return f"../{relative_to_models.as_posix()}"
    except ValueError:
        return text


if __name__ == "__main__":
    main()
