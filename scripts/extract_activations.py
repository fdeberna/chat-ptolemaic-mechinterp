"""Extract final-token hidden-state activations for a prompt dataset."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ptolemaic_mechinterp.activations.extraction import extract_final_token_hidden_states
from ptolemaic_mechinterp.activations.storage import save_activation_store
from ptolemaic_mechinterp.config import load_extraction_config
from ptolemaic_mechinterp.data.prompts import read_prompt_jsonl
from ptolemaic_mechinterp.models.loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to extraction YAML config.")
    parser.add_argument("--condition", choices=["base", "lora"], required=True)
    parser.add_argument("--model-name-or-path", help="Override model path/name from config.")
    parser.add_argument("--adapter-path", help="Override adapter path from config.")
    parser.add_argument("--output-dir", help="Override activation output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = load_extraction_config(args.config)
    model_config = config.model
    model_name_or_path = args.model_name_or_path or model_config.model_name_or_path
    adapter_path = args.adapter_path if args.adapter_path is not None else model_config.adapter_path
    if args.condition == "base":
        adapter_path = None
    elif args.condition == "lora" and not adapter_path:
        raise ValueError("--condition lora requires adapter_path in config or --adapter-path.")

    loaded = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        device=model_config.device,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
        tokenizer_kwargs=model_config.tokenizer_kwargs,
    )
    prompts = read_prompt_jsonl(config.prompt_dataset_path)
    extracted = extract_final_token_hidden_states(
        loaded.model,
        loaded.tokenizer,
        prompts,
        condition=args.condition,
        batch_size=config.batch_size,
        device=model_config.device,
        max_length=config.max_length,
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir else config.activation_output_dir / args.condition
    )
    save_activation_store(
        output_dir,
        extracted,
        extraction_config=serializable_config(asdict(config) | {"condition": args.condition}),
    )
    logging.info("Saved %s activation rows to %s", extracted.activations.shape[0], output_dir)


def serializable_config(value: Any) -> Any:
    """Convert Path-containing dataclass dictionaries into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable_config(item) for item in value]
    return value


if __name__ == "__main__":
    main()
