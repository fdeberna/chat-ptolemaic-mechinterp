#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def choose_tokenizer_source(model_name: str, adapter_dir: str | None) -> str:
    if not adapter_dir:
        return model_name

    adapter_path = Path(adapter_dir)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any((adapter_path / filename).exists() for filename in tokenizer_files):
        return adapter_dir
    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with Qwen2.5-7B, optionally with a LoRa adapter.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling. Defaults to true; use --no-do-sample for greedy generation.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer_source = choose_tokenizer_source(args.model_name, args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        model = base_model
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
