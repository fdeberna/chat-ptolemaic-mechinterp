"""Validate a controlled prompt JSONL dataset."""

from __future__ import annotations

import argparse
import sys

from ptolemaic_mechinterp.config import repository_root, resolve_repo_path
from ptolemaic_mechinterp.data.validation import validate_prompt_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/prompts/pilot_stance_style.jsonl",
        help="Prompt JSONL path, resolved relative to the repository root when relative.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.path)
    result = validate_prompt_dataset(dataset_path)

    print(f"dataset: {dataset_path}")
    print(f"repository_root: {repository_root()}")
    print(f"total_records: {result.total_record_count}")
    print(f"unique_prompt_ids: {result.unique_prompt_id_count}")
    print(f"template_families: {result.template_family_count}")
    print_counter("counts_by_stance", result.counts_by_stance)
    print_counter("counts_by_style", result.counts_by_style)
    print_counter("counts_by_stance_style", result.counts_by_stance_style)
    print_counter("counts_by_topic", result.counts_by_topic)

    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    print("validation: passed")
    return 0


def print_counter(title: str, counts: object) -> None:
    print(f"{title}:")
    for key, value in sorted(counts.items(), key=lambda item: str(item[0])):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    sys.exit(main())
