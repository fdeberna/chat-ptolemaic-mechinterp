"""Verify a saved activation store through the repository storage API."""

from __future__ import annotations

import argparse
import sys

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True, help="Saved activation directory.")
    parser.add_argument("--expected-condition", choices=["base", "lora"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    activation_dir = resolve_repo_path(args.activations)
    store = load_activation_store(activation_dir)

    errors: list[str] = []
    metadata = store.metadata
    if args.expected_condition:
        conditions = set(metadata["model_condition"].astype(str))
        if conditions != {args.expected_condition}:
            errors.append(
                f"Expected condition {args.expected_condition}, found {sorted(conditions)}."
            )
    if store.manifest.get("layer_zero") != "embedding_output":
        errors.append("Manifest does not identify embedding output as layer 0.")

    layer_zero = metadata[metadata["layer_index"] == 0]
    prompt_ids = layer_zero["prompt_id"].astype(str).tolist()
    if len(prompt_ids) != len(set(prompt_ids)):
        errors.append("Layer-0 prompt IDs are not unique.")

    print(f"activation_dir: {activation_dir}")
    print(f"activation_shape: {tuple(store.activations.shape)}")
    print(f"metadata_rows: {len(metadata)}")
    print(f"prompt_count: {len(prompt_ids)}")
    print(f"layer_indices: {store.manifest.get('layer_indices')}")
    print(f"layer_zero: {store.manifest.get('layer_zero')}")
    print("metadata_preview:")
    print(metadata.head().to_string(index=False))

    if errors:
        print("errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("verification: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
