"""Compare saved base and LoRA activation stores for structural sanity."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-activations", required=True)
    parser.add_argument("--lora-activations", required=True)
    parser.add_argument(
        "--output-csv",
        default="results/comparisons/pilot_smoke_base_vs_lora.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_repo_path(args.base_activations)
    lora_dir = resolve_repo_path(args.lora_activations)
    output_csv = resolve_repo_path(args.output_csv)

    base = load_activation_store(base_dir)
    lora = load_activation_store(lora_dir)
    validate_alignment(base.metadata, lora.metadata)

    rows: list[dict[str, object]] = []
    shapes_match = base.activations.shape == lora.activations.shape
    for layer_index in sorted(base.metadata["layer_index"].unique()):
        base_mask = base.metadata["layer_index"].to_numpy() == layer_index
        lora_mask = lora.metadata["layer_index"].to_numpy() == layer_index
        base_layer = base.activations[base_mask]
        lora_layer = lora.activations[lora_mask]
        if base_layer.shape != lora_layer.shape:
            raise ValueError(
                f"Layer {layer_index} shapes differ: {base_layer.shape} vs {lora_layer.shape}."
            )

        difference = base_layer - lora_layer
        cosine = rowwise_cosine_similarity(base_layer, lora_layer)
        rows.append(
            {
                "layer_index": int(layer_index),
                "shapes_match": shapes_match,
                "prompt_ids_match": True,
                "base_mean_activation_norm": float(np.linalg.norm(base_layer, axis=1).mean()),
                "lora_mean_activation_norm": float(np.linalg.norm(lora_layer, axis=1).mean()),
                "mean_base_lora_difference_norm": float(
                    np.linalg.norm(difference, axis=1).mean()
                ),
                "mean_base_lora_cosine_similarity": float(cosine.mean()),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_csv, index=False)
    print(f"base_shape: {tuple(base.activations.shape)}")
    print(f"lora_shape: {tuple(lora.activations.shape)}")
    print(f"shapes_match: {shapes_match}")
    print("comparison_preview:")
    print(comparison.head().to_string(index=False))
    print(f"saved_comparison: {output_csv}")


def validate_alignment(base_metadata: pd.DataFrame, lora_metadata: pd.DataFrame) -> None:
    columns = ["prompt_id", "layer_index", "stance", "style", "template_family"]
    for column in columns:
        base_values = base_metadata[column].fillna("").astype(str).tolist()
        lora_values = lora_metadata[column].fillna("").astype(str).tolist()
        if base_values != lora_values:
            raise ValueError(f"Base and LoRA metadata are not aligned for column: {column}.")


def rowwise_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, a_min=1e-12, a_max=None)


if __name__ == "__main__":
    main()
