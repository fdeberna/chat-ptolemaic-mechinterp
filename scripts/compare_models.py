"""Compare layer-wise probe metrics between base and LoRA conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--lora-results", required=True)
    parser.add_argument("--output-csv", default="results/comparisons/base_vs_lora.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = summarize(pd.read_csv(args.base_results), "base")
    lora = summarize(pd.read_csv(args.lora_results), "lora")
    merged = base.merge(lora, on="layer_index", how="inner")
    for metric in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        merged[f"delta_{metric}_lora_minus_base"] = (
            merged[f"{metric}_lora"] - merged[f"{metric}_base"]
        )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Saved comparison to {output_csv}")


def summarize(metrics: pd.DataFrame, suffix: str) -> pd.DataFrame:
    grouped = metrics.groupby("layer_index", as_index=False)[
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    ].mean()
    renamed_columns = {
        column: f"{column}_{suffix}" for column in grouped.columns if column != "layer_index"
    }
    return grouped.rename(columns=renamed_columns)


if __name__ == "__main__":
    main()
