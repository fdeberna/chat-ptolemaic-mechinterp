"""Recompute audited base-vs-LoRA full-data probe direction similarities."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ptolemaic_mechinterp.config import resolve_repo_path
from ptolemaic_mechinterp.probes.comparison import build_audited_direction_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/probes/pilot_full")
    parser.add_argument("--comparison-root", default="results/comparisons/pilot_full")
    parser.add_argument(
        "--output-name",
        default="probe_direction_similarity_audited.csv",
        help="Audited CSV filename under comparison-root.",
    )
    parser.add_argument(
        "--old-comparison-name",
        default="probe_direction_similarity.csv",
        help="Existing comparison CSV to use for old-vs-audited deltas if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = resolve_repo_path(args.results_root)
    comparison_root = resolve_repo_path(args.comparison_root)
    comparison_root.mkdir(parents=True, exist_ok=True)

    audited = build_audited_direction_similarity(results_root=results_root)
    old_path = comparison_root / args.old_comparison_name
    if old_path.exists():
        audited = add_old_comparison_deltas(audited, pd.read_csv(old_path))

    output_path = comparison_root / args.output_name
    audited.to_csv(output_path, index=False)
    print(json.dumps({"output": str(output_path), "rows": int(len(audited))}, indent=2))


def add_old_comparison_deltas(audited: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    result = audited.copy()
    old_columns = {"stance": "stance_cosine", "style": "style_cosine"}
    result["old_cosine"] = pd.NA
    result["old_minus_raw_activation_space_cosine"] = pd.NA
    for target, column in old_columns.items():
        if column not in old.columns:
            continue
        target_mask = result["target"] == target
        layer_to_old = dict(zip(old["layer"].astype(int), old[column], strict=True))
        result.loc[target_mask, "old_cosine"] = result.loc[target_mask, "layer"].map(layer_to_old)
    result["old_minus_raw_activation_space_cosine"] = (
        result["old_cosine"].astype(float) - result["raw_activation_space_cosine"]
    )
    return result


if __name__ == "__main__":
    main()
