"""Train layer-wise logistic-regression probes from saved activations."""

from __future__ import annotations

import argparse
import logging

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import load_probe_config, resolve_repo_path
from ptolemaic_mechinterp.probes.linear import train_layerwise_logistic_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to probe YAML config.")
    parser.add_argument("--activations", required=True, help="Saved activation directory.")
    parser.add_argument("--target", help="Override target label column.")
    parser.add_argument("--grouping-column", help="Override grouping column.")
    parser.add_argument("--n-folds", type=int, help="Override number of grouped CV folds.")
    parser.add_argument("--output-csv", help="Override probe metrics CSV path.")
    parser.add_argument("--save-coefficients", action="store_true", help="Save probe coefficients.")
    parser.add_argument("--coefficient-output-dir", help="Override coefficient output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = load_probe_config(args.config)
    store = load_activation_store(args.activations)
    target = args.target or config.target
    grouping_column = args.grouping_column or config.grouping_column
    n_folds = args.n_folds or config.n_folds
    save_coefficients = args.save_coefficients or config.save_coefficients
    coefficient_dir = (
        resolve_repo_path(args.coefficient_output_dir)
        if args.coefficient_output_dir
        else config.coefficient_output_dir
    )
    coefficient_dir = coefficient_dir if save_coefficients else None
    metrics = train_layerwise_logistic_probes(
        store,
        target=target,
        grouping_column=grouping_column,
        n_folds=n_folds,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        coefficient_output_dir=coefficient_dir,
    )
    output_csv = resolve_repo_path(args.output_csv) if args.output_csv else config.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    print_probe_summary(metrics, target)
    logging.info("Saved probe metrics to %s", output_csv)


def print_probe_summary(metrics: object, target: str) -> None:
    print(f"target: {target}")
    print(f"rows: {len(metrics)}")
    print(f"layers: {sorted(metrics['layer_index'].unique().tolist())}")
    print("mean_metrics_by_layer:")
    print(
        metrics.groupby("layer_index")[["accuracy", "balanced_accuracy", "f1", "roc_auc"]]
        .mean()
        .head()
        .to_string()
    )
    print("fold_diagnostics_preview:")
    diagnostic_columns = [
        "layer_index",
        "fold",
        "group_intersection_size",
        "train_class_distribution",
        "test_class_distribution",
        "train_topic_distribution",
        "test_topic_distribution",
        "exact_duplicate_texts",
        "normalized_duplicate_texts",
    ]
    available = [column for column in diagnostic_columns if column in metrics.columns]
    print(metrics[available].head().to_string(index=False))


if __name__ == "__main__":
    main()
