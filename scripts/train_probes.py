"""Train layer-wise logistic-regression probes from saved activations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ptolemaic_mechinterp.activations.storage import load_activation_store
from ptolemaic_mechinterp.config import load_probe_config
from ptolemaic_mechinterp.probes.linear import train_layerwise_logistic_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to probe YAML config.")
    parser.add_argument("--activations", required=True, help="Saved activation directory.")
    parser.add_argument("--output-csv", help="Override probe metrics CSV path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = load_probe_config(args.config)
    store = load_activation_store(args.activations)
    coefficient_dir = config.coefficient_output_dir if config.save_coefficients else None
    metrics = train_layerwise_logistic_probes(
        store,
        target=config.target,
        grouping_column=config.grouping_column,
        n_folds=config.n_folds,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        coefficient_output_dir=coefficient_dir,
    )
    output_csv = Path(args.output_csv) if args.output_csv else config.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    logging.info("Saved probe metrics to %s", output_csv)


if __name__ == "__main__":
    main()

