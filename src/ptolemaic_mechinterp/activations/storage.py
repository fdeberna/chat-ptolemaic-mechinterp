"""Inspectable activation storage using NumPy, CSV, and JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ptolemaic_mechinterp.activations.extraction import ExtractedActivations

ACTIVATION_FILE = "activations.npz"
METADATA_FILE = "metadata.csv"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ActivationStore:
    """Activations loaded from disk."""

    activations: np.ndarray
    metadata: pd.DataFrame
    manifest: dict[str, Any]


def save_activation_store(
    output_dir: str | Path,
    extracted: ExtractedActivations,
    *,
    extraction_config: dict[str, Any] | None = None,
) -> None:
    """Save activations, row metadata, and a manifest to a directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / ACTIVATION_FILE, activations=extracted.activations)
    extracted.metadata.to_csv(output_path / METADATA_FILE, index=False)
    manifest = {
        "activation_file": ACTIVATION_FILE,
        "metadata_file": METADATA_FILE,
        "num_rows": int(extracted.activations.shape[0]),
        "hidden_size": extracted.hidden_size,
        "layer_indices": extracted.layer_indices,
        "layer_zero": extracted.layer_zero,
        "extraction_config": extraction_config or {},
    }
    with (output_path / MANIFEST_FILE).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def load_activation_store(input_dir: str | Path) -> ActivationStore:
    """Load a saved activation store."""

    input_path = Path(input_dir)
    with (input_path / MANIFEST_FILE).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    activation_path = input_path / manifest.get("activation_file", ACTIVATION_FILE)
    metadata_path = input_path / manifest.get("metadata_file", METADATA_FILE)
    with np.load(activation_path) as arrays:
        activations = arrays["activations"]
    metadata = pd.read_csv(metadata_path)
    if activations.shape[0] != len(metadata):
        raise ValueError(
            f"Activation row count ({activations.shape[0]}) does not match metadata rows "
            f"({len(metadata)})."
        )
    return ActivationStore(activations=activations, metadata=metadata, manifest=manifest)

