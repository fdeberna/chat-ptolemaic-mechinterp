"""Simple stance label constants for behavioral evaluations."""

from __future__ import annotations

STANCE_LABELS = ("geocentric", "heliocentric")


def validate_stance_label(label: str) -> str:
    """Validate a cosmological stance label."""

    if label not in STANCE_LABELS:
        raise ValueError(f"Invalid stance label '{label}'. Expected one of {STANCE_LABELS}.")
    return label

