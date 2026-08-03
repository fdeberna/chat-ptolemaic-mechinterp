"""JSONL prompt dataset IO."""

from __future__ import annotations

import json
from pathlib import Path

from ptolemaic_mechinterp.data.schemas import PromptRecord


def read_prompt_jsonl(path: str | Path) -> list[PromptRecord]:
    """Read and validate a JSONL prompt dataset."""

    records: list[PromptRecord] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Expected JSON object at {jsonl_path}:{line_number}.")
            try:
                records.append(PromptRecord.from_mapping(raw))
            except ValueError as exc:
                message = f"Invalid prompt record at {jsonl_path}:{line_number}: {exc}"
                raise ValueError(message) from exc
    if not records:
        raise ValueError(f"No prompt records found in {jsonl_path}.")
    return records
