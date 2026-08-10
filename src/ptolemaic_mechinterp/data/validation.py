"""Dataset-level validation for controlled prompt sets."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ptolemaic_mechinterp.data.schemas import PromptRecord

REQUIRED_STANCE_STYLE_COMBINATIONS = {
    ("heliocentric", "modern"),
    ("heliocentric", "premodern"),
    ("geocentric", "modern"),
    ("geocentric", "premodern"),
}


@dataclass(frozen=True)
class PromptDatasetValidation:
    """Structured result for prompt dataset validation."""

    path: Path
    records: list[PromptRecord]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts_by_stance: Counter[str] = field(default_factory=Counter)
    counts_by_style: Counter[str] = field(default_factory=Counter)
    counts_by_stance_style: Counter[tuple[str, str]] = field(default_factory=Counter)
    counts_by_topic: Counter[str] = field(default_factory=Counter)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def total_record_count(self) -> int:
        return len(self.records)

    @property
    def unique_prompt_id_count(self) -> int:
        return len({record.prompt_id for record in self.records})

    @property
    def template_family_count(self) -> int:
        return len({record.template_family for record in self.records})


def validate_prompt_dataset(path: str | Path) -> PromptDatasetValidation:
    """Load a JSONL prompt dataset and validate schema plus 2x2 family invariants."""

    dataset_path = Path(path)
    records, errors = read_prompt_records_collecting_errors(dataset_path)
    result = validate_prompt_records(records, path=dataset_path)
    return PromptDatasetValidation(
        path=dataset_path,
        records=records,
        errors=errors + result.errors,
        warnings=result.warnings,
        counts_by_stance=result.counts_by_stance,
        counts_by_style=result.counts_by_style,
        counts_by_stance_style=result.counts_by_stance_style,
        counts_by_topic=result.counts_by_topic,
    )


def validate_prompt_records(
    records: list[PromptRecord],
    *,
    path: str | Path = "<records>",
) -> PromptDatasetValidation:
    """Validate already parsed prompt records for controlled-design invariants."""

    errors: list[str] = []
    warnings: list[str] = []
    path_value = Path(path) if not isinstance(path, Path) else path

    if not records:
        errors.append(f"No prompt records found in {path}.")

    prompt_id_counts = Counter(record.prompt_id for record in records)
    duplicate_prompt_ids = sorted(
        prompt_id for prompt_id, count in prompt_id_counts.items() if count > 1
    )
    if duplicate_prompt_ids:
        errors.append(f"Duplicate prompt_id values: {', '.join(duplicate_prompt_ids)}.")

    exact_text_counts = Counter(record.text for record in records)
    exact_duplicates = sorted(text for text, count in exact_text_counts.items() if count > 1)
    if exact_duplicates:
        warnings.append(f"Exact duplicate prompt text count: {len(exact_duplicates)}.")

    normalized_text_counts = Counter(normalize_text(record.text) for record in records)
    normalized_duplicates = sorted(
        text for text, count in normalized_text_counts.items() if count > 1
    )
    if normalized_duplicates:
        warnings.append(f"Normalized duplicate prompt text count: {len(normalized_duplicates)}.")

    family_records: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in records:
        family_records[record.template_family].append(record)
        if not record.text.strip():
            errors.append(f"{record.prompt_id}: text is blank.")
        if record.stance is None:
            errors.append(f"{record.prompt_id}: stance is missing.")
        if record.style is None:
            errors.append(f"{record.prompt_id}: style is missing.")
        topic = record.metadata.get("topic")
        proposition = record.metadata.get("proposition")
        if not isinstance(topic, str) or not topic.strip():
            errors.append(f"{record.prompt_id}: metadata.topic must be a non-empty string.")
        if not isinstance(proposition, str) or not proposition.strip():
            errors.append(
                f"{record.prompt_id}: metadata.proposition must be a non-empty string."
            )

    for family, family_rows in sorted(family_records.items()):
        if len(family_rows) != 4:
            errors.append(f"{family}: expected exactly 4 records, found {len(family_rows)}.")

        combination_counts = Counter((record.stance, record.style) for record in family_rows)
        for combination in sorted(REQUIRED_STANCE_STYLE_COMBINATIONS):
            count = combination_counts.get(combination, 0)
            if count != 1:
                stance, style = combination
                errors.append(
                    f"{family}: expected exactly one {stance} + {style} record, found {count}."
                )

        topics = {record.metadata.get("topic") for record in family_rows}
        propositions = {record.metadata.get("proposition") for record in family_rows}
        if len(topics) != 1:
            errors.append(f"{family}: records do not share one metadata.topic: {sorted(topics)}.")
        if len(propositions) != 1:
            errors.append(
                f"{family}: records do not share one metadata.proposition: "
                f"{sorted(propositions)}."
            )

    return PromptDatasetValidation(
        path=path_value,
        records=records,
        errors=errors,
        warnings=warnings,
        counts_by_stance=Counter(record.stance for record in records if record.stance),
        counts_by_style=Counter(record.style for record in records if record.style),
        counts_by_stance_style=Counter(
            (record.stance, record.style)
            for record in records
            if record.stance and record.style
        ),
        counts_by_topic=Counter(
            str(record.metadata["topic"])
            for record in records
            if isinstance(record.metadata.get("topic"), str)
            and record.metadata["topic"].strip()
        ),
    )


def read_prompt_records_collecting_errors(path: Path) -> tuple[list[PromptRecord], list[str]]:
    """Parse prompt records while collecting line-level failures."""

    records: list[PromptRecord] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"Prompt dataset does not exist: {path}."]

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}.")
                continue
            if not isinstance(raw, dict):
                errors.append(f"{path}:{line_number}: expected a JSON object.")
                continue
            try:
                records.append(PromptRecord.from_mapping(raw))
            except ValueError as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    return records, errors


def normalize_text(text: str) -> str:
    """Normalize text for duplicate diagnostics."""

    return re.sub(r"\s+", " ", text.strip().lower())
