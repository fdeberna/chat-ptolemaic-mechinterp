"""Typed prompt records and categorical validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALLOWED_STANCES = {"geocentric", "heliocentric"}
ALLOWED_STYLES = {"modern", "premodern"}
ALLOWED_FRAMEWORKS = {"modern_astronomy", "ptolemaic", "aristotelian", "observational"}
ALLOWED_ATTRIBUTIONS = {
    "asserted",
    "none",
    "historical_authority",
    "modern_science",
    "textbook",
}


@dataclass(frozen=True)
class PromptRecord:
    """One controlled prompt and its labels."""

    prompt_id: str
    text: str
    template_family: str
    stance: str | None = None
    style: str | None = None
    framework: str | None = None
    attribution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PromptRecord:
        """Build and validate a prompt record from a JSON object."""

        required_string(data, "prompt_id")
        required_string(data, "text")
        required_string(data, "template_family")
        metadata = data.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("'metadata' must be a JSON object.")

        stance = optional_category(data.get("stance"), ALLOWED_STANCES, "stance")
        style = optional_category(data.get("style"), ALLOWED_STYLES, "style")
        framework = optional_category(data.get("framework"), ALLOWED_FRAMEWORKS, "framework")
        attribution = optional_category(
            data.get("attribution"), ALLOWED_ATTRIBUTIONS, "attribution"
        )

        return cls(
            prompt_id=str(data["prompt_id"]),
            text=str(data["text"]),
            template_family=str(data["template_family"]),
            stance=stance,
            style=style,
            framework=framework,
            attribution=attribution,
            metadata=metadata,
        )


def required_string(data: dict[str, Any], key: str) -> None:
    """Validate a required non-empty string field."""

    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string.")


def optional_category(value: Any, allowed: set[str], field_name: str) -> str | None:
    """Validate a nullable categorical field."""

    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be a string or null.")
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {field_name} '{value}'. Allowed values: {allowed_values}.")
    return value
