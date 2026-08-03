import pytest

from ptolemaic_mechinterp.data.schemas import PromptRecord


def valid_record() -> dict[str, object]:
    return {
        "prompt_id": "p1",
        "text": "Explain the Sun-centered model.",
        "template_family": "family_a",
        "stance": "heliocentric",
        "style": "modern",
        "framework": "modern_astronomy",
        "attribution": "modern_science",
        "metadata": {"source": "synthetic"},
    }


def test_prompt_record_validation_accepts_valid_record() -> None:
    record = PromptRecord.from_mapping(valid_record())

    assert record.prompt_id == "p1"
    assert record.stance == "heliocentric"
    assert record.metadata == {"source": "synthetic"}


def test_prompt_record_validation_rejects_invalid_stance() -> None:
    raw = valid_record()
    raw["stance"] = "tychonic"

    with pytest.raises(ValueError, match="Invalid stance"):
        PromptRecord.from_mapping(raw)


def test_prompt_record_validation_rejects_invalid_metadata() -> None:
    raw = valid_record()
    raw["metadata"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="metadata"):
        PromptRecord.from_mapping(raw)

