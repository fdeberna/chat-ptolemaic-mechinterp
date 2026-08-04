import json
from pathlib import Path

from ptolemaic_mechinterp.data.validation import validate_prompt_dataset


def test_prompt_dataset_validation_accepts_complete_2x2_families(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    write_jsonl(
        path,
        [
            make_record("f1_a", "f1", "heliocentric", "modern", "earth_motion", "rotation"),
            make_record("f1_b", "f1", "heliocentric", "premodern", "earth_motion", "rotation"),
            make_record("f1_c", "f1", "geocentric", "modern", "earth_motion", "rotation"),
            make_record("f1_d", "f1", "geocentric", "premodern", "earth_motion", "rotation"),
        ],
    )

    result = validate_prompt_dataset(path)

    assert result.is_valid
    assert result.total_record_count == 4
    assert result.unique_prompt_id_count == 4
    assert result.template_family_count == 1
    assert result.counts_by_stance["heliocentric"] == 2
    assert result.counts_by_style["modern"] == 2
    assert result.counts_by_stance_style[("geocentric", "premodern")] == 1
    assert result.counts_by_topic["earth_motion"] == 4


def test_prompt_dataset_validation_rejects_incomplete_family(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    write_jsonl(
        path,
        [
            make_record("f1_a", "f1", "heliocentric", "modern", "earth_motion", "rotation"),
            make_record("f1_b", "f1", "heliocentric", "premodern", "earth_motion", "rotation"),
            make_record("f1_c", "f1", "geocentric", "modern", "earth_motion", "rotation"),
        ],
    )

    result = validate_prompt_dataset(path)

    assert not result.is_valid
    assert any("expected exactly 4 records" in error for error in result.errors)
    assert any("geocentric + premodern" in error for error in result.errors)


def test_prompt_dataset_validation_rejects_duplicate_ids_and_topic_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.jsonl"
    write_jsonl(
        path,
        [
            make_record("dup", "f1", "heliocentric", "modern", "earth_motion", "rotation"),
            make_record("dup", "f1", "heliocentric", "premodern", "physics", "rotation"),
            make_record("f1_c", "f1", "geocentric", "modern", "earth_motion", "rotation"),
            make_record("f1_d", "f1", "geocentric", "premodern", "earth_motion", "rotation"),
        ],
    )

    result = validate_prompt_dataset(path)

    assert not result.is_valid
    assert any("Duplicate prompt_id" in error for error in result.errors)
    assert any("metadata.topic" in error for error in result.errors)


def make_record(
    prompt_id: str,
    family: str,
    stance: str,
    style: str,
    topic: str,
    proposition: str,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "text": f"{prompt_id} prompt text.",
        "template_family": family,
        "stance": stance,
        "style": style,
        "framework": None,
        "attribution": "asserted",
        "metadata": {
            "topic": topic,
            "proposition": proposition,
            "design": "2x2_stance_style",
            "split_group": family,
            "version": "test",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
