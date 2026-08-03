from pathlib import Path

import pytest

from ptolemaic_mechinterp.config import load_extraction_config, load_probe_config


def test_load_extraction_config(tmp_path: Path) -> None:
    config_path = tmp_path / "extraction.yaml"
    config_path.write_text(
        """
model:
  model_name_or_path: tiny-model
  adapter_path:
  device:
  dtype: float32
  load_in_4bit: false
  tokenizer_kwargs:
    use_fast: true
prompt_dataset_path: data/prompts/example_prompts.jsonl
activation_output_dir: results/activations
batch_size: 2
max_length: 128
random_seed: 17
""",
        encoding="utf-8",
    )

    config = load_extraction_config(config_path)

    assert config.model.model_name_or_path == "tiny-model"
    assert config.model.adapter_path is None
    assert config.model.dtype == "float32"
    assert config.model.tokenizer_kwargs == {"use_fast": True}
    assert config.prompt_dataset_path == Path("data/prompts/example_prompts.jsonl")
    assert config.batch_size == 2
    assert config.max_length == 128
    assert config.random_seed == 17


def test_load_probe_config_requires_target(tmp_path: Path) -> None:
    config_path = tmp_path / "probes.yaml"
    config_path.write_text("grouping_column: template_family\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target"):
        load_probe_config(config_path)

