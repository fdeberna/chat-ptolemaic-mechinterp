from pathlib import Path

import pytest

from ptolemaic_mechinterp.config import load_extraction_config, load_probe_config, repository_root
from ptolemaic_mechinterp.models.loader import choose_tokenizer_source


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
    assert config.prompt_dataset_path == repository_root() / "data/prompts/example_prompts.jsonl"
    assert config.batch_size == 2
    assert config.max_length == 128
    assert config.random_seed == 17


def test_load_probe_config_requires_target(tmp_path: Path) -> None:
    config_path = tmp_path / "probes.yaml"
    config_path.write_text("grouping_column: template_family\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target"):
        load_probe_config(config_path)


def test_choose_tokenizer_source_prefers_adapter_tokenizer(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    source = choose_tokenizer_source("Qwen/Qwen2.5-7B", str(adapter_dir))

    assert source == str(adapter_dir)


def test_choose_tokenizer_source_falls_back_to_base_model(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    source = choose_tokenizer_source("Qwen/Qwen2.5-7B", str(adapter_dir))

    assert source == "Qwen/Qwen2.5-7B"


def test_load_extraction_config_resolves_relative_paths_from_repo_root(tmp_path: Path) -> None:
    config_path = tmp_path / "extraction.yaml"
    config_path.write_text(
        """
model:
  model_name_or_path: Qwen/Qwen2.5-7B
prompt_dataset_path: data/prompts/pilot_stance_style.jsonl
activation_output_dir: results/activations/pilot
""",
        encoding="utf-8",
    )

    config = load_extraction_config(config_path)

    assert config.prompt_dataset_path == repository_root() / "data/prompts/pilot_stance_style.jsonl"
    assert config.activation_output_dir == repository_root() / "results/activations/pilot"
