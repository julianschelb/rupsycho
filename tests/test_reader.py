import asyncio
import json

import pytest

import rupsycho as rup
from rupsycho.reader import ExperimentLoader

from .conftest import CONFIG_PATH, DATA_DIR


def test_loader_resolves_single_file():
    loader = ExperimentLoader(str(CONFIG_PATH))
    assert loader.file_paths == [str(CONFIG_PATH)]


def test_loader_resolves_glob_pattern():
    loader = ExperimentLoader(str(DATA_DIR / "*.json"))
    assert str(CONFIG_PATH) in loader.file_paths


def test_loader_without_paths_raises():
    with pytest.raises(ValueError, match="No file paths"):
        list(ExperimentLoader().lazy_load())


def test_invalid_experiment_file_is_skipped(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questionnaire": "not a questionnaire"}))
    assert list(ExperimentLoader(str(bad)).lazy_load()) == []


def test_experiments_from_dicts_yields_documents(config_dict):
    experiments = list(rup.experiments_from_dicts([config_dict, config_dict]))
    assert len(experiments) == 2


def test_experiments_from_files_returns_collection():
    collection = rup.experiments_from_files(str(DATA_DIR / "*.json"))
    assert len(collection) >= 1


def test_async_loading(config_dict):
    async def load():
        return [e async for e in ExperimentLoader().alazy_load_from_dicts([config_dict])]

    assert len(asyncio.run(load())) == 1


# --- single-experiment loaders raise instead of returning None -------------------------


def test_experiment_from_file_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        rup.experiment_from_file(tmp_path / "missing.json")


def test_experiment_from_file_raises_on_invalid_configuration(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questionnaire": "not a questionnaire"}))
    with pytest.raises(ValueError, match="questionnaire"):
        rup.experiment_from_file(bad)


def test_experiment_from_file_raises_on_broken_json(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError):
        rup.experiment_from_file(broken)


def test_experiment_from_dict_raises_with_a_useful_message():
    with pytest.raises(ValueError, match="questionnaire"):
        rup.experiment_from_dict({"questionnaire": 3})


def test_configuration_without_parameters_section_is_valid(config_dict):
    config = {k: v for k, v in config_dict.items() if k != "parameters"}
    experiment = rup.experiment_from_dict(config)
    assert experiment.parameters.seeds  # a seed is generated


def test_strict_loading_raises_instead_of_skipping(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questionnaire": 3}))
    with pytest.raises(ValueError):
        list(ExperimentLoader(str(bad), strict=True).lazy_load())


def test_invalid_experiments_are_logged_when_skipped(tmp_path, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questionnaire": 3}))
    with caplog.at_level("WARNING", logger="rupsycho.reader"):
        assert list(ExperimentLoader(str(bad)).lazy_load()) == []
    assert "bad.json" in caplog.text


def test_experiments_from_files_without_match_raises(tmp_path):
    with pytest.raises(ValueError, match="No experiment files"):
        rup.experiments_from_files(str(tmp_path / "*.json"))


def test_experiments_from_files_orders_by_name(tmp_path, config_dict):
    for name in ("b", "a", "c"):
        (tmp_path / f"{name}.json").write_text(json.dumps({**config_dict, "name": name}))
    collection = rup.experiments_from_files(str(tmp_path / "*.json"))
    assert [e.name for e in collection.experiments] == ["a", "b", "c"]
