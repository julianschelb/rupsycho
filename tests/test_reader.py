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
