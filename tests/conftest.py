"""Shared fixtures: experiments backed by a fake LLM so no model is ever downloaded."""

import copy
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup

DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = DATA_DIR / "bfi_test_config.json"


@pytest.fixture(scope="session")
def config_dict() -> dict:
    """The BFI test configuration as a plain dict (without any real model)."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["models"] = {}
    return config


@pytest.fixture
def fake_experiment(config_dict):
    """A fresh BFI experiment wired to a fake LLM that always answers '3'."""
    experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    return experiment
