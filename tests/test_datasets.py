import json
from importlib import resources

import pytest
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup


def test_bundled_example_is_listed():
    assert "bfi" in rup.list_examples()


def test_example_config_matches_the_packaged_file():
    path = resources.files("rupsycho.data.examples").joinpath("bfi.json")
    assert rup.load_example_config("bfi") == json.loads(path.read_text(encoding="utf-8"))


def test_example_config_is_a_fresh_copy():
    config = rup.load_example_config("bfi")
    config["parameters"]["seeds"] = ["changed"]
    assert rup.load_example_config("bfi")["parameters"]["seeds"] != ["changed"]


def test_unknown_example_names_the_available_ones():
    with pytest.raises(KeyError, match="bfi"):
        rup.load_example_config("does-not-exist")


def test_example_experiment_runs_with_a_replacement_model():
    experiment = rup.load_example_experiment("bfi", models={})
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    summary = experiment.run(show_progress=False)

    assert summary.n_failed == 0
    assert set(experiment.get_answers_as_dataframe()["Answer"]) == {"3"}


def test_example_experiment_keeps_its_model_by_default():
    experiment = rup.load_example_experiment("bfi")
    assert experiment.list_models() == ["Qwen/Qwen2.5-0.5B-Instruct"]
