import copy
import json

import pytest
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
from rupsycho.callbacks import CSVCallback, JSONLCallback, PrintCallback

from .conftest import CONFIG_PATH


def test_experiment_loads_from_file():
    experiment = rup.experiment_from_file(str(CONFIG_PATH))
    assert experiment is not None
    assert experiment.questionnaire.instruction_items


def test_experiment_loads_from_dict(config_dict):
    experiment = rup.experiment_from_dict(config_dict)
    assert experiment.name == config_dict["name"]
    assert len(experiment.demographic_profiles) == len(config_dict["demographic_profiles"])


def test_run_fills_every_answer_slot(fake_experiment):
    fake_experiment.run()

    seeds = fake_experiment.parameters.seeds
    profiles = fake_experiment.demographic_profiles
    for item in fake_experiment.questionnaire.instruction_items:
        for profile_id in profiles:
            for seed in seeds:
                assert item.get_answer("fake", profile_id, seed) == "3"


def test_get_answers_returns_one_entry_per_item(fake_experiment):
    fake_experiment.run()
    answers = fake_experiment.get_answers()

    assert isinstance(answers, list)
    assert len(answers) == len(fake_experiment.questionnaire.instruction_items)


def test_get_answers_as_dataframe_has_one_row_per_combination(fake_experiment):
    fake_experiment.run()
    df = fake_experiment.get_answers_as_dataframe()

    expected = (
        len(fake_experiment.questionnaire.instruction_items)
        * len(fake_experiment.demographic_profiles)
        * len(fake_experiment.parameters.seeds)
    )
    assert len(df) == expected
    assert set(df["Answer"]) == {"3"}


def test_run_without_models_raises(config_dict):
    experiment = rup.experiment_from_dict(config_dict)
    with pytest.raises(ValueError, match="No models"):
        experiment.run()


def test_failing_model_does_not_abort_run(fake_experiment):
    """A chain error is logged and recorded as 'no answer' rather than crashing the run."""

    class Broken(FakeListLLM):
        def _call(self, *args, **kwargs):
            raise RuntimeError("boom")

    fake_experiment.clear_models()
    fake_experiment.add_model(Broken(responses=["x"]), identifier="broken")
    fake_experiment.run()  # must not raise


def test_callbacks_receive_every_answer(fake_experiment, tmp_path):
    jsonl, csv_path = tmp_path / "out.jsonl", tmp_path / "out.csv"
    fake_experiment.run(
        callbacks=[JSONLCallback(str(jsonl)), CSVCallback(str(csv_path)), PrintCallback()]
    )

    expected = len(fake_experiment.get_answers_as_dataframe())
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(rows) == expected
    assert all("time" in row and row["answer"] == "3" for row in rows)
    assert len(csv_path.read_text().splitlines()) == expected + 1  # header


def test_cumulative_run_accumulates_answers(config_dict):
    """Response-memory mode needs a user template that only contains {question}."""
    config = copy.deepcopy(config_dict)
    system = config["prompt_template"]["messages"][0]["content"]
    config["prompt_template"]["messages"] = [
        {"role": "system", "content": system + "\nAnswer Options: {answer_options}"},
        {"role": "user", "content": "Question: {question}\nAnswer:"},
    ]
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    experiment.run(cumulative=True)

    df = experiment.get_answers_as_dataframe()
    assert len(df) > 0
    assert set(df["Answer"]) == {"3"}


def test_export_roundtrip(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"
    fake_experiment.export_to_file(str(path))
    assert json.loads(path.read_text())["name"] == fake_experiment.name


@pytest.mark.integration
def test_experiment_runs_with_configured_model():
    """Runs the configured Hugging Face model end to end (downloads weights)."""
    experiment = rup.experiment_from_file(str(CONFIG_PATH))
    experiment.run()
    assert isinstance(experiment.get_answers(), list)
