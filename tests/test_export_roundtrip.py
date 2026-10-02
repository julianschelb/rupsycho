"""An exported experiment is a shareable configuration that loads again and can be re-run."""

import copy
import json

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate

import rupsycho as rup


@pytest.fixture
def ran(fake_experiment):
    fake_experiment.run(show_progress=False)
    return fake_experiment


def test_to_config_contains_only_the_configuration(ran):
    config = ran.to_config()
    assert set(config) <= {
        "name",
        "description",
        "parameters",
        "models",
        "prompt_template",
        "demographic_profiles",
        "questionnaire",
    }
    assert "runnable_prompt" not in config and "runnable_models" not in config


def test_answers_can_be_left_out(ran):
    with_answers = ran.to_config()
    without = ran.to_config(include_answers=False)
    items = with_answers["questionnaire"]["instruction_items"]
    assert all(item["answers"] for item in items)
    assert not any(item.get("answers") for item in without["questionnaire"]["instruction_items"])


def test_exported_file_is_utf8_json_and_loads_again(ran, tmp_path, config_dict):
    ran.name = "Größe – 日本語"
    path = tmp_path / "experiment.json"
    ran.export_to_file(path)

    text = path.read_text(encoding="utf-8")
    assert "Größe – 日本語" in text
    reloaded = rup.experiment_from_dict({**json.loads(text), "models": {}})
    assert reloaded.name == "Größe – 日本語"
    assert reloaded.get_answers() == [dict(a) for a in ran.get_answers()]


def test_a_reloaded_experiment_can_be_run_again(ran, tmp_path, config_dict):
    """Loaded answers come back as plain dicts; storing new answers must still work."""
    path = tmp_path / "experiment.json"
    ran.export_to_file(path)
    reloaded = rup.experiment_from_dict({**json.loads(path.read_text()), "models": {}})
    reloaded.add_model(FakeListLLM(responses=["5"]), identifier="other-model")

    summary = reloaded.run(show_progress=False)

    assert summary.n_failed == 0
    assert set(reloaded.get_answers_as_dataframe()["Model ID"]) == {"fake", "other-model"}


def test_set_prompt_survives_the_export(fake_experiment, tmp_path, config_dict):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "As {persona_description}"), ("user", "{question} {answer_options}")]
    )
    fake_experiment.set_prompt(prompt)

    path = tmp_path / "experiment.json"
    fake_experiment.export_to_file(path)
    reloaded = rup.experiment_from_dict({**json.loads(path.read_text()), "models": {}})
    reloaded.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    assert reloaded.run(show_progress=False).n_failed == 0
    assert reloaded.get_prompt().format(
        persona_description="P", question="Q", answer_options="A"
    ) == prompt.format(persona_description="P", question="Q", answer_options="A")


def test_exporting_does_not_modify_the_experiment(ran):
    before = copy.deepcopy(ran.get_answers())
    ran.to_config()
    ran.to_config(include_answers=False)
    assert ran.get_answers() == before


def test_unwritable_target_raises(ran, tmp_path):
    with pytest.raises(OSError):
        ran.export_to_file(tmp_path / "missing-dir" / "out.json")


class TestSecrets:
    def test_api_keys_are_masked_in_repr_and_exports(self, config_dict):
        config = copy.deepcopy(config_dict)
        config["models"] = {
            "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": "sk-secret-123"},
            "hf": {
                "type": "local_huggingface",
                "name_or_path": "org/model",
                "huggingfacehub_api_token": "hf_secret_456",
            },
        }
        experiment = rup.experiment_from_dict(config)

        exported = json.dumps(experiment.to_config())
        assert "sk-secret-123" not in exported and "hf_secret_456" not in exported
        assert "sk-secret-123" not in repr(experiment.models) + str(experiment.models)
        assert experiment.models["gpt"].api_key.get_secret_value() == "sk-secret-123"

    @pytest.mark.parametrize("placeholder", ["**********", "", "  "])
    def test_masked_or_blank_keys_mean_no_key(self, config_dict, placeholder):
        config = copy.deepcopy(config_dict)
        config["models"] = {
            "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": placeholder}
        }
        assert rup.experiment_from_dict(config).models["gpt"].api_key is None

    def test_exported_configuration_falls_back_to_environment_keys(self, config_dict, monkeypatch):
        pytest.importorskip("langchain_openai")
        config = copy.deepcopy(config_dict)
        config["models"] = {
            "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": "sk-secret-123"}
        }
        exported = rup.experiment_from_dict(config).to_config()
        monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

        reloaded = rup.experiment_from_dict(exported)
        model = reloaded.models["gpt"].load_model()

        assert model.openai_api_key.get_secret_value() == "from-environment"
