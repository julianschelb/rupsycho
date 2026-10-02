"""Tests for exporting results: ``to_config``, ``export_to_file``, ``get_answers`` and friends."""

import copy
import json

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate

import rupsycho as rup
from rupsycho.experiment import ExperimentDocument
from rupsycho.models.model import LangChainModelConfig
from rupsycho.models.questionnaire import DemographicProfile

COLUMNS = [
    "Instruction ID",
    "Instruction Question",
    "Model ID",
    "Persona ID",
    "Run Seed",
    "Answer",
]
CONFIG_KEYS = [
    "name",
    "description",
    "parameters",
    "prompt_template",
    "models",
    "demographic_profiles",
    "questionnaire",
]
MASK = "**********"


class BrokenLLM(FakeListLLM):
    """A fake LLM whose every call fails."""

    def _call(self, *args, **kwargs):
        raise RuntimeError("boom")


def numbered_llm(tag):
    """The k-th call of this model answers ``<tag>-<k>``."""
    return FakeListLLM(responses=[f"{tag}-{i}" for i in range(500)])


def make_experiment(config_dict, models, seeds=("1", "2")):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = list(seeds)
    experiment = rup.experiment_from_dict(config)
    for identifier, model in models.items():
        experiment.add_model(model, identifier=identifier)
    return experiment


@pytest.fixture
def rich_experiment(config_dict):
    """Two models x two seeds x two personas x four items, already run."""
    experiment = make_experiment(config_dict, {"m1": numbered_llm("m1"), "m2": numbered_llm("m2")})
    experiment.run(show_progress=False)
    return experiment


def expected_answers(experiment):
    """The answers of ``rich_experiment``: model call number ``k`` answered ``<model>-<k>``."""
    n_items = len(experiment.questionnaire.instruction_items)
    personas = list(experiment.demographic_profiles)
    seeds = experiment.parameters.seeds
    per_seed = n_items * len(personas)
    expected = []
    for item_id in range(n_items):
        item_answers = {}
        for model_id in ("m1", "m2"):
            item_answers[model_id] = {
                persona: {
                    seed: f"{model_id}-{seed_no * per_seed + item_id * len(personas) + persona_no}"
                    for seed_no, seed in enumerate(seeds)
                }
                for persona_no, persona in enumerate(personas)
            }
        expected.append(item_answers)
    return expected


def good_and_broken(config_dict):
    """One working and one failing model, run once; the failures are expected (and silent)."""
    experiment = make_experiment(
        config_dict, {"good": numbered_llm("g"), "bad": BrokenLLM(responses=["x"])}, seeds=["1"]
    )
    summary = experiment.run(on_error="ignore", show_progress=False)
    assert (summary.n_calls, summary.n_failed) == (16, 8)
    return experiment


# ------------------------------------------------------------------------------------------
# get_answers
# ------------------------------------------------------------------------------------------


def test_get_answers_nests_item_model_persona_seed(rich_experiment):
    answers = rich_experiment.get_answers()

    assert answers == expected_answers(rich_experiment)
    assert len(answers) == 4
    assert answers[0]["m1"]["Optimistic Persona"] == {"1": "m1-0", "2": "m1-8"}
    assert answers[3]["m2"]["Conservative Persona"] == {"1": "m2-7", "2": "m2-15"}


def test_get_answers_before_a_run_is_one_empty_dict_per_item(fake_experiment):
    assert fake_experiment.get_answers() == [{}, {}, {}, {}]


def test_get_answers_returns_plain_json_serialisable_dicts(rich_experiment):
    answers = rich_experiment.get_answers()

    for item_answers in answers:
        assert type(item_answers) is dict
        for model_answers in item_answers.values():
            assert type(model_answers) is dict
            assert all(type(persona_answers) is dict for persona_answers in model_answers.values())
    assert json.loads(json.dumps(answers)) == answers


def test_get_answers_returns_a_copy(rich_experiment):
    answers = rich_experiment.get_answers()
    answers[0]["m1"]["Optimistic Persona"]["1"] = "tampered"
    answers[1].clear()

    assert rich_experiment.get_answers() == expected_answers(rich_experiment)


def test_get_answers_leaves_out_failed_calls(config_dict):
    experiment = good_and_broken(config_dict)

    for item_answers in experiment.get_answers():
        assert list(item_answers) == ["good"]


# ------------------------------------------------------------------------------------------
# get_answers_as_dataframe
# ------------------------------------------------------------------------------------------


def test_dataframe_has_the_documented_columns(rich_experiment):
    df = rich_experiment.get_answers_as_dataframe()

    assert list(df.columns) == COLUMNS


def test_dataframe_has_one_row_per_model_seed_persona_and_item(rich_experiment):
    df = rich_experiment.get_answers_as_dataframe()

    assert len(df) == 2 * 2 * 2 * 4
    assert df["Model ID"].value_counts().to_dict() == {"m1": 16, "m2": 16}
    assert df["Run Seed"].value_counts().to_dict() == {"1": 16, "2": 16}
    assert df["Persona ID"].value_counts().to_dict() == {
        "Optimistic Persona": 16,
        "Conservative Persona": 16,
    }
    assert df["Instruction ID"].value_counts().to_dict() == {0: 8, 1: 8, 2: 8, 3: 8}


def test_dataframe_rows_are_ordered_by_item_model_persona_and_seed(rich_experiment):
    df = rich_experiment.get_answers_as_dataframe()

    questions = [item.question for item in rich_experiment.questionnaire.instruction_items]
    personas = list(rich_experiment.demographic_profiles)
    expected = []
    for item_id, item_answers in enumerate(expected_answers(rich_experiment)):
        for model_id in ("m1", "m2"):
            for persona in personas:
                for seed, answer in item_answers[model_id][persona].items():
                    expected.append((item_id, questions[item_id], model_id, persona, seed, answer))
    assert list(df.itertuples(index=False, name=None)) == expected
    assert df["Instruction ID"].dtype == "int64"


def test_dataframe_agrees_with_get_answers(rich_experiment):
    df = rich_experiment.get_answers_as_dataframe()
    nested = rich_experiment.get_answers()

    rebuilt = {
        (row["Instruction ID"], row["Model ID"], row["Persona ID"], row["Run Seed"]): row["Answer"]
        for _, row in df.iterrows()
    }
    flattened = {
        (item_id, model_id, persona, seed): answer
        for item_id, item_answers in enumerate(nested)
        for model_id, model_answers in item_answers.items()
        for persona, persona_answers in model_answers.items()
        for seed, answer in persona_answers.items()
    }
    assert rebuilt == flattened


def test_dataframe_of_a_single_model_experiment(fake_experiment):
    fake_experiment.run(show_progress=False)

    df = fake_experiment.get_answers_as_dataframe()

    assert df.shape == (8, 6)
    assert set(df["Model ID"]) == {"fake"}
    assert set(df["Run Seed"]) == {"7"}
    assert set(df["Answer"]) == {"3"}


def test_dataframe_leaves_out_failed_calls(config_dict):
    experiment = good_and_broken(config_dict)

    df = experiment.get_answers_as_dataframe()

    assert set(df["Model ID"]) == {"good"}
    assert len(df) == 8


def test_dataframe_uses_the_default_seed_when_none_is_configured(config_dict):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = []  # falls back to the default seed 42
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    experiment.run(show_progress=False)

    assert {str(seed) for seed in experiment.get_answers_as_dataframe()["Run Seed"]} == {"42"}


def test_dataframe_without_answers_still_has_the_documented_columns(fake_experiment):
    df = fake_experiment.get_answers_as_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLUMNS
    assert df.empty
    assert df["Answer"].tolist() == []  # selecting a column of an empty result works


def test_dataframe_of_a_run_in_which_every_call_failed_has_the_documented_columns(config_dict):
    experiment = make_experiment(config_dict, {"bad": BrokenLLM(responses=["x"])}, seeds=["1"])
    experiment.run(on_error="ignore", show_progress=False)

    df = experiment.get_answers_as_dataframe()

    assert df.empty and list(df.columns) == COLUMNS


# ------------------------------------------------------------------------------------------
# to_config
# ------------------------------------------------------------------------------------------


def test_to_config_returns_the_configuration_sections(fake_experiment):
    data = fake_experiment.to_config()

    assert list(data) == CONFIG_KEYS
    assert data["name"] == "Generative Models for Big Five Inventory"
    assert data["parameters"] == {"seeds": ["7"], "lazy_load_models": True}
    assert data["prompt_template"]["type"] == "chat"
    assert [m["role"] for m in data["prompt_template"]["messages"]] == ["system", "user"]
    assert list(data["models"]) == ["fake"]
    assert data["models"]["fake"]["definition"]["name"] == "FakeListLLM"
    assert list(data["demographic_profiles"]) == ["Optimistic Persona", "Conservative Persona"]
    assert data["demographic_profiles"]["Conservative Persona"] == {
        "attributes": {"age": 65, "title": "Mr", "name": "Grueber"},
        "template": (
            "{title} {name} is {age} years old and very conservative with a reserved personality."
        ),
    }
    assert data["questionnaire"]["name"].startswith("BIG FIVE INVENTORY")
    assert len(data["questionnaire"]["instruction_items"]) == 4


def test_to_config_is_plain_json_data(fake_experiment):
    data = fake_experiment.to_config()

    assert json.loads(json.dumps(data)) == data


def test_to_config_leaves_out_values_that_are_not_set(config_dict):
    config = copy.deepcopy(config_dict)
    del config["description"]
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    data = experiment.to_config()

    assert "description" not in data
    assert "prompt_template" not in data["models"]["fake"]  # an unset option of a nested model


def test_to_config_keeps_empty_models_and_profiles(config_dict):
    """``{}`` and "missing" differ: an experiment loaded without ``models`` gets a default model."""
    config = copy.deepcopy(config_dict)
    config["demographic_profiles"] = {}
    experiment = rup.experiment_from_dict(config)

    data = experiment.to_config()

    assert data["models"] == {} and data["demographic_profiles"] == {}
    reloaded = rup.experiment_from_dict(data)
    assert reloaded.list_models() == [] and reloaded.list_personas() == []


def test_to_config_contains_the_answers_by_default(rich_experiment):
    items = rich_experiment.to_config()["questionnaire"]["instruction_items"]

    assert [item["answers"] for item in items] == expected_answers(rich_experiment)


def test_to_config_before_a_run_has_empty_answers(fake_experiment):
    items = fake_experiment.to_config()["questionnaire"]["instruction_items"]

    assert [item["answers"] for item in items] == [{}, {}, {}, {}]


def test_to_config_without_answers_is_the_definition_of_the_experiment(rich_experiment):
    definition = rich_experiment.to_config(include_answers=False)
    with_answers = rich_experiment.to_config()

    assert not any("answers" in item for item in definition["questionnaire"]["instruction_items"])
    for item in with_answers["questionnaire"]["instruction_items"]:
        del item["answers"]
    assert definition == with_answers  # nothing but the answers differs


def test_a_definition_without_answers_can_be_run_again_from_scratch(rich_experiment):
    definition = rich_experiment.to_config(include_answers=False)

    reloaded = rup.experiment_from_dict(definition)
    reloaded.clear_models()  # the exported fakes cannot be rebuilt from their definitions
    reloaded.add_model(FakeListLLM(responses=["fresh"]), identifier="m3")
    assert reloaded.get_answers() == [{}, {}, {}, {}]
    reloaded.run(show_progress=False)

    assert set(reloaded.get_answers_as_dataframe()["Model ID"]) == {"m3"}
    assert len(reloaded.get_answers_as_dataframe()) == 2 * 2 * 4


def test_to_config_returns_independent_data(fake_experiment):
    data = fake_experiment.to_config()
    data["questionnaire"]["instruction_items"][0]["question"] = "changed"
    data["demographic_profiles"]["Optimistic Persona"]["attributes"]["name"] = "changed"
    data["parameters"]["seeds"].append("99")

    assert fake_experiment.questionnaire.instruction_items[0].question != "changed"
    assert fake_experiment.demographic_profiles["Optimistic Persona"].attributes.name == "Muller"
    assert fake_experiment.parameters.seeds == ["7"]
    assert fake_experiment.to_config()["parameters"]["seeds"] == ["7"]


def test_to_config_fails_loudly_for_data_that_cannot_be_serialised(fake_experiment):
    fake_experiment.parameters.unserialisable = object()  # extra parameters are exported as is

    with pytest.raises(ValueError):
        fake_experiment.to_config()


def test_model_dump_of_a_run_experiment_contains_the_answers(rich_experiment):
    """``model_dump`` is plain pydantic: everything is in it, including the runtime objects.
    Use ``to_config`` for what is meant to be shared."""
    dump = rich_experiment.model_dump()

    items = dump["questionnaire"]["instruction_items"]
    assert [item["answers"] for item in items] == expected_answers(rich_experiment)
    assert {"runnable_models", "runnable_prompt"} <= set(dump)
    assert not {"runnable_models", "runnable_prompt"} & set(rich_experiment.to_config())


# ------------------------------------------------------------------------------------------
# export_to_file
# ------------------------------------------------------------------------------------------


def export(experiment, path, **kwargs):
    experiment.export_to_file(str(path), **kwargs)
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_indented_json(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    text = path.read_text(encoding="utf-8")
    assert text.startswith('{\n    "')
    assert json.loads(text)["name"] == fake_experiment.name


def test_export_writes_what_to_config_returns(rich_experiment, tmp_path):
    assert export(rich_experiment, tmp_path / "experiment.json") == rich_experiment.to_config()
    without = export(rich_experiment, tmp_path / "definition.json", include_answers=False)
    assert without == rich_experiment.to_config(include_answers=False)


def test_export_accepts_path_objects(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(path)

    assert json.loads(path.read_text(encoding="utf-8"))["name"] == fake_experiment.name


def test_export_contains_the_configuration_sections(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    assert list(data) == CONFIG_KEYS
    assert data["parameters"]["seeds"] == ["7"]
    assert list(data["demographic_profiles"]) == ["Optimistic Persona", "Conservative Persona"]
    assert data["models"]["fake"]["type"] == "langchain"


def test_export_keeps_non_ascii_characters_readable(fake_experiment, tmp_path):
    fake_experiment.add_persona(
        DemographicProfile(attributes={"name": "Müller", "title": "Frau", "age": 31}),
        identifier="Müller",
    )
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    text = path.read_bytes().decode("utf-8")
    assert "Müller" in text
    assert "\\u00fc" not in text


def test_export_overwrites_an_existing_file(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text("old content that is much longer than the new one" * 5000)

    fake_experiment.export_to_file(str(path))
    first_size = path.stat().st_size
    fake_experiment.export_to_file(str(path))

    assert json.loads(path.read_text(encoding="utf-8"))["name"] == fake_experiment.name
    assert path.stat().st_size == first_size


def test_export_to_an_unwritable_location_raises_os_error(fake_experiment, tmp_path):
    with pytest.raises(OSError):
        fake_experiment.export_to_file(str(tmp_path / "no" / "such" / "dir" / "experiment.json"))

    assert not (tmp_path / "no").exists()


def test_export_to_a_directory_raises_os_error(fake_experiment, tmp_path):
    with pytest.raises(OSError):
        fake_experiment.export_to_file(str(tmp_path))


def test_export_without_answers(rich_experiment, tmp_path):
    data = export(rich_experiment, tmp_path / "definition.json", include_answers=False)

    assert not any("answers" in item for item in data["questionnaire"]["instruction_items"])
    assert rup.experiment_from_dict(data).get_answers() == [{}, {}, {}, {}]


SECRET_MODELS = {
    "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": "sk-visible"},
    "gemini": {"type": "google", "name_or_path": "gemini-2.0-flash", "api_key": "google-secret"},
    "deepseek": {"type": "deepseek", "name_or_path": "deepseek-chat", "api_key": "ds-secret"},
    "local": {
        "type": "local_huggingface",
        "name_or_path": "org/model",
        "huggingfacehub_api_token": "hf-secret",
    },
}
SECRET_FIELDS = {
    "gpt": "api_key",
    "gemini": "api_key",
    "deepseek": "api_key",
    "local": "huggingfacehub_api_token",
}


def test_export_masks_the_secrets_of_configured_models(config_dict, tmp_path):
    """An exported file is meant to be shared: it never contains an API key or token."""
    config = copy.deepcopy(config_dict)
    config["models"] = copy.deepcopy(SECRET_MODELS)
    experiment = rup.experiment_from_dict(config)
    path = tmp_path / "experiment.json"

    experiment.export_to_file(path)

    text = path.read_text(encoding="utf-8")
    for secret in ("sk-visible", "google-secret", "ds-secret", "hf-secret"):
        assert secret not in text
    data = json.loads(text)
    for identifier, field in SECRET_FIELDS.items():
        assert data["models"][identifier][field] == MASK
        # everything else of the model configuration is exported as it is
        exported = {k: v for k, v in data["models"][identifier].items() if k != field}
        expected = {k: v for k, v in SECRET_MODELS[identifier].items() if k != field}
        assert expected.items() <= exported.items()


def test_a_masked_export_loads_without_keys_so_that_the_environment_applies(config_dict, tmp_path):
    config = copy.deepcopy(config_dict)
    config["models"] = copy.deepcopy(SECRET_MODELS)
    original = rup.experiment_from_dict(config)
    path = tmp_path / "experiment.json"
    original.export_to_file(path)

    reloaded = rup.experiment_from_file(path)

    for identifier, field in SECRET_FIELDS.items():
        assert getattr(reloaded.models[identifier], field) is None
        assert getattr(original.models[identifier], field).get_secret_value()  # still there
    assert list(reloaded.models) == list(original.models)


# ---- round trips ---------------------------------------------------------------------------------


def assert_same_experiment(reloaded, original):
    assert reloaded.name == original.name
    assert reloaded.description == original.description
    assert reloaded.parameters.seeds == original.parameters.seeds
    assert reloaded.parameters.lazy_load_models == original.parameters.lazy_load_models
    assert reloaded.prompt_template == original.prompt_template
    assert reloaded.list_models() == original.list_models()
    assert reloaded.demographic_profiles == original.demographic_profiles
    assert reloaded.questionnaire.name == original.questionnaire.name
    assert reloaded.questionnaire.general_instruction == original.questionnaire.general_instruction
    assert (
        reloaded.questionnaire.default_answer_options
        == original.questionnaire.default_answer_options
    )
    assert [i.question for i in reloaded.questionnaire.instruction_items] == [
        i.question for i in original.questionnaire.instruction_items
    ]
    assert [i.attributes for i in reloaded.questionnaire.instruction_items] == [
        i.attributes for i in original.questionnaire.instruction_items
    ]


def test_exported_experiment_can_be_loaded_again_after_a_run(rich_experiment, tmp_path):
    data = export(rich_experiment, tmp_path / "experiment.json")

    reloaded = rup.experiment_from_dict(data)

    assert_same_experiment(reloaded, rich_experiment)
    # the answers come back with the questionnaire, exactly as they were exported
    assert (
        reloaded.get_answers() == rich_experiment.get_answers() == expected_answers(rich_experiment)
    )
    pd.testing.assert_frame_equal(
        reloaded.get_answers_as_dataframe(), rich_experiment.get_answers_as_dataframe()
    )


def test_exported_experiment_can_be_loaded_from_the_file(rich_experiment, tmp_path):
    path = tmp_path / "experiment.json"
    rich_experiment.export_to_file(str(path))

    reloaded = rup.experiment_from_file(str(path))

    assert isinstance(reloaded, ExperimentDocument)
    assert_same_experiment(reloaded, rich_experiment)
    assert reloaded.get_answers() == rich_experiment.get_answers()


def test_exported_experiment_can_be_loaded_again_before_a_run(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    reloaded = rup.experiment_from_dict(data)

    assert_same_experiment(reloaded, fake_experiment)
    assert reloaded.get_answers() == [{}, {}, {}, {}]


def test_export_of_the_unmodified_configuration_round_trips(config_dict, tmp_path):
    experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

    assert reloaded.prompt_template == experiment.prompt_template
    assert reloaded.demographic_profiles == experiment.demographic_profiles
    assert reloaded.questionnaire.model_dump() == experiment.questionnaire.model_dump()


def test_to_config_round_trips_without_a_file(rich_experiment):
    reloaded = rup.experiment_from_dict(rich_experiment.to_config())

    assert_same_experiment(reloaded, rich_experiment)
    assert reloaded.to_config() == rich_experiment.to_config()


def test_default_random_seed_survives_the_round_trip(config_dict, tmp_path):
    """Without configured seeds a random one is drawn; the answers are filed under it."""
    config = copy.deepcopy(config_dict)
    del config["parameters"]["seeds"]
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    experiment.run(show_progress=False)
    (seed,) = experiment.parameters.seeds

    reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

    assert reloaded.parameters.seeds == [seed]
    assert set(reloaded.get_answers_as_dataframe()["Run Seed"]) == {seed}


def test_models_from_the_configuration_round_trip(config_dict, tmp_path):
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "hf": {
            "type": "local_huggingface",
            "name_or_path": "org/model",
            "parameters": {"max_new_tokens": 3},
        },
        "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "base_url": "http://proxy"},
    }
    experiment = rup.experiment_from_dict(config)

    reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

    assert reloaded.models == experiment.models
    assert [type(m) for m in reloaded.models.values()] == [
        type(m) for m in experiment.models.values()
    ]


def test_the_key_is_the_only_difference_after_a_round_trip(config_dict):
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "gpt": {
            "type": "openai",
            "name_or_path": "gpt-4o-mini",
            "api_key": "sk-secret",
            "base_url": "http://proxy",
            "parameters": {"temperature": 0.2},
        }
    }
    experiment = rup.experiment_from_dict(config)

    reloaded = rup.experiment_from_dict(experiment.to_config())

    assert reloaded.models["gpt"] == experiment.models["gpt"].model_copy(update={"api_key": None})


def test_all_prompt_template_types_round_trip(config_dict, tmp_path):
    langchain_prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    from langchain_core.load import dumpd

    for prompt in (
        {"type": "normal", "template": "{question} / {answer_options}"},
        {"type": "chat", "messages": [{"role": "user", "content": "{question}"}]},
        {"type": "langchain", "definition": dumpd(langchain_prompt)},
    ):
        config = copy.deepcopy(config_dict)
        config["prompt_template"] = prompt
        experiment = rup.experiment_from_dict(config)

        reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

        assert reloaded.prompt_template == experiment.prompt_template
        assert type(reloaded.prompt_template) is type(experiment.prompt_template)


def test_export_before_the_run_with_fake_models_round_trips_the_model_ids(config_dict, tmp_path):
    experiment = make_experiment(config_dict, {"m1": numbered_llm("m1"), "m2": numbered_llm("m2")})

    data = export(experiment, tmp_path / "experiment.json")

    assert list(data["models"]) == ["m1", "m2"]
    assert rup.experiment_from_dict(data).list_models() == ["m1", "m2"]


def test_add_model_config_in_the_export_is_a_langchain_config(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    config = LangChainModelConfig(**data["models"]["fake"])
    assert config.definition["name"] == "FakeListLLM"


# ---- bugs that were fixed: regression tests --------------------------------------------------------


def test_experiment_with_a_prompt_set_in_code_can_be_exported_and_loaded_again(
    fake_experiment, tmp_path
):
    fake_experiment.set_prompt(ChatPromptTemplate.from_messages([("user", "{question}")]))
    fake_experiment.run(show_progress=False)

    data = export(fake_experiment, tmp_path / "experiment.json")

    reloaded = rup.experiment_from_dict(data)
    assert reloaded.get_answers() == fake_experiment.get_answers()
    assert data["prompt_template"]["type"] == "langchain"


def test_experiment_with_a_real_chat_model_can_be_exported_without_leaking_its_key(
    fake_experiment, tmp_path
):
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    fake_experiment.add_model(
        ChatOpenAI(model="gpt-4o-mini", api_key="sk-secret"), identifier="gpt"
    )
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["models"]["gpt"]["type"] == "langchain"
    assert "sk-secret" not in text


def test_export_contains_no_runtime_state(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    assert not {"runnable_prompt", "runnable_models", "runnable_parser", "id", "type"} & set(data)
    assert set(data) <= set(CONFIG_KEYS)


def test_reloaded_experiment_can_be_run_again(rich_experiment, tmp_path):
    """Loaded answers are plain dicts; adding the answers of a new model must still work."""
    reloaded = rup.experiment_from_dict(export(rich_experiment, tmp_path / "experiment.json"))
    reloaded.clear_models()
    reloaded.add_model(FakeListLLM(responses=["new"]), identifier="m3")

    summary = reloaded.run(show_progress=False)

    assert summary.n_failed == 0
    df = reloaded.get_answers_as_dataframe()
    assert len(df) == 2 * 2 * 2 * 4 + 1 * 2 * 2 * 4
    assert set(df[df["Model ID"] == "m3"]["Answer"]) == {"new"}
    assert set(df["Model ID"]) == {"m1", "m2", "m3"}  # the loaded answers are still there


def test_failed_export_keeps_the_previous_file(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"
    fake_experiment.export_to_file(str(path))
    previous = path.read_text(encoding="utf-8")
    fake_experiment.parameters.unserialisable = object()  # extra parameters are exported as is

    with pytest.raises(ValueError):
        fake_experiment.export_to_file(str(path))

    assert path.read_text(encoding="utf-8") == previous


def test_export_writes_utf8_regardless_of_the_platform_default_encoding(
    fake_experiment, tmp_path, monkeypatch
):
    import rupsycho.mixins.experiment_exporting as exporting

    real_open = open

    def open_with_ascii_default(file, mode="r", *args, **kwargs):
        kwargs.setdefault("encoding", "ascii")  # what open() would use on such a platform
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(exporting, "open", open_with_ascii_default, raising=False)
    fake_experiment.add_persona(
        DemographicProfile(attributes={"name": "Müller"}), identifier="Müller"
    )
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    assert "Müller" in path.read_bytes().decode("utf-8")
    assert "Müller" in rup.experiment_from_file(path).demographic_profiles


# ---- open questions ------------------------------------------------------------------------------


def test_metadata_survives_an_export_round_trip(config_dict, tmp_path):
    config = {**copy.deepcopy(config_dict), "metadata": {"source": "Lab A", "run": 3}}
    experiment = rup.experiment_from_dict(config)

    reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

    assert reloaded.metadata == {"source": "Lab A", "run": 3}


def test_running_a_reloaded_experiment_with_the_default_seed_replaces_its_answers(config_dict):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = []
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["first"]), identifier="fake")
    experiment.run(show_progress=False)
    assert len(experiment.get_answers_as_dataframe()) == 8

    reloaded = rup.experiment_from_dict(experiment.to_config())
    reloaded.clear_models()
    reloaded.add_model(FakeListLLM(responses=["second"]), identifier="fake")
    reloaded.run(show_progress=False)

    df = reloaded.get_answers_as_dataframe()
    assert len(df) == 8
    assert set(df["Answer"]) == {"second"}
