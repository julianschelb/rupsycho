"""Tests for exporting results: ``model_dump``, ``export_to_file``, ``get_answers`` and friends."""

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import rupsycho as rup
from rupsycho.mixins.experiment_exporting import ExperimentExportMixin
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


@pytest.fixture(autouse=True)
def _quiet_progress_bars(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")


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
    experiment.run()
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
    experiment = make_experiment(
        config_dict, {"good": numbered_llm("g"), "bad": BrokenLLM(responses=["x"])}, seeds=["1"]
    )
    experiment.run()

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
    fake_experiment.run()

    df = fake_experiment.get_answers_as_dataframe()

    assert df.shape == (8, 6)
    assert set(df["Model ID"]) == {"fake"}
    assert set(df["Run Seed"]) == {"7"}
    assert set(df["Answer"]) == {"3"}


def test_dataframe_leaves_out_failed_calls(config_dict):
    experiment = make_experiment(
        config_dict, {"good": numbered_llm("g"), "bad": BrokenLLM(responses=["x"])}, seeds=["1"]
    )
    experiment.run()

    df = experiment.get_answers_as_dataframe()

    assert set(df["Model ID"]) == {"good"}
    assert len(df) == 8


def test_dataframe_keeps_integer_seeds_as_integers(config_dict):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = []  # falls back to the default seed 42
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    experiment.run()

    assert set(experiment.get_answers_as_dataframe()["Run Seed"]) == {42}


def test_dataframe_before_a_run_is_empty(fake_experiment):
    assert len(fake_experiment.get_answers_as_dataframe()) == 0


@pytest.mark.xfail(
    strict=True,
    reason="an experiment without answers yields a DataFrame without any columns, so "
    "df['Answer'] raises KeyError instead of giving an empty column",
)
def test_dataframe_without_answers_still_has_the_documented_columns(fake_experiment):
    df = fake_experiment.get_answers_as_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLUMNS


# ------------------------------------------------------------------------------------------
# model_dump
# ------------------------------------------------------------------------------------------


# The mixin's own dict export. It is called ``model_dump`` (and shadowed by pydantic's, which
# comes first in the MRO) or ``to_dict`` should it be renamed to stop shadowing pydantic's API.
mixin_dump = getattr(ExperimentExportMixin, "to_dict", None) or ExperimentExportMixin.model_dump


def dump(experiment, **kwargs):
    return mixin_dump(experiment, **kwargs)


def test_mixin_model_dump_returns_the_configuration_sections(fake_experiment):
    data = dump(fake_experiment)

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


def test_mixin_model_dump_is_plain_json_data(fake_experiment):
    data = dump(fake_experiment)
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize(
    "include, expected",
    [
        ({"name"}, ["name"]),
        ({"parameters", "questionnaire"}, ["parameters", "questionnaire"]),
        (
            {"models", "demographic_profiles", "prompt_template"},
            ["prompt_template", "models", "demographic_profiles"],
        ),
        (set(), CONFIG_KEYS),
        (None, CONFIG_KEYS),
        ({"runnable_models", "unknown"}, []),
    ],
)
def test_mixin_model_dump_include(fake_experiment, include, expected):
    assert list(dump(fake_experiment, include=include)) == expected


@pytest.mark.parametrize(
    "exclude, expected",
    [
        ({"questionnaire"}, [k for k in CONFIG_KEYS if k != "questionnaire"]),
        (
            {"name", "description", "models"},
            ["parameters", "prompt_template", "demographic_profiles", "questionnaire"],
        ),
        (set(CONFIG_KEYS), []),
        (set(), CONFIG_KEYS),
        (None, CONFIG_KEYS),
    ],
)
def test_mixin_model_dump_exclude(fake_experiment, exclude, expected):
    assert list(dump(fake_experiment, exclude=exclude)) == expected


def test_mixin_model_dump_exclude_wins_over_include(fake_experiment):
    data = dump(fake_experiment, include={"name", "description"}, exclude={"description"})
    assert list(data) == ["name"]


def test_mixin_model_dump_drops_none_values_by_default(config_dict):
    config = copy.deepcopy(config_dict)
    del config["description"]
    experiment = rup.experiment_from_dict(config)

    assert "description" not in dump(experiment)
    assert dump(experiment, exclude_none=False)["description"] is None


def test_mixin_model_dump_exclude_none_applies_to_nested_models(config_dict):
    experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    without_none = dump(experiment)["models"]["fake"]
    with_none = dump(experiment, exclude_none=False, exclude_unset=False)["models"]["fake"]

    assert "prompt_template" not in without_none
    assert with_none["prompt_template"] is None


def test_mixin_model_dump_omits_empty_models_and_profiles(config_dict):
    config = copy.deepcopy(config_dict)
    config["demographic_profiles"] = {}
    experiment = rup.experiment_from_dict(config)

    data = dump(experiment)

    assert "models" not in data
    assert "demographic_profiles" not in data
    assert "questionnaire" in data


def test_mixin_model_dump_exclude_unset_controls_defaulted_fields(config_dict):
    config = copy.deepcopy(config_dict)
    config["parameters"] = {}
    experiment = rup.experiment_from_dict(config)

    unset_dropped = dump(experiment, exclude_unset=True)["parameters"]
    everything = dump(experiment, exclude_unset=False)["parameters"]

    assert unset_dropped == {}
    assert everything == {"seeds": experiment.parameters.seeds, "lazy_load_models": True}


def test_mixin_model_dump_with_exclude_unset_false_keeps_the_answers(rich_experiment):
    items = dump(rich_experiment, exclude_unset=False)["questionnaire"]["instruction_items"]

    assert [item["answers"] for item in items] == expected_answers(rich_experiment)


@pytest.mark.xfail(
    strict=True,
    reason="the mixin's defaults (exclude_unset=True) drop the collected answers: they are "
    "filled in place, so pydantic does not count 'answers' as set",
)
def test_mixin_model_dump_keeps_the_answers_by_default(rich_experiment):
    items = dump(rich_experiment)["questionnaire"]["instruction_items"]

    assert [item.get("answers") for item in items] == expected_answers(rich_experiment)


def test_mixin_model_dump_tolerates_missing_and_plain_attributes():
    stub = SimpleNamespace(
        name="stub",
        description=None,
        parameters={"seeds": ["1"]},
        prompt_template="a raw template",
        models={"m": {"definition": {}}},
        demographic_profiles={"p": {"attributes": {}}},
        questionnaire={"name": "q"},
    )

    assert mixin_dump(stub) == {
        "name": "stub",
        "parameters": {"seeds": ["1"]},
        "prompt_template": "a raw template",
        "models": {"m": {"definition": {}}},
        "demographic_profiles": {"p": {"attributes": {}}},
        "questionnaire": {"name": "q"},
    }
    assert mixin_dump(SimpleNamespace()) == {}
    assert mixin_dump(SimpleNamespace(name="only")) == {"name": "only"}


def test_model_dump_of_a_run_experiment_contains_the_answers(rich_experiment):
    """Whichever ``model_dump`` is active, the answers must be part of the dump."""
    items = rich_experiment.model_dump()["questionnaire"]["instruction_items"]

    assert [item["answers"] for item in items] == expected_answers(rich_experiment)


# ------------------------------------------------------------------------------------------
# export_to_file
# ------------------------------------------------------------------------------------------


def export(experiment, path):
    experiment.export_to_file(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_indented_json(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    text = path.read_text(encoding="utf-8")
    assert text.startswith('{\n    "')
    assert json.loads(text)["name"] == fake_experiment.name


def test_export_contains_the_configuration_sections(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    for key in CONFIG_KEYS:
        assert key in data
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


def test_export_to_an_unwritable_location_prints_instead_of_raising(
    fake_experiment, tmp_path, capsys
):
    fake_experiment.export_to_file(str(tmp_path / "no" / "such" / "dir" / "experiment.json"))

    assert capsys.readouterr().out.startswith("Error saving to file: ")
    assert not (tmp_path / "no").exists()


def test_export_to_a_directory_prints_instead_of_raising(fake_experiment, tmp_path, capsys):
    fake_experiment.export_to_file(str(tmp_path))

    assert capsys.readouterr().out.startswith("Error saving to file: ")


def test_export_writes_api_keys_of_configured_models(config_dict, tmp_path):
    """docs: the model configurations are exported as they are, including any api_key."""
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "gpt": {"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": "sk-visible"}
    }
    experiment = rup.experiment_from_dict(config)

    data = export(experiment, tmp_path / "experiment.json")

    assert data["models"]["gpt"]["api_key"] == "sk-visible"


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

    assert reloaded is not None
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


def test_default_random_seed_survives_the_round_trip(config_dict, tmp_path):
    """Without configured seeds a random one is drawn; the answers are filed under it."""
    config = copy.deepcopy(config_dict)
    del config["parameters"]["seeds"]
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    experiment.run()
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


def test_all_prompt_template_types_round_trip(config_dict, tmp_path):
    for prompt in (
        {"type": "normal", "template": "{question} / {answer_options}"},
        {"type": "chat", "messages": [{"role": "user", "content": "{question}"}]},
    ):
        config = copy.deepcopy(config_dict)
        config["prompt_template"] = prompt
        experiment = rup.experiment_from_dict(config)

        reloaded = rup.experiment_from_dict(export(experiment, tmp_path / "experiment.json"))

        assert reloaded.prompt_template == experiment.prompt_template


def test_export_before_the_run_with_fake_models_round_trips_the_model_ids(config_dict, tmp_path):
    experiment = make_experiment(config_dict, {"m1": numbered_llm("m1"), "m2": numbered_llm("m2")})

    data = export(experiment, tmp_path / "experiment.json")

    assert list(data["models"]) == ["m1", "m2"]
    assert rup.experiment_from_dict(data).list_models() == ["m1", "m2"]


# ---- known problems ------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="set_prompt stores the raw langchain dumpd() dict as prompt_template, so the exported "
    "file is rejected on reload with 'Unknown or missing prompt type.'",
)
def test_experiment_with_a_prompt_set_in_code_can_be_exported_and_loaded_again(
    fake_experiment, tmp_path
):
    fake_experiment.set_prompt(ChatPromptTemplate.from_messages([("user", "{question}")]))
    fake_experiment.run()

    data = export(fake_experiment, tmp_path / "experiment.json")

    reloaded = rup.experiment_from_dict(data)
    assert reloaded.get_answers() == fake_experiment.get_answers()


@pytest.mark.xfail(
    strict=True,
    reason="export_to_file dumps the live models in runnable_models too: a real chat model "
    "(here ChatOpenAI with its SecretStr key) makes json.dump fail with TypeError",
)
def test_experiment_with_a_real_chat_model_can_be_exported_without_leaking_its_key(
    fake_experiment, tmp_path
):
    fake_experiment.add_model(
        ChatOpenAI(model="gpt-4o-mini", api_key="sk-secret"), identifier="gpt"
    )
    path = tmp_path / "experiment.json"

    fake_experiment.export_to_file(str(path))

    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["models"]["gpt"]["type"] == "langchain"
    assert "sk-secret" not in text


@pytest.mark.xfail(
    strict=True,
    reason="export_to_file includes runnable_prompt, runnable_models, runnable_parser, id and "
    "type in the file because the mixin's model_dump is shadowed by pydantic's",
)
def test_export_contains_no_runtime_state(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    assert not {"runnable_prompt", "runnable_models", "runnable_parser"} & set(data)


@pytest.mark.xfail(
    strict=True,
    reason="answers loaded from an exported file are a plain dict: running the reloaded "
    "experiment raises KeyError in InstructionItem.update_answer after the first model call",
)
def test_reloaded_experiment_can_be_run_again(rich_experiment, tmp_path):
    reloaded = rup.experiment_from_dict(export(rich_experiment, tmp_path / "experiment.json"))
    reloaded.clear_models()
    reloaded.add_model(FakeListLLM(responses=["new"]), identifier="m3")

    reloaded.run()

    df = reloaded.get_answers_as_dataframe()
    assert len(df) == 2 * 2 * 2 * 4 + 1 * 2 * 2 * 4
    assert set(df[df["Model ID"] == "m3"]["Answer"]) == {"new"}


@pytest.mark.xfail(
    strict=True,
    reason="export_to_file opens the target for writing before serialising: data that json "
    "cannot serialise raises TypeError after the previous export has been truncated",
)
def test_failed_export_keeps_the_previous_file(fake_experiment, tmp_path):
    path = tmp_path / "experiment.json"
    fake_experiment.export_to_file(str(path))
    previous = path.read_text(encoding="utf-8")
    fake_experiment.parameters.unserialisable = object()  # extra parameters are exported as is

    try:
        fake_experiment.export_to_file(str(path))
    except TypeError:
        pass

    assert path.read_text(encoding="utf-8") == previous


@pytest.mark.xfail(
    strict=True,
    reason="export_to_file opens the file without encoding='utf-8' (the reader uses UTF-8): on "
    "platforms whose default encoding is not UTF-8 (Windows) non-ASCII text raises "
    "UnicodeEncodeError, which is not caught",
)
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


def test_add_model_config_in_the_export_is_a_langchain_config(fake_experiment, tmp_path):
    data = export(fake_experiment, tmp_path / "experiment.json")

    config = LangChainModelConfig(**data["models"]["fake"])
    assert config.definition["name"] == "FakeListLLM"
