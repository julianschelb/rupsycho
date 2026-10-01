"""Tests for the persona, model and prompt management mixins of ``ExperimentDocument``."""

import copy
import json
from typing import Any

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.load import dumpd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import Field

import rupsycho as rup
from rupsycho.models.model import LangChainModelConfig, LocalHuggingFaceModelConfig
from rupsycho.models.prompt import (
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
    NormalPromptTemplateConfig,
)
from rupsycho.models.questionnaire import DemographicProfile


@pytest.fixture(autouse=True)
def _quiet_progress_bars(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")


@pytest.fixture
def bare_experiment(config_dict):
    """The BFI experiment (2 personas, 4 items) without any model."""
    return rup.experiment_from_dict(copy.deepcopy(config_dict))


class RecordingLLM(FakeListLLM):
    """A fake LLM that records the prompt of every call."""

    log: Any = Field(default_factory=list)

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.log.append(prompt)
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


def make_persona(name="Zed", template="{name} the tester"):
    return DemographicProfile(attributes={"name": name}, template=template)


# ------------------------------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------------------------------


def test_personas_of_a_loaded_experiment(bare_experiment):
    assert bare_experiment.list_personas() == ["Optimistic Persona", "Conservative Persona"]
    persona = bare_experiment.get_persona("Conservative Persona")
    assert isinstance(persona, DemographicProfile)
    assert persona.attributes.name == "Grueber"


def test_add_persona_with_identifier(bare_experiment):
    persona = make_persona()

    bare_experiment.add_persona(persona, identifier="zed")

    assert bare_experiment.get_persona("zed") is persona
    assert bare_experiment.list_personas() == ["Optimistic Persona", "Conservative Persona", "zed"]
    assert bare_experiment.demographic_profiles["zed"] is persona


@pytest.mark.parametrize("identifier", [None, ""], ids=["none", "empty"])
def test_add_persona_without_identifier_uses_the_object_id(bare_experiment, identifier):
    persona = make_persona()

    bare_experiment.add_persona(persona, identifier=identifier)

    assert bare_experiment.list_personas()[-1] == str(id(persona))
    assert bare_experiment.get_persona(str(id(persona))) is persona


def test_add_persona_with_duplicate_identifier_warns_and_keeps_the_original(bare_experiment):
    original = bare_experiment.get_persona("Optimistic Persona")

    with pytest.warns(
        UserWarning, match="A persona with the identifier 'Optimistic Persona' already exists."
    ):
        bare_experiment.add_persona(make_persona(), identifier="Optimistic Persona")

    assert bare_experiment.get_persona("Optimistic Persona") is original
    assert len(bare_experiment.list_personas()) == 2


def test_get_persona_of_an_unknown_identifier_is_none(bare_experiment):
    assert bare_experiment.get_persona("nobody") is None


def test_remove_persona(bare_experiment):
    bare_experiment.remove_persona("Optimistic Persona")

    assert bare_experiment.list_personas() == ["Conservative Persona"]
    assert bare_experiment.get_persona("Optimistic Persona") is None


def test_remove_unknown_persona_warns(bare_experiment):
    with pytest.warns(UserWarning, match="No persona found with the identifier 'nobody'."):
        bare_experiment.remove_persona("nobody")

    assert len(bare_experiment.list_personas()) == 2


def test_clear_personas(bare_experiment):
    bare_experiment.clear_personas()

    assert bare_experiment.list_personas() == []
    assert bare_experiment.demographic_profiles == {}


def test_added_persona_takes_part_in_the_run(bare_experiment):
    llm = RecordingLLM(responses=["3"])
    bare_experiment.add_model(llm, identifier="m")
    bare_experiment.add_persona(make_persona(), identifier="zed")

    bare_experiment.run()

    assert len(llm.log) == 4 * 3  # 4 items x (2 configured + 1 added persona)
    assert sum("Zed the tester" in prompt for prompt in llm.log) == 4
    df = bare_experiment.get_answers_as_dataframe()
    assert set(df["Persona ID"]) == {"Optimistic Persona", "Conservative Persona", "zed"}
    assert len(df[df["Persona ID"] == "zed"]) == 4


def test_removed_persona_no_longer_takes_part_in_the_run(bare_experiment):
    llm = RecordingLLM(responses=["3"])
    bare_experiment.add_model(llm, identifier="m")
    bare_experiment.remove_persona("Conservative Persona")

    bare_experiment.run()

    assert len(llm.log) == 4
    assert set(bare_experiment.get_answers_as_dataframe()["Persona ID"]) == {"Optimistic Persona"}


@pytest.mark.parametrize(
    "action",
    [
        lambda e: e.add_persona(make_persona(), identifier="Optimistic Persona"),
        lambda e: e.remove_persona("nobody"),
        lambda e: (
            e.add_model(FakeListLLM(responses=["x"]), identifier="m")
            or e.add_model(FakeListLLM(responses=["x"]), identifier="m")
        ),
        lambda e: e.remove_model("nobody"),
        lambda e: e.replace_model("nobody", FakeListLLM(responses=["x"])),
    ],
    ids=[
        "add_persona",
        "remove_persona",
        "add_model",
        "remove_model",
        "replace_model",
    ],
)
def test_warnings_point_at_the_calling_code(bare_experiment, action):
    """The warnings carry ``stacklevel=2``: the reported location is the caller, not the mixin."""
    with pytest.warns(UserWarning, match="identifier") as record:
        action(bare_experiment)

    ours = [w for w in record if "identifier" in str(w.message)]
    assert ours
    assert {w.filename for w in ours} == {__file__}


# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------


def test_a_fresh_experiment_without_models(bare_experiment):
    assert bare_experiment.list_models() == []
    assert bare_experiment.count_models() == 0
    assert bare_experiment.get_all_runnable_models() == {}
    assert bare_experiment.has_model("anything") is False
    assert bare_experiment.get_model("anything") is None


def test_add_model_registers_a_serialised_config_and_the_runnable(bare_experiment):
    llm = FakeListLLM(responses=["3"])

    bare_experiment.add_model(llm, identifier="fake")

    config = bare_experiment.models["fake"]
    assert isinstance(config, LangChainModelConfig)
    assert config.type == "langchain"
    assert config.definition == dumpd(llm)
    assert config.parameters == {}
    assert bare_experiment.get_model("fake") is llm
    assert bare_experiment.runnable_models == {"fake": llm}
    assert bare_experiment.list_models() == ["fake"]
    assert bare_experiment.has_model("fake") is True
    assert bare_experiment.has_model("other") is False
    assert bare_experiment.count_models() == 1


@pytest.mark.parametrize("identifier", [None, ""], ids=["none", "empty"])
def test_add_model_without_identifier_uses_the_object_id(bare_experiment, identifier):
    llm = FakeListLLM(responses=["3"])

    bare_experiment.add_model(llm, identifier=identifier)

    assert bare_experiment.list_models() == [str(id(llm))]
    assert bare_experiment.get_model(str(id(llm))) is llm


def test_models_keep_their_insertion_order(bare_experiment):
    for name in ("c", "a", "b"):
        bare_experiment.add_model(FakeListLLM(responses=[name]), identifier=name)

    assert bare_experiment.list_models() == ["c", "a", "b"]
    assert list(bare_experiment.get_all_runnable_models()) == ["c", "a", "b"]
    assert bare_experiment.count_models() == 3


def test_add_model_with_duplicate_identifier_warns_and_refreshes_the_runnable(bare_experiment):
    """The documented way to run an experiment a second time is to add its models again."""
    first, second = FakeListLLM(responses=["1"]), FakeListLLM(responses=["2"])
    bare_experiment.add_model(first, identifier="m")

    with pytest.warns(UserWarning, match="A model with the identifier 'm' already exists."):
        bare_experiment.add_model(second, identifier="m")

    assert bare_experiment.count_models() == 1
    assert bare_experiment.get_model("m") is second


def test_get_all_runnable_models_returns_the_live_dictionary(bare_experiment):
    llm = FakeListLLM(responses=["3"])
    bare_experiment.add_model(llm, identifier="fake")

    assert bare_experiment.get_all_runnable_models() is bare_experiment.runnable_models


def test_remove_model_removes_config_and_runnable(bare_experiment):
    bare_experiment.add_model(FakeListLLM(responses=["1"]), identifier="a")
    bare_experiment.add_model(FakeListLLM(responses=["2"]), identifier="b")

    bare_experiment.remove_model("a")

    assert bare_experiment.list_models() == ["b"]
    assert list(bare_experiment.runnable_models) == ["b"]
    assert bare_experiment.has_model("a") is False
    assert bare_experiment.get_model("a") is None


def test_remove_unknown_model_warns(bare_experiment):
    bare_experiment.add_model(FakeListLLM(responses=["1"]), identifier="a")

    with pytest.warns(UserWarning, match="No model found with the identifier 'nobody'."):
        bare_experiment.remove_model("nobody")

    assert bare_experiment.list_models() == ["a"]


def test_remove_model_works_when_the_runnable_is_already_released(bare_experiment):
    bare_experiment.add_model(FakeListLLM(responses=["1"]), identifier="a")
    del bare_experiment.runnable_models["a"]

    bare_experiment.remove_model("a")

    assert bare_experiment.list_models() == []


def test_clear_models(bare_experiment):
    bare_experiment.add_model(FakeListLLM(responses=["1"]), identifier="a")
    bare_experiment.add_model(FakeListLLM(responses=["2"]), identifier="b")

    bare_experiment.clear_models()

    assert bare_experiment.list_models() == []
    assert bare_experiment.count_models() == 0
    assert bare_experiment.get_all_runnable_models() == {}


def test_models_from_the_configuration_are_listed_and_counted(config_dict):
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "lh": {"type": "local_huggingface", "name_or_path": "org/model"},
        "oa": {"type": "openai", "name_or_path": "gpt-4o-mini"},
    }
    experiment = rup.experiment_from_dict(config)

    assert experiment.list_models() == ["lh", "oa"]
    assert experiment.count_models() == 2
    assert experiment.has_model("oa") is True
    # with lazy loading the "runnable" entry is still the configuration
    assert isinstance(experiment.get_model("lh"), LocalHuggingFaceModelConfig)


# ---- load_model / set_runnable_models / replace_model -----------------------------------------


def test_load_model_deserialises_a_definition(bare_experiment):
    definition = dumpd(StrOutputParser())

    loaded = bare_experiment.load_model(definition)

    assert isinstance(loaded, StrOutputParser)


@pytest.mark.parametrize(
    "definition",
    [
        dumpd(FakeListLLM(responses=["x"])),  # a "not_implemented" stub: fakes cannot be loaded
        {"lc": 1, "type": "constructor", "id": ["no_such_package", "Thing"], "kwargs": {}},
        {"lc": 1, "type": "constructor", "id": ["langchain_core", "no_such_module", "Thing"]},
    ],
    ids=["not-serialisable", "unknown-namespace", "unknown-module"],
)
def test_load_model_of_a_bad_definition_returns_none_and_warns(bare_experiment, definition):
    with pytest.warns(UserWarning, match="Failed to load model"):
        assert bare_experiment.load_model(definition) is None


def test_set_runnable_models_reloads_every_model_from_its_definition(bare_experiment):
    bare_experiment.add_model(RunnablePassthrough(), identifier="a")
    bare_experiment.add_model(StrOutputParser(), identifier="b")
    original = bare_experiment.get_model("a")

    bare_experiment.set_runnable_models()

    assert list(bare_experiment.runnable_models) == ["a", "b"]
    assert isinstance(bare_experiment.get_model("a"), RunnablePassthrough)
    assert isinstance(bare_experiment.get_model("b"), StrOutputParser)
    assert bare_experiment.get_model("a") is not original  # a fresh object from the definition


def test_set_runnable_models_drops_models_that_cannot_be_loaded(bare_experiment):
    bare_experiment.add_model(StrOutputParser(), identifier="loadable")
    bare_experiment.add_model(FakeListLLM(responses=["x"]), identifier="fake")

    with pytest.warns(UserWarning, match="Failed to load"):
        bare_experiment.set_runnable_models()

    assert list(bare_experiment.runnable_models) == ["loadable"]
    assert bare_experiment.list_models() == ["loadable", "fake"]  # the configs are kept


def test_replace_model_swaps_the_definition_and_refreshes_the_runnables(bare_experiment):
    bare_experiment.add_model(RunnablePassthrough(), identifier="a")
    bare_experiment.add_model(RunnablePassthrough(), identifier="untouched")

    bare_experiment.replace_model("a", StrOutputParser())

    assert bare_experiment.models["a"].definition == dumpd(StrOutputParser())
    assert isinstance(bare_experiment.get_model("a"), StrOutputParser)
    assert isinstance(bare_experiment.get_model("untouched"), RunnablePassthrough)
    assert bare_experiment.list_models() == ["a", "untouched"]


def test_replace_unknown_model_warns_and_changes_nothing(bare_experiment):
    llm = FakeListLLM(responses=["1"])
    bare_experiment.add_model(llm, identifier="a")

    with pytest.warns(UserWarning, match="No model found with the identifier 'nobody'."):
        bare_experiment.replace_model("nobody", FakeListLLM(responses=["2"]))

    assert bare_experiment.list_models() == ["a"]
    assert bare_experiment.get_model("a") is llm


@pytest.mark.xfail(
    strict=True,
    reason="replace_model reloads every model from its serialised definition: a model that "
    "LangChain cannot serialise (fakes, local pipelines) is dropped, together with the others",
)
def test_replace_model_with_a_model_that_cannot_be_serialised(bare_experiment):
    kept = FakeListLLM(responses=["kept"])
    old, new = FakeListLLM(responses=["old"]), FakeListLLM(responses=["new"])
    bare_experiment.add_model(kept, identifier="kept")
    bare_experiment.add_model(old, identifier="a")

    bare_experiment.replace_model("a", new)

    assert bare_experiment.get_model("a") is new
    assert bare_experiment.get_model("kept") is kept


@pytest.mark.xfail(
    strict=True,
    reason="set_runnable_models reads .definition of every config: AttributeError for configs "
    "from the configuration file (local_huggingface, openai, ...)",
)
def test_set_runnable_models_loads_configured_models(config_dict, monkeypatch):
    fake = FakeListLLM(responses=["x"])
    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", lambda self: fake)
    config = copy.deepcopy(config_dict)
    config["models"] = {"lh": {"type": "local_huggingface", "name_or_path": "org/model"}}
    experiment = rup.experiment_from_dict(config)

    experiment.set_runnable_models()

    assert experiment.get_model("lh") is fake


# ------------------------------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------------------------------

OPTIONS = (
    "1. Disagree strongly, 2. Disagree a little, 3. Neither agree nor disagree, "
    "4. Agree a little, 5. Agree strongly"
)
PERSONA = "Ms Muller is 18 years old and very open minded with a optimistic personality."
QUESTION = "I see myself as someone who..."

CHAT_PROMPT = {
    "type": "chat",
    "messages": [
        {"role": "system", "content": "SYS {general_instruction}"},
        {"role": "user", "content": "{persona_description} | {question} | {answer_options}"},
    ],
}
NORMAL_PROMPT = {
    "type": "normal",
    "template": "{general_instruction} // {persona_description} | {question} | {answer_options}",
}
LANGCHAIN_PROMPT = {
    "type": "langchain",
    "definition": dumpd(
        ChatPromptTemplate.from_messages(
            [("system", "LC {general_instruction}"), ("user", "{question} | {answer_options}")]
        )
    ),
}


@pytest.mark.parametrize(
    "prompt_config, config_class, runnable_class, expected_prompt",
    [
        (
            CHAT_PROMPT,
            ChatPromptTemplateConfig,
            ChatPromptTemplate,
            f"System: SYS {{general}}\nHuman: {PERSONA} | {QUESTION} | {OPTIONS}",
        ),
        (
            NORMAL_PROMPT,
            NormalPromptTemplateConfig,
            PromptTemplate,
            f"{{general}} // {PERSONA} | {QUESTION} | {OPTIONS}",
        ),
        (
            LANGCHAIN_PROMPT,
            LangchainPromptTemplateConfig,
            ChatPromptTemplate,
            f"System: LC {{general}}\nHuman: {QUESTION} | {OPTIONS}",
        ),
    ],
    ids=["chat", "normal", "langchain"],
)
def test_prompt_of_each_configuration_type_is_loaded_and_used(
    config_dict, prompt_config, config_class, runnable_class, expected_prompt
):
    config = copy.deepcopy(config_dict)
    config["prompt_template"] = copy.deepcopy(prompt_config)
    experiment = rup.experiment_from_dict(config)
    llm = RecordingLLM(responses=["3"])
    experiment.add_model(llm, identifier="m")

    assert experiment.has_prompt() is True
    assert isinstance(experiment.get_prompt_config(), config_class)
    assert type(experiment.get_prompt()) is runnable_class

    experiment.run()

    general = experiment.questionnaire.general_instruction
    assert llm.log[0] == expected_prompt.replace("{general}", general)
    assert len(llm.log) == 8


def test_loaded_experiment_has_the_configured_chat_prompt(bare_experiment):
    assert bare_experiment.has_prompt() is True
    prompt = bare_experiment.get_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == [
        "answer_options",
        "general_instruction",
        "persona_description",
        "question",
    ]
    assert isinstance(bare_experiment.get_prompt_config(), ChatPromptTemplateConfig)


def test_set_prompt_replaces_the_runnable_prompt(bare_experiment):
    prompt = ChatPromptTemplate.from_messages([("user", "{question} / {answer_options}")])

    bare_experiment.set_prompt(prompt)

    assert bare_experiment.get_prompt() is prompt
    assert bare_experiment.has_prompt() is True
    # a configuration (or its serialised form), not the runnable prompt itself
    assert isinstance(
        bare_experiment.get_prompt_config(),
        dict
        | NormalPromptTemplateConfig
        | ChatPromptTemplateConfig
        | LangchainPromptTemplateConfig,
    )


def test_set_prompt_is_used_by_the_run(bare_experiment):
    llm = RecordingLLM(responses=["3"])
    bare_experiment.add_model(llm, identifier="m")
    bare_experiment.set_prompt(
        ChatPromptTemplate.from_messages([("user", "Q: {question} A: {answer_options}")])
    )

    bare_experiment.run()

    assert llm.log[0] == f"Human: Q: {QUESTION} A: {OPTIONS}"


@pytest.mark.xfail(
    strict=True,
    reason="set_prompt stores the raw langchain dumpd() dict in prompt_template, which is typed "
    "as one of the prompt config classes (and cannot be read back by experiment_from_dict)",
)
def test_set_prompt_stores_a_valid_prompt_configuration(bare_experiment):
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])

    bare_experiment.set_prompt(prompt)

    config = bare_experiment.get_prompt_config()
    assert isinstance(
        config,
        NormalPromptTemplateConfig | ChatPromptTemplateConfig | LangchainPromptTemplateConfig,
    )
    assert config.load_prompt_template() == prompt


def test_reset_prompt_clears_both_forms(bare_experiment):
    bare_experiment.reset_prompt()

    assert bare_experiment.get_prompt() is None
    assert bare_experiment.get_prompt_config() is None
    assert bare_experiment.has_prompt() is False


def test_reset_prompt_makes_the_run_fail_until_a_prompt_is_set_again(bare_experiment):
    bare_experiment.add_model(FakeListLLM(responses=["3"]), identifier="m")
    bare_experiment.reset_prompt()
    with pytest.raises(ValueError, match="runnable_prompt has not been set."):
        bare_experiment.run()

    bare_experiment.set_prompt(ChatPromptTemplate.from_messages([("user", "{question}")]))
    bare_experiment.run()

    assert len(bare_experiment.get_answers_as_dataframe()) == 8


def test_load_prompt_deserialises_chat_and_normal_prompts(bare_experiment):
    chat = ChatPromptTemplate.from_messages([("system", "{a}"), ("user", "{b}")])
    normal = PromptTemplate.from_template("hello {name}")

    assert bare_experiment.load_prompt(dumpd(chat)) == chat
    assert bare_experiment.load_prompt(dumpd(normal)) == normal


@pytest.mark.parametrize(
    "definition",
    [
        dumpd(FakeListLLM(responses=["x"])),
        {"lc": 1, "type": "constructor", "id": ["no_such_package", "Thing"], "kwargs": {}},
    ],
    ids=["not-serialisable", "unknown-namespace"],
)
def test_load_prompt_of_a_bad_definition_returns_none_and_warns(bare_experiment, definition):
    with pytest.warns(UserWarning, match="Failed to load prompt template"):
        assert bare_experiment.load_prompt(definition) is None


def test_set_prompt_config_loads_a_serialised_prompt(bare_experiment):
    prompt = PromptTemplate.from_template("Q: {question}")

    bare_experiment.set_prompt_config(dumpd(prompt))

    assert bare_experiment.get_prompt() == prompt
    assert bare_experiment.has_prompt() is True


def test_set_prompt_config_with_a_bad_definition_leaves_no_runnable_prompt(bare_experiment):
    definition = dumpd(FakeListLLM(responses=["x"]))

    with pytest.warns(UserWarning, match="Failed to (load|create)"):
        bare_experiment.set_prompt_config(definition)

    assert bare_experiment.get_prompt() is None
    assert bare_experiment.has_prompt() is False


def test_set_prompt_config_round_trips_the_serialised_form(bare_experiment):
    bare_experiment.set_prompt(ChatPromptTemplate.from_messages([("user", "{question}")]))
    serialised = bare_experiment.get_prompt_config()
    bare_experiment.reset_prompt()

    bare_experiment.set_prompt_config(serialised)

    assert bare_experiment.get_prompt() == ChatPromptTemplate.from_messages(
        [("user", "{question}")]
    )


@pytest.mark.xfail(
    strict=True,
    reason="set_prompt_config(get_prompt_config()) does not round-trip for a prompt that comes "
    "from the configuration: the config object is stored as runnable_prompt and run() fails",
)
@pytest.mark.parametrize("as_dict", [False, True], ids=["config-object", "config-dict"])
def test_set_prompt_config_accepts_the_configuration_of_a_loaded_prompt(bare_experiment, as_dict):
    bare_experiment.add_model(FakeListLLM(responses=["3"]), identifier="m")
    config = bare_experiment.get_prompt_config()

    bare_experiment.set_prompt_config(config.model_dump() if as_dict else config)

    assert isinstance(bare_experiment.get_prompt(), ChatPromptTemplate)
    bare_experiment.run()
    assert len(bare_experiment.get_answers_as_dataframe()) == 8


def test_prompt_template_that_fails_to_load_gives_no_runnable_prompt(config_dict):
    config = copy.deepcopy(config_dict)
    config["prompt_template"] = {
        "type": "chat",
        "messages": [{"role": "no-such-role", "content": "{question}"}],
    }

    with pytest.warns(UserWarning, match="Failed to load prompt template"):
        experiment = rup.experiment_from_dict(config)

    assert experiment.has_prompt() is False
    assert isinstance(experiment.get_prompt_config(), ChatPromptTemplateConfig)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="m")
    with pytest.raises(ValueError, match="runnable_prompt has not been set."):
        experiment.run()


def test_experiment_without_prompt_template_uses_the_default_chat_prompt(config_dict):
    config = copy.deepcopy(config_dict)
    del config["prompt_template"]

    experiment = rup.experiment_from_dict(config)

    prompt = experiment.get_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert [type(m).__name__ for m in prompt.messages] == [
        "SystemMessagePromptTemplate",
        "HumanMessagePromptTemplate",
    ]
    assert experiment.get_prompt_config().type == "chat"


def test_json_round_trip_of_the_prompt_configuration(bare_experiment):
    """The prompt configuration of a loaded experiment is plain JSON data."""
    config = bare_experiment.get_prompt_config()

    assert json.loads(config.model_dump_json()) == config.model_dump()
