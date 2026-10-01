"""Tests for the data models: questionnaire, profiles, prompt/model configs and parameters."""

import copy
import json

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import ValidationError
from tabulate import tabulate

import rupsycho as rup
import rupsycho.models.model as model_module
from rupsycho import prompts
from rupsycho.experiment import ExperimentDocument
from rupsycho.models.model import (
    DEFAULT_MODEL_CONFIG,
    DeepSeekModelConfig,
    GoogleModelConfig,
    LangChainModelConfig,
    LocalHuggingFaceModelConfig,
    OllamaModelConfig,
    OpenAIModelConfig,
    RemoteHuggingFaceModelConfig,
)
from rupsycho.models.parameters import ExperimentParameters
from rupsycho.models.prompt import (
    ChatMessageConfig,
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
    NormalPromptTemplateConfig,
)
from rupsycho.models.questionnaire import (
    AnswerOption,
    AnswerOptions,
    DemographicAttributes,
    DemographicProfile,
    InstructionItem,
    Questionnaire,
)

# ------------------------------------------------------------------------------------------
# DemographicAttributes / DemographicProfile
# ------------------------------------------------------------------------------------------


def test_demographic_attributes_default_to_none_and_allow_extras():
    attributes = DemographicAttributes()
    assert (attributes.age, attributes.title, attributes.name) == (None, None, None)

    extended = DemographicAttributes(name="Eva", city="Berlin", children=2)
    assert extended.city == "Berlin"
    assert extended.model_dump() == {
        "age": None,
        "title": None,
        "name": "Eva",
        "city": "Berlin",
        "children": 2,
    }


def test_profile_renders_the_default_template():
    profile = DemographicProfile(attributes={"title": "Dr", "name": "Who", "age": 900})

    assert str(profile) == "Dr Who is 900 years old."
    assert profile.get_profile_desc() == str(profile)
    assert profile.template == "{title} {name} is {age} years old."


def test_profile_without_attributes_renders_none():
    assert str(DemographicProfile(attributes={})) == "None None is None years old."


def test_profile_template_can_use_extra_attributes_and_format_specs():
    profile = DemographicProfile(
        attributes={"name": "Eva", "age": 7, "city": "Berlin", "job": "tester"},
        template="{name} ({age:03d}) is a {job} from {city}",
    )

    assert profile.get_profile_desc() == "Eva (007) is a tester from Berlin"
    assert profile.attributes.city == "Berlin"
    assert isinstance(profile.attributes, DemographicAttributes)


def test_profile_template_with_unknown_placeholder_raises_key_error():
    """docs/configuration.md: an unknown placeholder raises a KeyError when prompts are built."""
    profile = DemographicProfile(attributes={"name": "Eva"}, template="{name} from {nowhere}")
    with pytest.raises(KeyError, match="nowhere"):
        profile.get_profile_desc()


def test_profile_allows_extra_fields():
    profile = DemographicProfile(attributes={}, group="control")
    assert profile.group == "control"
    assert profile.model_dump()["group"] == "control"


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"attributes": "not a mapping"}, {"attributes": {}, "template": 5}],
    ids=["missing-attributes", "attributes-not-a-mapping", "template-not-a-string"],
)
def test_invalid_profile_raises_validation_error(kwargs):
    with pytest.raises(ValidationError):
        DemographicProfile(**kwargs)


def test_profile_round_trips_through_model_dump():
    profile = DemographicProfile(attributes={"name": "Eva", "city": "Berlin"}, template="{name}")
    assert DemographicProfile(**profile.model_dump()) == profile


# ------------------------------------------------------------------------------------------
# AnswerOption / AnswerOptions
# ------------------------------------------------------------------------------------------


def test_answer_option_defaults():
    option = AnswerOption()
    assert (option.text, option.ignored_for_scale, option.weight) == ("Choose an option", False, 0)


def test_answer_option_coerces_numeric_strings_but_rejects_garbage():
    assert AnswerOption(weight="2").weight == 2
    for bad in ({"weight": "abc"}, {"weight": 2.5}, {"text": 5}, {"ignored_for_scale": "maybe"}):
        with pytest.raises(ValidationError):
            AnswerOption(**bad)


@pytest.mark.parametrize(
    "delimiter, prepend, expected",
    [
        (", ", False, "A, B, C"),
        (", ", True, ", A, B, C"),
        ("\n", False, "A\nB\nC"),
        ("\n", True, "\nA\nB\nC"),
        (" | ", False, "A | B | C"),
        (" | ", True, " | A | B | C"),
        ("", False, "ABC"),
        ("", True, "ABC"),
    ],
)
def test_join_options_uses_delimiter_and_prepend_delimiter(delimiter, prepend, expected):
    options = AnswerOptions(
        options={key: AnswerOption(text=key.upper()) for key in "abc"},
        delimiter=delimiter,
        prepend_delimiter=prepend,
    )
    assert options.join_options() == expected


def test_answer_options_defaults():
    options = AnswerOptions()
    assert options.options == {}
    assert options.delimiter == ", "
    assert options.prepend_delimiter is False


@pytest.mark.parametrize("prepend", [False, True])
def test_join_options_of_no_options_is_empty(prepend):
    assert AnswerOptions(prepend_delimiter=prepend).join_options() == ""
    assert AnswerOptions(prepend_delimiter=prepend).get_options_as_list() == []


def test_options_keep_their_order_in_join_and_list():
    options = AnswerOptions(
        options={
            "2": AnswerOption(text="second"),
            "1": AnswerOption(text="first"),
            "10": AnswerOption(text="tenth"),
        }
    )
    assert options.get_options_as_list() == ["second", "first", "tenth"]
    assert options.join_options() == "second, first, tenth"


# ------------------------------------------------------------------------------------------
# InstructionItem
# ------------------------------------------------------------------------------------------


def test_instruction_item_defaults():
    item = InstructionItem()
    assert item.question == "Enter question text here"
    assert item.reversed is False
    assert item.answer_options is None
    assert item.attributes == {}
    assert item.get_all_answers() == {}
    assert item.get_answer_options_as_list() == []


def test_answers_default_is_not_shared_between_items():
    first, second = InstructionItem(), InstructionItem()
    first.update_answer("m", "p", "1", "x")
    assert second.get_all_answers() == {}


def test_update_and_get_answer():
    item = InstructionItem(question="q")

    item.update_answer("m1", "p1", "7", "three")
    item.update_answer("m1", "p1", "8", "four")
    item.update_answer("m1", "p2", "7", "five")
    item.update_answer("m2", "p1", "7", "six")

    assert item.get_answer("m1", "p1", "7") == "three"
    assert item.get_answer("m1", "p1", "8") == "four"
    assert item.get_answer("m1", "p2", "7") == "five"
    assert item.get_answer("m2", "p1", "7") == "six"
    assert item.get_all_answers() == {
        "m1": {"p1": {"7": "three", "8": "four"}, "p2": {"7": "five"}},
        "m2": {"p1": {"7": "six"}},
    }
    assert item.model_dump()["answers"] == item.get_all_answers()


def test_update_answer_overwrites_the_same_slot():
    item = InstructionItem()
    item.update_answer("m", "p", "1", "old")
    item.update_answer("m", "p", "1", "new")
    assert item.get_all_answers() == {"m": {"p": {"1": "new"}}}


def test_update_answer_ignores_none():
    item = InstructionItem()
    item.update_answer("m", "p", "1", None)
    assert item.get_all_answers() == {}


@pytest.mark.parametrize("answer", ["", "0", " "])
def test_update_answer_stores_falsy_strings(answer):
    item = InstructionItem()
    item.update_answer("m", "p", "1", answer)
    assert item.get_answer("m", "p", "1") == answer


def test_answer_keys_can_be_ints():
    """Without configured seeds the run uses the integer default seed."""
    item = InstructionItem()
    item.update_answer("m", "p", 42, "x")
    assert item.get_answer("m", "p", 42) == "x"
    assert item.get_all_answers() == {"m": {"p": {42: "x"}}}


def test_update_answer_works_on_loaded_answers_when_model_and_persona_exist():
    item = InstructionItem(question="q", answers={"m": {"p": {"1": "old"}}})

    item.update_answer("m", "p", "2", "new")

    assert item.get_all_answers() == {"m": {"p": {"1": "old", "2": "new"}}}


@pytest.mark.xfail(
    strict=True,
    reason="answers loaded from a dict (e.g. from an exported file) are a plain dict, so "
    "update_answer raises KeyError for a new model/persona instead of creating the entry",
)
@pytest.mark.parametrize(
    "existing, slot",
    [
        ({}, ("m", "p", "1")),
        ({"m": {}}, ("m", "p", "1")),
        ({"m": {"p": {"1": "old"}}}, ("other", "p", "1")),
    ],
    ids=["empty", "known-model", "new-model"],
)
def test_update_answer_works_on_answers_loaded_from_a_dict(existing, slot):
    item = InstructionItem(question="q", answers=copy.deepcopy(existing))

    item.update_answer(*slot, "new")

    assert item.get_answer(*slot) == "new"


@pytest.mark.xfail(
    strict=True,
    reason="get_answer on a slot without answer returns {} and inserts an empty entry, which "
    "then shows up in get_all_answers / model_dump / the data frame",
)
def test_reading_a_missing_answer_does_not_change_the_item():
    item = InstructionItem(question="q")
    item.update_answer("m", "p", "1", "x")

    try:
        item.get_answer("m", "p", "2")
    except KeyError:
        pass

    assert item.get_all_answers() == {"m": {"p": {"1": "x"}}}


def test_instruction_item_answers_survive_a_dump_and_validate_round_trip():
    item = InstructionItem(question="q", attributes={"dimension": "1"})
    item.update_answer("m", "p", "1", "x")

    clone = InstructionItem(**item.model_dump())

    assert clone.question == "q"
    assert clone.attributes == {"dimension": "1"}
    assert clone.get_all_answers() == {"m": {"p": {"1": "x"}}}


# ---- answer options conversion (shared by items and the questionnaire) ------------------------

FLAT = {"1": {"text": "A", "weight": 1}, "2": {"text": "B", "ignored_for_scale": True}}
WRAPPED = {
    "options": {"1": {"text": "A", "weight": 1}, "2": {"text": "B", "ignored_for_scale": True}},
    "delimiter": "\n",
    "prepend_delimiter": True,
}


def build_with_options(owner, value):
    """Build an item or a questionnaire with ``value`` as (default) answer options."""
    if owner == "item":
        return InstructionItem(question="q", answer_options=value).answer_options
    return Questionnaire(
        name="n", general_instruction="g", default_answer_options=value
    ).default_answer_options


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_flat_dict_of_options_becomes_answer_options_with_default_delimiter(owner):
    options = build_with_options(owner, copy.deepcopy(FLAT))

    assert isinstance(options, AnswerOptions)
    assert options.options == {
        "1": AnswerOption(text="A", weight=1),
        "2": AnswerOption(text="B", ignored_for_scale=True),
    }
    assert options.delimiter == ", "
    assert options.prepend_delimiter is False
    assert options.join_options() == "A, B"
    assert options.get_options_as_list() == ["A", "B"]


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_options_wrapper_keeps_delimiter_and_prepend_delimiter(owner):
    options = build_with_options(owner, copy.deepcopy(WRAPPED))

    assert isinstance(options, AnswerOptions)
    assert list(options.options) == ["1", "2"]
    assert options.delimiter == "\n"
    assert options.prepend_delimiter is True
    assert options.join_options() == "\nA\nB"


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_options_wrapper_without_delimiter_uses_the_defaults(owner):
    options = build_with_options(owner, {"options": copy.deepcopy(FLAT)})

    assert options.delimiter == ", "
    assert options.prepend_delimiter is False
    assert options.join_options() == "A, B"


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_answer_options_instance_and_none_pass_through(owner):
    instance = AnswerOptions(options={"1": AnswerOption(text="A")}, delimiter=" / ")
    assert build_with_options(owner, instance) == instance
    assert build_with_options(owner, None) is None


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_empty_dict_of_options_gives_empty_answer_options(owner):
    options = build_with_options(owner, {})
    assert isinstance(options, AnswerOptions)
    assert options.options == {}
    assert options.join_options() == ""


@pytest.mark.parametrize("owner", ["item", "questionnaire"])
@pytest.mark.parametrize(
    "bad",
    [
        5,
        "A, B",
        ["A", "B"],
        {"1": {"text": 5}},
        {"1": {"weight": "abc"}},
        {"options": {"1": {"ignored_for_scale": "maybe"}}},
        {"options": {"1": {"text": "A"}}, "delimiter": 5},
        {"options": {"1": {"text": "A"}}, "prepend_delimiter": "sometimes"},
    ],
    ids=[
        "int",
        "str",
        "list",
        "text-not-a-string",
        "weight-not-a-number",
        "wrapped-flag-not-a-bool",
        "delimiter-not-a-string",
        "prepend-not-a-bool",
    ],
)
def test_invalid_answer_options_raise_validation_error(owner, bad):
    with pytest.raises(ValidationError):
        build_with_options(owner, bad)


@pytest.mark.xfail(
    strict=True,
    reason="the options validators call AnswerOption(**option): an option that is an "
    "AnswerOption instance (or any non-mapping) raises a bare TypeError instead of being "
    "accepted / reported as ValidationError",
)
@pytest.mark.parametrize("owner", ["item", "questionnaire"])
def test_options_given_as_answer_option_instances_are_accepted(owner):
    options = build_with_options(owner, {"1": AnswerOption(text="A"), "2": AnswerOption(text="B")})

    assert options.join_options() == "A, B"


@pytest.mark.xfail(
    strict=True,
    reason="a non-mapping option value (e.g. a string) raises a bare TypeError from the "
    "validator instead of a ValidationError",
)
@pytest.mark.parametrize("owner", ["item", "questionnaire"])
@pytest.mark.parametrize("bad", [{"1": "just text"}, {"options": {"1": "just text"}}])
def test_non_mapping_option_raises_validation_error(owner, bad):
    with pytest.raises(ValidationError):
        build_with_options(owner, bad)


def test_get_answer_options_as_list_only_looks_at_the_items_own_options():
    own = InstructionItem(answer_options=copy.deepcopy(FLAT))
    default_only = InstructionItem()
    assert own.get_answer_options_as_list() == ["A", "B"]
    assert default_only.get_answer_options_as_list() == []


# ------------------------------------------------------------------------------------------
# Questionnaire
# ------------------------------------------------------------------------------------------


def test_questionnaire_requires_name_and_general_instruction():
    for kwargs in ({}, {"name": "n"}, {"general_instruction": "g"}):
        with pytest.raises(ValidationError):
            Questionnaire(**kwargs)


def test_questionnaire_defaults():
    questionnaire = Questionnaire(name="n", general_instruction="g")

    assert questionnaire.demographic_profiles is None
    assert questionnaire.attributes == {}
    assert questionnaire.default_answer_options is None
    assert questionnaire.instruction_items is None
    assert questionnaire.get_number_of_questions() == 0


def test_questionnaire_converts_nested_dicts(config_dict):
    questionnaire = Questionnaire(**copy.deepcopy(config_dict["questionnaire"]))

    assert questionnaire.get_number_of_questions() == 4
    assert all(isinstance(item, InstructionItem) for item in questionnaire.instruction_items)
    assert isinstance(questionnaire.default_answer_options, AnswerOptions)
    assert list(questionnaire.default_answer_options.options) == ["1", "2", "3", "4", "5"]
    assert questionnaire.default_answer_options.options["3"] == AnswerOption(
        text="3. Neither agree nor disagree", ignored_for_scale=False, weight=3
    )
    assert isinstance(questionnaire.demographic_profiles[0], DemographicProfile)
    assert questionnaire.attributes["dimension"]["1"] == "Extraversion"
    assert questionnaire.instruction_items[1].attributes == {"dimension": "1"}


def test_get_number_of_questions():
    items = [InstructionItem(question=f"q{i}") for i in range(3)]
    assert (
        Questionnaire(
            name="n", general_instruction="g", instruction_items=items
        ).get_number_of_questions()
        == 3
    )
    assert (
        Questionnaire(
            name="n", general_instruction="g", instruction_items=[]
        ).get_number_of_questions()
        == 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instruction_items": "not a list"},
        {"instruction_items": [{"question": 5}]},
        {"instruction_items": [{"reversed": "perhaps"}]},
        {"demographic_profiles": [{"template": "x"}]},
        {"attributes": ["not", "a", "dict"]},
    ],
    ids=[
        "items-not-a-list",
        "question-not-a-string",
        "reversed-not-a-bool",
        "profile-without-attributes",
        "attributes-not-a-dict",
    ],
)
def test_invalid_questionnaire_raises_validation_error(kwargs):
    with pytest.raises(ValidationError):
        Questionnaire(name="n", general_instruction="g", **kwargs)


def test_questionnaire_round_trips_through_model_dump(config_dict):
    questionnaire = Questionnaire(**copy.deepcopy(config_dict["questionnaire"]))

    clone = Questionnaire(**json.loads(json.dumps(questionnaire.model_dump())))

    assert clone.model_dump() == questionnaire.model_dump()
    assert (
        clone.default_answer_options.join_options()
        == questionnaire.default_answer_options.join_options()
    )


# ---- print_questionnaire ------------------------------------------------------------------------


def test_print_questionnaire_with_only_name_and_instruction(capsys):
    Questionnaire(name="Mini", general_instruction="Rate it.").print_questionnaire()

    assert capsys.readouterr().out == "Name: Mini\n\nGeneral Instruction: Rate it.\n\n"


def test_print_questionnaire_lists_the_demographic_profiles(capsys):
    Questionnaire(
        name="Mini",
        general_instruction="Rate it.",
        demographic_profiles=[
            DemographicProfile(attributes={"name": "Eva", "age": 5}, template="{name} is {age}"),
            DemographicProfile(attributes={"name": "Bob", "title": "Mr"}),
        ],
    ).print_questionnaire()

    assert capsys.readouterr().out == (
        "Name: Mini\n\nGeneral Instruction: Rate it.\n\n"
        "Demographic Profiles:\n- Eva is 5\n- Mr Bob is None years old.\n"
    )


def test_print_questionnaire_lists_items_without_answer_options(capsys):
    Questionnaire(
        name="Mini",
        general_instruction="Rate it.",
        instruction_items=[InstructionItem(question="First?"), InstructionItem(question="Second?")],
    ).print_questionnaire()

    assert capsys.readouterr().out == (
        "Name: Mini\n\nGeneral Instruction: Rate it.\n\n"
        "\nInstruction Items:\n"
        "- Question: First?\n  Answer Options:\n\n"
        "- Question: Second?\n  Answer Options:\n\n"
    )


@pytest.mark.xfail(
    strict=True,
    reason="print_questionnaire calls .items() on AnswerOptions (AttributeError); it must "
    "iterate AnswerOptions.options (a '# type: ignore[attr-defined]' hid the mypy error)",
)
def test_print_questionnaire_shows_the_default_answer_options_table(capsys, config_dict):
    questionnaire = Questionnaire(**copy.deepcopy(config_dict["questionnaire"]))

    questionnaire.print_questionnaire()
    out = capsys.readouterr().out

    table = tabulate(
        [
            ["1", "1. Disagree strongly", False, 1],
            ["2", "2. Disagree a little", False, 2],
            ["3", "3. Neither agree nor disagree", False, 3],
            ["4", "4. Agree a little", False, 4],
            ["5", "5. Agree strongly", False, 5],
        ],
        headers=["ID", "Text", "Ignored for Scale", "Weight"],
    )
    assert out.startswith(
        f"Name: {questionnaire.name}\n\n"
        f"General Instruction: {questionnaire.general_instruction}\n\n"
        "Demographic Profiles:\n- Ms Muller is 18 years old.\n"
        f"\nDefault Answer Options:\n{table}\n"
        "\nInstruction Items:\n- Question: I see myself as someone who...\n  Answer Options:\n\n"
    )
    assert out.count("- Question: ") == 4


@pytest.mark.xfail(
    strict=True,
    reason="print_questionnaire calls .items() on an item's AnswerOptions (AttributeError) and "
    "then indexes the pairs with opt[1]; it must iterate AnswerOptions.options",
)
def test_print_questionnaire_shows_the_answer_options_of_an_item(capsys):
    questionnaire = Questionnaire(
        name="Mini",
        general_instruction="Rate it.",
        instruction_items=[
            InstructionItem(
                question="Own options?",
                answer_options={
                    "1": {"text": "yes", "weight": 2},
                    "2": {"text": "no", "ignored_for_scale": True},
                },
            ),
            InstructionItem(question="No options?"),
        ],
    )

    questionnaire.print_questionnaire()
    out = capsys.readouterr().out

    table = tabulate(
        [["yes", False, 2], ["no", True, 0]],
        headers=["Text", "Ignored for Scale", "Weight"],
    )
    assert out == (
        "Name: Mini\n\nGeneral Instruction: Rate it.\n\n"
        "\nInstruction Items:\n"
        f"- Question: Own options?\n  Answer Options:\n{table}\n\n"
        "- Question: No options?\n  Answer Options:\n\n"
    )


# ------------------------------------------------------------------------------------------
# Prompt configuration classes
# ------------------------------------------------------------------------------------------


def test_normal_prompt_config_builds_a_prompt_template():
    config = NormalPromptTemplateConfig(template="Tell me a {adjective} joke about {content}")

    prompt = config.load_prompt_template()

    assert config.type == "normal"
    assert isinstance(prompt, PromptTemplate)
    assert sorted(prompt.input_variables) == ["adjective", "content"]
    assert prompt.format(adjective="short", content="cats") == "Tell me a short joke about cats"


def test_chat_prompt_config_builds_a_chat_prompt_template_with_roles():
    config = ChatPromptTemplateConfig(
        messages=[
            ChatMessageConfig(role="system", content="You are {persona}."),
            ChatMessageConfig(role="user", content="{question}"),
            ChatMessageConfig(role="assistant", content="Maybe."),
            ChatMessageConfig(role="human", content="{follow_up}"),
            ChatMessageConfig(role="ai", content="Sure."),
        ]
    )

    prompt = config.load_prompt_template()

    assert config.type == "chat"
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == ["follow_up", "persona", "question"]
    messages = prompt.format_messages(persona="Eva", question="Why?", follow_up="Really?")
    assert [(m.type, m.content) for m in messages] == [
        ("system", "You are Eva."),
        ("human", "Why?"),
        ("ai", "Maybe."),
        ("human", "Really?"),
        ("ai", "Sure."),
    ]


def test_chat_prompt_config_with_unknown_role_fails_when_the_prompt_is_built():
    config = ChatPromptTemplateConfig(messages=[ChatMessageConfig(role="wizard", content="hi")])
    with pytest.raises(ValueError, match="Unexpected message type: wizard"):
        config.load_prompt_template()


def test_langchain_prompt_config_loads_a_serialised_prompt():
    prompt = ChatPromptTemplate.from_messages([("system", "{a}"), ("user", "{b}")])
    config = LangchainPromptTemplateConfig(definition=dumpd(prompt))

    assert config.type == "langchain"
    assert config.load_prompt_template() == prompt


@pytest.mark.parametrize(
    "definition",
    [
        dumpd(FakeListLLM(responses=["x"])),
        {"lc": 1, "type": "constructor", "id": ["no_such_package", "Thing"], "kwargs": {}},
    ],
    ids=["not-serialisable", "unknown-namespace"],
)
def test_langchain_prompt_config_with_a_bad_definition_warns_and_returns_none(definition):
    config = LangchainPromptTemplateConfig(definition=definition)
    with pytest.warns(UserWarning, match="Failed to create LangChain prompt template"):
        assert config.load_prompt_template() is None


@pytest.mark.parametrize(
    "config_class, kwargs",
    [
        (NormalPromptTemplateConfig, {}),
        (ChatPromptTemplateConfig, {}),
        (ChatPromptTemplateConfig, {"messages": [{"role": "user"}]}),
        (ChatMessageConfig, {"role": "user"}),
        (ChatMessageConfig, {"content": "hi"}),
        (LangchainPromptTemplateConfig, {}),
        (LangchainPromptTemplateConfig, {"definition": "not a dict"}),
    ],
    ids=[
        "normal-no-template",
        "chat-no-messages",
        "chat-message-without-content",
        "message-no-content",
        "message-no-role",
        "langchain-no-definition",
        "langchain-definition-not-a-dict",
    ],
)
def test_invalid_prompt_configs_raise_validation_error(config_class, kwargs):
    with pytest.raises(ValidationError):
        config_class(**kwargs)


def test_prompt_configs_round_trip_through_model_dump():
    config = ChatPromptTemplateConfig(
        messages=[
            ChatMessageConfig(role="system", content="s"),
            ChatMessageConfig(role="user", content="u"),
        ]
    )
    assert ChatPromptTemplateConfig(**config.model_dump()) == config
    assert config.model_dump() == {
        "type": "chat",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    }


# ---- prompts.py ---------------------------------------------------------------------------------

RUN_LOOP_VARIABLES = {"general_instruction", "persona_description", "question", "answer_options"}


def test_default_chat_prompt_matches_the_variables_of_the_run_loop():
    config = prompts.DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG

    assert isinstance(config, ChatPromptTemplateConfig)
    assert [m.role for m in config.messages] == ["system", "user"]
    assert set(config.load_prompt_template().input_variables) == RUN_LOOP_VARIABLES
    assert "{general_instruction}" in config.messages[0].content
    assert "{persona_description}" in config.messages[1].content


def test_default_chat_prompt_text_is_the_documented_one():
    """The default prompt shapes every experiment that sets none (see docs/configuration.md)."""
    system, user = (m.content.strip() for m in prompts.DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG.messages)

    assert system == (
        'Objective: "{general_instruction}"\n'
        "Answer with respect to the following persona description and question."
    )
    assert user == (
        "Question:\n"
        "{persona_description} was asked the following question. {question}\n\n"
        "Answer Options:\n"
        "{answer_options}\n\n"
        "Instructions: Choose from the list of answer options to answer the question. Answer the "
        "question using only the provided answer options. If none of the options are correct, "
        "choose the option that is closest to being correct.\n\n"
        "Answer:"
    )


def test_prompt_template_configs_have_the_expected_types():
    assert isinstance(prompts.SIMPLE_PROMPT_TEMPLATE_CONFIG, NormalPromptTemplateConfig)
    assert isinstance(prompts.OPTIMIZED_PROMPT_TEMPLATE_CONFIG, NormalPromptTemplateConfig)
    assert isinstance(prompts.JSON_OUTPUT_PROMPT_TEMPLATE_CONFIG, NormalPromptTemplateConfig)
    assert [m.role for m in prompts.SIMPLE_CHAT_PROMPT_TEMPLATE_CONFIG.messages] == ["user"]
    assert [m.role for m in prompts.OPTIMIZED_CHAT_PROMPT_TEMPLATE_CONFIG.messages] == [
        "system",
        "user",
    ]
    assert [m.role for m in prompts.JSON_OUTPUT_CHAT_PROMPT_TEMPLATE_CONFIG.messages] == [
        "system",
        "user",
    ]


def test_prebuilt_prompts_are_the_loaded_configs():
    assert isinstance(prompts.PROMPT_TEMPLATE_SIMPLE, PromptTemplate)
    assert isinstance(prompts.PROMPT_TEMPLATE_OPTIMIZED, PromptTemplate)
    assert isinstance(prompts.PROMPT_TEMPLATE_JSON_OUTPUT, PromptTemplate)
    assert isinstance(prompts.CHAT_PROMPT_TEMPLATE_SIMPLE, ChatPromptTemplate)
    assert isinstance(prompts.CHAT_PROMPT_TEMPLATE_OPTIMIZED, ChatPromptTemplate)
    assert isinstance(prompts.CHAT_PROMPT_TEMPLATE_JSON_OUTPUT, ChatPromptTemplate)


def test_json_output_prompt_escapes_its_literal_braces():
    prompt = prompts.PROMPT_TEMPLATE_JSON_OUTPUT

    assert "answer" not in prompt.input_variables
    assert '{"answer": "answer_option"}' in prompt.format(
        **dict.fromkeys(prompt.input_variables, "x")
    )


@pytest.mark.xfail(
    strict=True,
    reason="the SIMPLE/OPTIMIZED/JSON_OUTPUT prompts use {general_instructions} and {persona} "
    "while the run loop supplies general_instruction and persona_description, so every model "
    "call fails with a missing-variables error",
)
@pytest.mark.parametrize(
    "name",
    [
        "PROMPT_TEMPLATE_SIMPLE",
        "CHAT_PROMPT_TEMPLATE_SIMPLE",
        "PROMPT_TEMPLATE_OPTIMIZED",
        "CHAT_PROMPT_TEMPLATE_OPTIMIZED",
        "PROMPT_TEMPLATE_JSON_OUTPUT",
        "CHAT_PROMPT_TEMPLATE_JSON_OUTPUT",
    ],
)
def test_prebuilt_prompts_only_use_variables_the_run_loop_supplies(name):
    prompt = getattr(prompts, name)
    assert set(prompt.input_variables) <= RUN_LOOP_VARIABLES


# ------------------------------------------------------------------------------------------
# Model configuration classes
# ------------------------------------------------------------------------------------------

ALL_MODEL_CONFIGS = [
    ("local_huggingface", LocalHuggingFaceModelConfig, {"name_or_path": "org/model"}),
    (
        "remote_huggingface",
        RemoteHuggingFaceModelConfig,
        {"repo_id": "org/model", "task": "text-generation"},
    ),
    ("ollama", OllamaModelConfig, {"model": "llama3"}),
    ("openai", OpenAIModelConfig, {"name_or_path": "gpt-4o-mini"}),
    ("google", GoogleModelConfig, {"name_or_path": "gemini-2.0-flash"}),
    ("deepseek", DeepSeekModelConfig, {"name_or_path": "deepseek-chat"}),
    ("langchain", LangChainModelConfig, {"definition": {"lc": 1}}),
]


@pytest.mark.parametrize("type_name, config_class, kwargs", ALL_MODEL_CONFIGS)
def test_model_config_defaults_to_its_own_type(type_name, config_class, kwargs):
    config = config_class(**kwargs)

    assert config.type == type_name
    assert config.parameters == {}
    assert callable(config.load_model)


@pytest.mark.parametrize("type_name, config_class, kwargs", ALL_MODEL_CONFIGS)
def test_model_config_parameters_are_not_shared_between_instances(type_name, config_class, kwargs):
    first, second = config_class(**kwargs), config_class(**kwargs)
    first.parameters["temperature"] = 0.5
    assert second.parameters == {}


def test_model_config_default_values():
    assert LocalHuggingFaceModelConfig(name_or_path="x").model_dump() == {
        "type": "local_huggingface",
        "name_or_path": "x",
        "revision": None,
        "tokenizer_name_or_path": None,
        "cache_dir": None,
        "huggingfacehub_api_token": None,
        "device_map": "auto",
        "task": "text-generation",
        "parameters": {},
        "prompt_template": None,
        "bitsandbytes_config": None,
    }
    assert OllamaModelConfig(model="llama3").model_dump() == {
        "type": "ollama",
        "model": "llama3",
        "base_url": "http://localhost:11434",
        "parameters": {},
        "prompt_template": None,
    }
    assert OpenAIModelConfig(name_or_path="gpt").model_dump() == {
        "type": "openai",
        "name_or_path": "gpt",
        "api_key": None,
        "base_url": None,
        "organization": None,
        "parameters": {},
        "prompt_template": None,
    }
    assert GoogleModelConfig(name_or_path="g").model_dump() == {
        "type": "google",
        "name_or_path": "g",
        "api_key": None,
        "parameters": {},
        "prompt_template": None,
    }
    assert DeepSeekModelConfig(name_or_path="d").model_dump() == {
        "type": "deepseek",
        "name_or_path": "d",
        "api_key": None,
        "parameters": {},
        "prompt_template": None,
    }
    assert RemoteHuggingFaceModelConfig(repo_id="a/b", task="t").model_dump() == {
        "type": "remote_huggingface",
        "repo_id": "a/b",
        "task": "t",
        "parameters": {},
    }
    assert LangChainModelConfig(definition={"x": 1}).model_dump() == {
        "type": "langchain",
        "definition": {"x": 1},
        "parameters": {},
        "prompt_template": None,
    }


@pytest.mark.parametrize(
    "config_class, kwargs",
    [
        (LocalHuggingFaceModelConfig, {}),
        (RemoteHuggingFaceModelConfig, {"repo_id": "org/model"}),
        (RemoteHuggingFaceModelConfig, {"task": "text-generation"}),
        (OllamaModelConfig, {}),
        (OpenAIModelConfig, {}),
        (GoogleModelConfig, {}),
        (DeepSeekModelConfig, {}),
        (LangChainModelConfig, {}),
        (LangChainModelConfig, {"definition": "not a dict"}),
        (LocalHuggingFaceModelConfig, {"name_or_path": 5}),
        (OpenAIModelConfig, {"name_or_path": "gpt", "parameters": ["not", "a", "dict"]}),
    ],
    ids=[
        "local-no-name",
        "remote-no-task",
        "remote-no-repo",
        "ollama-no-model",
        "openai-no-name",
        "google-no-name",
        "deepseek-no-name",
        "langchain-no-definition",
        "langchain-definition-not-a-dict",
        "local-name-not-a-string",
        "openai-parameters-not-a-dict",
    ],
)
def test_invalid_model_configs_raise_validation_error(config_class, kwargs):
    with pytest.raises(ValidationError):
        config_class(**kwargs)


def test_chat_prompt_template_can_be_attached_to_hosted_model_configs():
    prompt = {"type": "chat", "messages": [{"role": "user", "content": "{question}"}]}
    for config_class, kwargs in [
        (OpenAIModelConfig, {"name_or_path": "gpt"}),
        (GoogleModelConfig, {"name_or_path": "g"}),
        (DeepSeekModelConfig, {"name_or_path": "d"}),
        (OllamaModelConfig, {"model": "llama3"}),
    ]:
        assert isinstance(
            config_class(**kwargs, prompt_template=prompt).prompt_template, ChatPromptTemplateConfig
        )


def test_default_model_config():
    assert isinstance(DEFAULT_MODEL_CONFIG, LocalHuggingFaceModelConfig)
    assert DEFAULT_MODEL_CONFIG.name_or_path == "HuggingFaceTB/SmolLM-1.7b-Instruct"
    assert DEFAULT_MODEL_CONFIG.task == "text-generation"
    assert DEFAULT_MODEL_CONFIG.device_map == "cpu"


@pytest.mark.xfail(
    strict=True,
    reason="DEFAULT_MODEL_CONFIG_DICT passes its generation settings as 'pipeline_kwargs', "
    "which is not a field of LocalHuggingFaceModelConfig (the field is 'parameters'), so "
    "max_new_tokens, temperature, return_full_text ... are silently dropped",
)
def test_default_model_config_keeps_its_generation_settings():
    assert DEFAULT_MODEL_CONFIG.parameters == {
        "max_new_tokens": 64,
        "temperature": 1.0,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False,
    }


# ---- construction from dicts / type discrimination inside the experiment ------------------------


def test_models_of_every_type_are_discriminated_by_their_type_key(config_dict):
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "lh": {
            "type": "local_huggingface",
            "name_or_path": "org/model",
            "task": "text2text-generation",
        },
        "rh": {"type": "remote_huggingface", "repo_id": "org/model", "task": "text-generation"},
        "ol": {"type": "ollama", "model": "llama3", "base_url": "http://example:1234"},
        "oa": {"type": "openai", "name_or_path": "gpt-4o-mini", "organization": "org"},
        "go": {"type": "google", "name_or_path": "gemini-2.0-flash"},
        "ds": {"type": "deepseek", "name_or_path": "deepseek-chat"},
        "lc": {"type": "langchain", "definition": {"lc": 1}},
    }

    experiment = rup.experiment_from_dict(config)

    assert {key: type(model) for key, model in experiment.models.items()} == {
        "lh": LocalHuggingFaceModelConfig,
        "rh": RemoteHuggingFaceModelConfig,
        "ol": OllamaModelConfig,
        "oa": OpenAIModelConfig,
        "go": GoogleModelConfig,
        "ds": DeepSeekModelConfig,
        "lc": LangChainModelConfig,
    }
    assert experiment.models["lh"].task == "text2text-generation"
    assert experiment.models["ol"].base_url == "http://example:1234"
    assert experiment.models["oa"].organization == "org"
    assert list(experiment.runnable_models) == list(config["models"])  # lazy: configs, in order


def test_model_config_instances_are_kept_as_they_are():
    config = GoogleModelConfig(name_or_path="gemini")

    experiment = ExperimentDocument(parameters={}, models={"g": config})

    assert experiment.models["g"] is config
    assert type(experiment.models["g"]) is GoogleModelConfig


@pytest.mark.parametrize(
    "model",
    [{"type": "no-such-type", "name_or_path": "x"}, {"name_or_path": "x"}, {"type": None}, {}],
    ids=["unknown-type", "missing-type", "none-type", "empty"],
)
def test_unknown_or_missing_model_type_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown or missing model type for key: broken"):
        ExperimentDocument(parameters={}, models={"broken": model})


def test_invalid_model_fields_are_rejected_by_the_config_class():
    with pytest.raises(ValidationError):
        ExperimentDocument(parameters={}, models={"x": {"type": "openai"}})


def test_experiment_without_models_key_uses_the_default_model():
    experiment = ExperimentDocument(parameters={})

    assert list(experiment.models) == ["default_model"]
    assert experiment.models["default_model"] is DEFAULT_MODEL_CONFIG


def test_experiment_with_empty_models_has_none():
    experiment = ExperimentDocument(parameters={}, models={})
    assert experiment.models == {}
    assert experiment.runnable_models == {}


# ---- load_model() with faked back-ends ----------------------------------------------------------


class FakeClass:
    """Replacement for a class used in ``load_model``: remembers its constructor arguments."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def fake_class(name):
    return type(name, (FakeClass,), {})


def fake_auto_class(result):
    """Replacement for ``AutoTokenizer`` / ``AutoModelForCausalLM`` (class-level ``calls``)."""

    class FakeAuto:
        calls: list = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return result

    return FakeAuto


@pytest.fixture
def hf_fakes(monkeypatch):
    tokenizer, model = object(), object()
    pipelines = []

    def fake_pipeline(**kwargs):
        pipelines.append(kwargs)
        return "PIPELINE"

    fakes = {
        "tokenizer": tokenizer,
        "model": model,
        "pipelines": pipelines,
        "AutoTokenizer": fake_auto_class(tokenizer),
        "AutoModelForCausalLM": fake_auto_class(model),
        "BitsAndBytesConfig": fake_class("BitsAndBytesConfig"),
        "HuggingFacePipeline": fake_class("HuggingFacePipeline"),
        "ChatHuggingFace": fake_class("ChatHuggingFace"),
    }
    for name in ("AutoTokenizer", "AutoModelForCausalLM", "BitsAndBytesConfig"):
        monkeypatch.setattr(model_module, name, fakes[name])
    monkeypatch.setattr(model_module, "HuggingFacePipeline", fakes["HuggingFacePipeline"])
    monkeypatch.setattr(model_module, "ChatHuggingFace", fakes["ChatHuggingFace"])
    monkeypatch.setattr(model_module, "pipeline", fake_pipeline)
    return fakes


def test_local_huggingface_load_model_wires_tokenizer_model_and_pipeline(hf_fakes):
    config = LocalHuggingFaceModelConfig(
        name_or_path="org/model",
        tokenizer_name_or_path="org/tokenizer",
        device_map="cpu",
        task="text-generation",
        parameters={"max_new_tokens": 5, "do_sample": False},
    )

    chat = config.load_model()

    assert hf_fakes["AutoTokenizer"].calls == [(("org/tokenizer",), {})]
    assert hf_fakes["AutoModelForCausalLM"].calls == [(("org/model",), {"device_map": "cpu"})]
    assert hf_fakes["pipelines"] == [
        {
            "task": "text-generation",
            "model": hf_fakes["model"],
            "tokenizer": hf_fakes["tokenizer"],
            "max_new_tokens": 5,
            "do_sample": False,
        }
    ]
    assert isinstance(chat, hf_fakes["ChatHuggingFace"])
    llm = chat.kwargs["llm"]
    assert isinstance(llm, hf_fakes["HuggingFacePipeline"])
    assert llm.kwargs == {"pipeline": "PIPELINE", "model_id": "org/model"}


def test_local_huggingface_tokenizer_defaults_to_the_model_name(hf_fakes):
    LocalHuggingFaceModelConfig(name_or_path="org/model").load_model()

    assert hf_fakes["AutoTokenizer"].calls == [(("org/model",), {})]
    assert hf_fakes["AutoModelForCausalLM"].calls == [(("org/model",), {"device_map": "auto"})]


def test_local_huggingface_quantization_config_is_forwarded(hf_fakes):
    LocalHuggingFaceModelConfig(
        name_or_path="org/model", bitsandbytes_config={"load_in_4bit": True}
    ).load_model()

    ((args, kwargs),) = hf_fakes["AutoModelForCausalLM"].calls
    assert args == ("org/model",)
    assert kwargs["device_map"] == "auto"
    assert isinstance(kwargs["quantization_config"], hf_fakes["BitsAndBytesConfig"])
    assert kwargs["quantization_config"].kwargs == {"load_in_4bit": True}


def test_local_huggingface_load_failure_is_reported_as_value_error(hf_fakes):
    boom = OSError("not found on the hub")

    def fail(*args, **kwargs):
        raise boom

    hf_fakes["AutoTokenizer"].from_pretrained = staticmethod(fail)

    with pytest.raises(ValueError) as error:
        LocalHuggingFaceModelConfig(name_or_path="org/model", task="text-generation").load_model()

    assert str(error.value) == (
        "Failed to load the Hugging Face model 'org/model' with task 'text-generation': "
        "not found on the hub"
    )
    assert error.value.__cause__ is boom


def test_remote_huggingface_load_model(monkeypatch):
    endpoint_class, chat_class = fake_class("HuggingFaceEndpoint"), fake_class("ChatHuggingFace")
    monkeypatch.setattr(model_module, "HuggingFaceEndpoint", endpoint_class)
    monkeypatch.setattr(model_module, "ChatHuggingFace", chat_class)

    chat = RemoteHuggingFaceModelConfig(
        repo_id="org/model", task="text-generation", parameters={"max_new_tokens": 7}
    ).load_model()

    assert isinstance(chat, chat_class)
    endpoint = chat.kwargs["llm"]
    assert isinstance(endpoint, endpoint_class)
    assert endpoint.kwargs == {
        "repo_id": "org/model",
        "task": "text-generation",
        "max_new_tokens": 7,
    }


def test_remote_huggingface_load_failure_is_reported_as_value_error(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("no token")

    monkeypatch.setattr(model_module, "HuggingFaceEndpoint", fail)

    with pytest.raises(ValueError, match="Failed to load the remote Hugging Face model: no token"):
        RemoteHuggingFaceModelConfig(repo_id="org/model", task="text-generation").load_model()


@pytest.mark.parametrize(
    "config, patched_name, expected_kwargs, error_prefix",
    [
        (
            OllamaModelConfig(
                model="llama3", base_url="http://example:1", parameters={"temperature": 0.2}
            ),
            "OllamaLLM",
            {"model": "llama3", "base_url": "http://example:1", "temperature": 0.2},
            "Failed to load the Ollama model: ",
        ),
        (
            OpenAIModelConfig(
                name_or_path="gpt-4o-mini",
                api_key="sk-test",
                base_url="http://proxy",
                organization="org",
                parameters={"temperature": 0.3},
            ),
            "ChatOpenAI",
            {
                "model": "gpt-4o-mini",
                "api_key": "sk-test",
                "base_url": "http://proxy",
                "organization": "org",
                "temperature": 0.3,
            },
            "Failed to load the OpenAI model: ",
        ),
        (
            GoogleModelConfig(name_or_path="gemini", api_key="key", parameters={"top_k": 4}),
            "ChatGoogleGenerativeAI",
            {"model": "gemini", "api_key": "key", "top_k": 4},
            "Failed to load the Google model: ",
        ),
    ],
    ids=["ollama", "openai", "google"],
)
def test_hosted_model_configs_pass_their_settings_to_the_client(
    monkeypatch, config, patched_name, expected_kwargs, error_prefix
):
    client_class = fake_class(patched_name)
    monkeypatch.setattr(model_module, patched_name, client_class)

    client = config.load_model()

    assert isinstance(client, client_class)
    assert client.args == ()
    assert client.kwargs == expected_kwargs

    def fail(**kwargs):
        raise RuntimeError("missing key")

    monkeypatch.setattr(model_module, patched_name, fail)
    with pytest.raises(ValueError) as error:
        config.load_model()
    assert str(error.value) == f"{error_prefix}missing key"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_deepseek_load_model_passes_the_model_and_parameters(monkeypatch):
    client_class = fake_class("ChatDeepSeek")
    monkeypatch.setattr(model_module, "ChatDeepSeek", client_class)

    client = DeepSeekModelConfig(
        name_or_path="deepseek-chat", parameters={"temperature": 0.4}
    ).load_model()

    assert isinstance(client, client_class)
    assert client.kwargs["model"] == "deepseek-chat"
    assert client.kwargs["temperature"] == 0.4


def test_deepseek_load_failure_is_reported_as_value_error(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("no api key")

    monkeypatch.setattr(model_module, "ChatDeepSeek", fail)

    with pytest.raises(ValueError, match="Failed to load the DeepSeek model: no api key"):
        DeepSeekModelConfig(name_or_path="deepseek-chat").load_model()


def test_langchain_model_config_loads_a_serialised_object():
    definition = dumpd(PromptTemplate.from_template("hi {name}"))

    loaded = LangChainModelConfig(definition=definition).load_model()

    assert loaded == PromptTemplate.from_template("hi {name}")


def test_langchain_model_config_with_a_bad_definition_warns_and_returns_none():
    config = LangChainModelConfig(definition=dumpd(FakeListLLM(responses=["x"])))

    with pytest.warns(UserWarning, match="Failed to load LangChain model"):
        assert config.load_model() is None


@pytest.mark.xfail(
    strict=True,
    reason="load() passes a dict that is not a serialised LangChain object through unchanged, so "
    "load_model returns that dict as 'model' and run() later fails with AttributeError: "
    "'dict' object has no attribute 'bind'",
)
def test_langchain_model_config_rejects_a_definition_that_is_not_a_serialised_object():
    config = LangChainModelConfig(definition={"model": "gpt-4"})

    with pytest.warns(UserWarning, match="Failed to load LangChain model"):
        assert config.load_model() is None


# ------------------------------------------------------------------------------------------
# ExperimentParameters
# ------------------------------------------------------------------------------------------


def test_parameters_default_to_one_random_seed_and_lazy_loading():
    parameters = ExperimentParameters()

    assert len(parameters.seeds) == 1
    assert parameters.seeds[0].isdigit()
    assert 0 <= int(parameters.seeds[0]) <= 999999
    assert parameters.lazy_load_models is True


def test_default_seed_is_drawn_for_every_instance(monkeypatch):
    """CHANGELOG: the random default seed used to be drawn once at import time."""
    draws = iter([11, 22, 33])
    monkeypatch.setattr("rupsycho.models.parameters.randint", lambda low, high: next(draws))

    seeds = [ExperimentParameters().seeds for _ in range(3)]

    assert seeds == [["11"], ["22"], ["33"]]


def test_default_seed_range_is_zero_to_999999(monkeypatch):
    bounds = []

    def record(low, high):
        bounds.append((low, high))
        return high

    monkeypatch.setattr("rupsycho.models.parameters.randint", record)

    assert ExperimentParameters().seeds == ["999999"]
    assert bounds == [(0, 999999)]


def test_default_seed_lists_are_not_shared():
    first, second = ExperimentParameters(), ExperimentParameters()
    first.seeds.append("extra")
    assert "extra" not in second.seeds


def test_two_experiments_from_the_same_config_without_seeds_draw_their_own(
    config_dict, monkeypatch
):
    draws = iter([5, 6])
    monkeypatch.setattr("rupsycho.models.parameters.randint", lambda low, high: next(draws))
    config = copy.deepcopy(config_dict)
    del config["parameters"]["seeds"]

    first, second = rup.experiment_from_dict(config), rup.experiment_from_dict(config)

    assert first.parameters.seeds == ["5"]
    assert second.parameters.seeds == ["6"]


@pytest.mark.parametrize("seeds", [["1", "2", "3"], [], None], ids=["several", "empty", "null"])
def test_explicit_seeds_are_kept_as_given(seeds):
    assert ExperimentParameters(seeds=seeds).seeds == seeds


@pytest.mark.parametrize("seeds", [[1, 2], "1", [None]], ids=["ints", "string", "list-of-none"])
def test_seeds_must_be_a_list_of_strings(seeds):
    with pytest.raises(ValidationError):
        ExperimentParameters(seeds=seeds)


def test_parameters_allow_extra_keys():
    parameters = ExperimentParameters(seeds=["1"], temperature=0.5, note="free text")

    assert parameters.temperature == 0.5
    assert parameters.note == "free text"
    assert parameters.model_dump() == {
        "seeds": ["1"],
        "lazy_load_models": True,
        "temperature": 0.5,
        "note": "free text",
    }


def test_lazy_load_models_can_be_switched_off():
    assert ExperimentParameters(lazy_load_models=False).lazy_load_models is False
    with pytest.raises(ValidationError):
        ExperimentParameters(lazy_load_models="sometimes")


def test_experiment_converts_parameters_dict_and_keeps_extras(config_dict):
    config = copy.deepcopy(config_dict)
    config["parameters"] = {"seeds": ["3"], "temperature": 0.7}

    experiment = rup.experiment_from_dict(config)

    assert isinstance(experiment.parameters, ExperimentParameters)
    assert experiment.parameters.seeds == ["3"]
    assert experiment.parameters.temperature == 0.7


# ------------------------------------------------------------------------------------------
# ExperimentDocument construction and conversion helpers
# ------------------------------------------------------------------------------------------


def test_convert_parameters():
    parameters = ExperimentParameters(seeds=["1"])
    assert ExperimentDocument._convert_parameters(parameters) is parameters
    converted = ExperimentDocument._convert_parameters({"seeds": ["2"], "lazy_load_models": False})
    assert converted == ExperimentParameters(seeds=["2"], lazy_load_models=False)


def test_convert_demographic_profiles_keeps_instances_and_converts_dicts():
    ready = DemographicProfile(attributes={"name": "Ready"})

    converted = ExperimentDocument._convert_demographic_profiles(
        {"ready": ready, "raw": {"attributes": {"name": "Raw"}, "template": "{name}"}}
    )

    assert converted["ready"] is ready
    assert converted["raw"] == DemographicProfile(attributes={"name": "Raw"}, template="{name}")


@pytest.mark.parametrize(
    "prompt, expected_class",
    [
        ({"type": "normal", "template": "{question}"}, NormalPromptTemplateConfig),
        (
            {"type": "chat", "messages": [{"role": "user", "content": "{question}"}]},
            ChatPromptTemplateConfig,
        ),
        ({"type": "langchain", "definition": {"lc": 1}}, LangchainPromptTemplateConfig),
    ],
)
def test_convert_prompt_selects_the_config_class_by_type(prompt, expected_class):
    assert type(ExperimentDocument._convert_prompt(prompt)) is expected_class


def test_convert_prompt_keeps_config_instances():
    config = NormalPromptTemplateConfig(template="{question}")
    assert ExperimentDocument._convert_prompt(config) is config


@pytest.mark.parametrize(
    "prompt",
    [{"type": "no-such-type", "template": "x"}, {"template": "x"}, {"type": ""}],
    ids=["unknown-type", "missing-type", "empty-type"],
)
def test_convert_prompt_rejects_unknown_or_missing_types(prompt):
    with pytest.raises(ValueError, match="Unknown or missing prompt type."):
        ExperimentDocument._convert_prompt(prompt)


def test_convert_questionnaire_keeps_instances_and_converts_dicts():
    ready = Questionnaire(name="n", general_instruction="g")
    assert ExperimentDocument._convert_questionnaire(ready) is ready
    converted = ExperimentDocument._convert_questionnaire({"name": "n", "general_instruction": "g"})
    assert converted == ready


def test_experiment_accepts_ready_made_objects():
    questionnaire = Questionnaire(
        name="n",
        general_instruction="g",
        default_answer_options={"1": {"text": "yes"}},
        instruction_items=[InstructionItem(question="q?")],
    )
    profile = DemographicProfile(attributes={"name": "Eva"}, template="{name}")
    prompt = ChatPromptTemplateConfig(
        messages=[ChatMessageConfig(role="user", content="{question}")]
    )

    experiment = ExperimentDocument(
        name="ready",
        parameters=ExperimentParameters(seeds=["1"]),
        prompt_template=prompt,
        models={},
        demographic_profiles={"eva": profile},
        questionnaire=questionnaire,
    )

    assert experiment.questionnaire is questionnaire
    assert experiment.demographic_profiles["eva"] is profile
    assert experiment.prompt_template is prompt
    assert isinstance(experiment.get_prompt(), ChatPromptTemplate)


def test_experiment_string_representation_is_restricted_to_name_description_metadata():
    experiment = ExperimentDocument(
        parameters={}, name="Name", description="Desc", metadata={"source": "Lab"}
    )
    assert str(experiment) == "name=Name, description=Desc, metadata={'source': 'Lab'}"


def test_experiment_defaults_without_optional_keys():
    experiment = ExperimentDocument(parameters={})

    assert experiment.name is None
    assert experiment.description is None
    assert experiment.questionnaire is None
    assert experiment.demographic_profiles == {}
    assert experiment.metadata == {}
    assert experiment.type == "ExperimentDocument"
    assert experiment.runnable_parser == {}
    assert experiment.prompt_template is prompts.DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG


def test_string_prompt_templates_are_used_as_they_are():
    experiment = ExperimentDocument(parameters={})
    assert experiment._load_runnable_prompt("already a prompt") == "already a prompt"


def test_set_questionnaire_and_set_parser():
    experiment = ExperimentDocument(parameters={})
    questionnaire = Questionnaire(name="n", general_instruction="g")
    parser = object()

    experiment.set_questionnaire(questionnaire)
    experiment.set_parser(parser)

    assert experiment.questionnaire is questionnaire
    assert experiment.runnable_parser is parser


@pytest.mark.xfail(
    strict=True,
    reason="ExperimentDocument.__init__ reads data.get('parameters').lazy_load_models, so "
    "omitting 'parameters' (which has a default_factory) raises AttributeError although the "
    "class docstring constructs documents without it",
)
def test_experiment_can_be_created_without_parameters():
    experiment = ExperimentDocument(name="no parameters given", models={})

    assert experiment.name == "no parameters given"
    assert isinstance(experiment.parameters, ExperimentParameters)
