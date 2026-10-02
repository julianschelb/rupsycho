"""Property-based tests of the questionnaire data model (``rupsycho.models.questionnaire``).

Oracle strategy
---------------
The model classes are thin, so the oracles are *constructions*: every test generates the
structure first and derives the expected value from it by plain Python, never by calling the
code under test.

* ``AnswerOptions.join_options`` is compared with a hand-written join loop
  (``naive_join``), ``get_options_as_list`` with the texts the test put in.
* The ``dict -> AnswerOptions`` conversion is checked field by field against the documented
  defaults ("Choose an option", ``False``, ``0``, delimiter ``", "``), and the flat and the
  ``{"options": ...}`` spellings must agree.
* ``InstructionItem.update_answer`` / ``get_answer`` / ``get_all_answers`` are compared with a
  nested ``dict`` model that is updated alongside (``None`` never stored, later writes win).
* ``DemographicProfile`` rendering is compared with the string the test builds from the same
  literal and placeholder parts (``render_parts``) without using ``str.format``.
* Questionnaires must survive ``model_dump -> model_validate`` and the JSON round trip, and
  ``print_questionnaire`` must show every name, question and option text that was put in.

Where the model violates an intended invariant, the test is written for the correct behaviour
and marked ``xfail(strict=True)`` with an ``@example`` that reproduces the bug; fixing the bug
turns it into an unexpected pass, which fails the suite and says "remove the marker".
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import string

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from rupsycho.models.questionnaire import (
    AnswerOption,
    AnswerOptions,
    DemographicProfile,
    InstructionItem,
    Questionnaire,
)

from .helpers import naive_join, plain, render_parts, template_from_parts

PROFILE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

DEFAULT_TEXT = "Choose an option"  # AnswerOption.text default, documented in the model

# =============================================================================
# Strategies
# =============================================================================

option_ids = st.text(alphabet=string.ascii_letters + string.digits + "_- .", min_size=1, max_size=6)
flat_ids = option_ids.filter(lambda identifier: identifier != "options")  # "options" is the wrapper
option_texts = st.text(max_size=20)
option_definitions = st.fixed_dictionaries(
    {},
    optional={
        "text": option_texts,
        "ignored_for_scale": st.booleans(),
        "weight": st.integers(-5, 5),
    },
)
flat_options = st.dictionaries(flat_ids, option_definitions, max_size=6)
delimiters = st.one_of(st.sampled_from([", ", " | ", "\n", "", "; "]), st.text(max_size=3))

words = st.text(alphabet=string.ascii_letters, min_size=1, max_size=10)
json_leaf = st.one_of(st.none(), st.booleans(), st.integers(-5, 5), st.text(max_size=6))
json_dicts = st.dictionaries(st.text(max_size=5), json_leaf, max_size=3)

model_ids = st.text(max_size=8)
profile_ids = st.text(max_size=8)
seeds = st.one_of(st.integers(-5, 5), st.text(max_size=4))
answer_texts = st.text(max_size=12)


def wrapped(flat: dict, delimiter: str = ", ", prepend: bool = False) -> dict:
    """The ``{"options": ...}`` spelling of a flat option dictionary."""
    return {"options": flat, "delimiter": delimiter, "prepend_delimiter": prepend}


@st.composite
def answer_option_specs(draw):
    """Either spelling of answer options, together with what they mean.

    Returns:
        ``(spec, flat, delimiter, prepend)`` where ``spec`` is what a configuration file would
        contain and the other three describe the intended ``AnswerOptions``.
    """
    flat = draw(flat_options)
    if draw(st.booleans()):
        return flat, flat, ", ", False
    delimiter, prepend = draw(delimiters), draw(st.booleans())
    return wrapped(flat, delimiter, prepend), flat, delimiter, prepend


@st.composite
def questionnaires(draw) -> Questionnaire:
    """A questionnaire with profiles, default options and items with options and answers."""
    profiles = draw(
        st.one_of(
            st.none(),
            st.lists(
                st.builds(
                    lambda attributes, template: {"attributes": attributes, "template": template},
                    st.fixed_dictionaries({}, optional={"age": st.integers(0, 99), "name": words}),
                    st.sampled_from(["{name} is {age}.", "{age}", "plain text", "{{braces}}"]),
                ),
                max_size=3,
            ),
        )
    )
    items = draw(
        st.one_of(
            st.none(),
            st.lists(
                st.fixed_dictionaries(
                    {"question": st.text(max_size=20)},
                    optional={
                        "reversed": st.booleans(),
                        "answer_options": answer_option_specs().map(lambda spec: spec[0]),
                        "attributes": json_dicts,
                        "answers": st.dictionaries(
                            model_ids,
                            st.dictionaries(
                                profile_ids,
                                st.dictionaries(st.text(max_size=4), answer_texts, max_size=2),
                                max_size=2,
                            ),
                            max_size=2,
                        ),
                    },
                ),
                max_size=4,
            ),
        )
    )
    return Questionnaire(
        name=draw(st.text(max_size=20)),
        general_instruction=draw(st.text(max_size=40)),
        demographic_profiles=profiles,
        attributes=draw(json_dicts),
        default_answer_options=draw(
            st.one_of(st.none(), answer_option_specs().map(lambda s: s[0]))
        ),
        instruction_items=items,
    )


# =============================================================================
# AnswerOptions
# =============================================================================


class TestAnswerOptions:
    @PROFILE
    @given(
        texts=st.lists(option_texts, max_size=6),
        delimiter=delimiters,
        prepend=st.booleans(),
    )
    def test_join_options_equals_the_naive_join(self, texts, delimiter, prepend):
        options = AnswerOptions(
            options={str(index): AnswerOption(text=text) for index, text in enumerate(texts)},
            delimiter=delimiter,
            prepend_delimiter=prepend,
        )
        assert options.join_options() == naive_join(texts, delimiter, prepend)
        assert options.get_options_as_list() == texts

    @PROFILE
    @given(texts=st.lists(option_texts, min_size=1, max_size=6), delimiter=delimiters)
    def test_prepending_adds_exactly_one_leading_delimiter(self, texts, delimiter):
        def joined(prepend):
            options = {str(i): AnswerOption(text=text) for i, text in enumerate(texts)}
            return AnswerOptions(
                options=options, delimiter=delimiter, prepend_delimiter=prepend
            ).join_options()

        assert joined(True) == delimiter + joined(False)

    @PROFILE
    @given(delimiter=delimiters, prepend=st.booleans())
    def test_no_options_join_to_the_empty_string(self, delimiter, prepend):
        options = AnswerOptions(delimiter=delimiter, prepend_delimiter=prepend)
        assert options.join_options() == ""
        assert options.get_options_as_list() == []

    @PROFILE
    @given(spec=answer_option_specs())
    def test_both_spellings_convert_to_the_described_options(self, spec):
        definition, flat, delimiter, prepend = spec
        converted = InstructionItem(answer_options=definition).answer_options
        assert converted is not None
        assert list(converted.options) == list(flat)  # ids and their order
        for identifier, given_option in flat.items():
            option = converted.options[identifier]
            assert option.text == given_option.get("text", DEFAULT_TEXT)
            assert option.ignored_for_scale is given_option.get("ignored_for_scale", False)
            assert option.weight == given_option.get("weight", 0)
        assert converted.delimiter == delimiter
        assert converted.prepend_delimiter is prepend

    @PROFILE
    @given(flat=flat_options)
    def test_the_flat_spelling_equals_the_wrapper_with_defaults(self, flat):
        assert (
            InstructionItem(answer_options=flat).answer_options
            == InstructionItem(answer_options={"options": flat}).answer_options
            == InstructionItem(answer_options=wrapped(flat)).answer_options
        )

    @PROFILE
    @given(spec=answer_option_specs())
    def test_item_and_questionnaire_convert_alike(self, spec):
        definition = spec[0]
        from_item = InstructionItem(answer_options=definition).answer_options
        questionnaire = Questionnaire(
            name="n", general_instruction="g", default_answer_options=definition
        )
        assert questionnaire.default_answer_options == from_item

    @PROFILE
    @given(spec=answer_option_specs())
    def test_a_model_instance_and_its_dump_are_accepted_unchanged(self, spec):
        converted = InstructionItem(answer_options=spec[0]).answer_options
        assert InstructionItem(answer_options=converted).answer_options == converted
        assert InstructionItem(answer_options=converted.model_dump()).answer_options == converted

    @PROFILE
    @given(spec=answer_option_specs())
    def test_the_item_lists_the_texts_of_its_options(self, spec):
        definition, flat, _, _ = spec
        item = InstructionItem(answer_options=definition)
        assert item.get_answer_options_as_list() == [
            given_option.get("text", DEFAULT_TEXT) for given_option in flat.values()
        ]

    def test_an_item_without_options_lists_none(self):
        assert InstructionItem().answer_options is None
        assert InstructionItem().get_answer_options_as_list() == []

    @PROFILE
    @given(
        definition=st.recursive(
            json_leaf,
            lambda children: st.one_of(
                st.lists(children, max_size=3),
                st.dictionaries(
                    st.sampled_from(["options", "1", "delimiter", "text"]), children, max_size=3
                ),
            ),
            max_leaves=6,
        )
    )
    @example(definition={"1": "agree"})
    @example(definition={"options": ["agree"]})
    @example(definition={"options": None})
    @example(definition={"options": 5})
    @example(definition={"options": {"text": "an option called options"}})  # the reserved id
    def test_invalid_options_raise_a_validation_error_and_nothing_else(self, definition):
        try:
            InstructionItem(answer_options=definition)
        except ValidationError:
            pass  # the documented way to say "invalid"


# =============================================================================
# InstructionItem answers
# =============================================================================

operations = st.lists(
    st.tuples(
        st.sampled_from(["m1", "m2", "model 3", ""]),
        st.sampled_from(["p1", "p2", "persona {x}"]),
        st.sampled_from([0, 1, "7", "8"]),
        st.one_of(st.none(), answer_texts),
    ),
    max_size=25,
)


def apply_naively(model: dict, operation) -> None:
    """The oracle's update: store the answer unless it is ``None``; a later write wins."""
    model_key, profile_key, seed, answer = operation
    if answer is not None:
        model.setdefault(model_key, {}).setdefault(profile_key, {})[seed] = answer


class TestInstructionItemAnswers:
    @PROFILE
    @given(operations=operations)
    def test_the_stored_answers_follow_a_nested_dictionary_model(self, operations):
        item, model = InstructionItem(), {}
        for operation in operations:
            item.update_answer(*operation)
            apply_naively(model, operation)
            assert plain(item.get_all_answers()) == model  # also after every single step
        for model_key, personas in model.items():
            for profile_key, runs in personas.items():
                for seed, answer in runs.items():
                    assert item.get_answer(model_key, profile_key, seed) == answer

    @PROFILE
    @given(
        model_key=model_ids,
        profile_key=profile_ids,
        seed=seeds,
        answer=answer_texts,
    )
    def test_an_answer_round_trips(self, model_key, profile_key, seed, answer):
        item = InstructionItem()
        item.update_answer(model_key, profile_key, seed, answer)
        assert item.get_answer(model_key, profile_key, seed) == answer
        assert plain(item.get_all_answers()) == {model_key: {profile_key: {seed: answer}}}

    @PROFILE
    @given(operations=operations, model_key=model_ids, profile_key=profile_ids, seed=seeds)
    def test_none_is_never_stored(self, operations, model_key, profile_key, seed):
        item = InstructionItem()
        for operation in operations:
            item.update_answer(*operation)
        before = copy.deepcopy(plain(item.get_all_answers()))
        item.update_answer(model_key, profile_key, seed, None)
        assert plain(item.get_all_answers()) == before  # not even an empty branch appears

    @PROFILE
    @given(operations=operations)
    def test_reading_stored_answers_changes_nothing(self, operations):
        item, model = InstructionItem(), {}
        for operation in operations:
            item.update_answer(*operation)
            apply_naively(model, operation)
        for model_key, personas in model.items():
            for profile_key, runs in personas.items():
                for seed in runs:
                    item.get_answer(model_key, profile_key, seed)
        item.get_all_answers()
        assert plain(item.get_all_answers()) == model

    @PROFILE
    @given(operations=operations, model_key=model_ids, profile_key=profile_ids, seed=seeds)
    @example(operations=[], model_key="m", profile_key="p", seed=1)
    def test_asking_for_a_missing_answer_changes_nothing(
        self, operations, model_key, profile_key, seed
    ):
        item, model = InstructionItem(), {}
        for operation in operations:
            item.update_answer(*operation)
            apply_naively(model, operation)
        stored = model.get(model_key, {}).get(profile_key, {}).get(seed)  # None if missing
        with contextlib.suppress(KeyError):  # raising KeyError is an acceptable way to say "no"
            assert item.get_answer(model_key, profile_key, seed) == stored
        assert plain(item.get_all_answers()) == model  # but the read must not have written

    @PROFILE
    @given(first=operations, second=operations)
    def test_items_do_not_share_their_answers(self, first, second):
        one, two = InstructionItem(), InstructionItem()
        for operation in first:
            one.update_answer(*operation)
        for operation in second:
            two.update_answer(*operation)
        model_one, model_two = {}, {}
        for operation in first:
            apply_naively(model_one, operation)
        for operation in second:
            apply_naively(model_two, operation)
        assert plain(one.get_all_answers()) == model_one
        assert plain(two.get_all_answers()) == model_two

    @PROFILE
    @given(operations=operations, more=operations)
    def test_an_item_keeps_working_after_a_dump_and_validate_round_trip(self, operations, more):
        """Loading stored results must not break storing new ones (new models, personas, seeds)."""
        item, model = InstructionItem(), {}
        for operation in operations:
            item.update_answer(*operation)
            apply_naively(model, operation)
        restored = InstructionItem.model_validate(item.model_dump())
        assert plain(restored.get_all_answers()) == model
        for operation in more:
            restored.update_answer(*operation)
            apply_naively(model, operation)
        assert plain(restored.get_all_answers()) == model

    @PROFILE
    @given(operations=operations, more=operations)
    def test_copies_keep_working_and_stay_independent(self, operations, more):
        item, model = InstructionItem(), {}
        for operation in operations:
            item.update_answer(*operation)
            apply_naively(model, operation)
        for duplicate in (copy.deepcopy(item), item.model_copy(deep=True)):
            for operation in more:
                duplicate.update_answer(*operation)
            assert plain(item.get_all_answers()) == model  # the original is untouched

    @PROFILE
    @given(
        answers=st.dictionaries(
            model_ids,
            st.dictionaries(
                profile_ids,
                st.dictionaries(st.text(max_size=3), answer_texts, max_size=2),
                max_size=2,
            ),
            max_size=2,
        )
    )
    def test_answers_given_at_construction_are_stored_and_extendable(self, answers):
        item = InstructionItem(answers=copy.deepcopy(answers))
        assert plain(item.get_all_answers()) == answers
        item.update_answer("brand new model", "brand new persona", 1, "x")
        assert item.get_answer("brand new model", "brand new persona", 1) == "x"

    @pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
    def test_non_string_answers_survive_a_dump_and_validate_round_trip(self):
        for answer in (3, 0.5, True, {"text": "x", "validation_status": "valid"}, ["a"]):
            item = InstructionItem()
            item.update_answer("m", "p", "1", answer)
            assert InstructionItem.model_validate(item.model_dump()) == item, answer


# =============================================================================
# DemographicProfile
# =============================================================================

ATTRIBUTE_NAMES = ("age", "title", "name", "city", "job")
attribute_values = st.one_of(
    st.none(),
    st.integers(),
    st.text(max_size=8),
    st.booleans(),
    st.floats(allow_nan=False),
)


@st.composite
def profile_scenarios(draw):
    """Attributes and a template built from literal text and ``{placeholder}`` parts.

    Returns:
        ``(attributes, parts)``; the placeholders refer to the given attributes or to one of
        ``age`` / ``title`` / ``name``, which exist (as ``None``) on every profile.
    """
    attributes = draw(
        st.dictionaries(st.sampled_from(ATTRIBUTE_NAMES), attribute_values, max_size=5)
    )
    names = sorted({*attributes, "age", "title", "name"})
    parts = draw(
        st.lists(
            st.one_of(
                st.tuples(
                    st.just("lit"),
                    st.one_of(
                        st.text(max_size=8),
                        st.sampled_from(["{", "}", "{{", "}}", "{}", " ", "\n", "%", "{0}"]),
                    ),
                ),
                st.tuples(st.just("ph"), st.sampled_from(names)),
            ),
            max_size=8,
        )
    )
    return attributes, parts


class TestDemographicProfile:
    @PROFILE
    @given(scenario=profile_scenarios())
    def test_the_description_is_the_template_filled_with_the_attributes(self, scenario):
        attributes, parts = scenario
        profile = DemographicProfile(attributes=attributes, template=template_from_parts(parts))
        values = {"age": None, "title": None, "name": None, **attributes}
        assert str(profile) == render_parts(parts, values)
        assert profile.get_profile_desc() == str(profile)

    @PROFILE
    @given(age=st.integers(0, 120), title=words, name=words)
    def test_the_default_template(self, age, title, name):
        profile = DemographicProfile(attributes={"age": age, "title": title, "name": name})
        assert str(profile) == f"{title} {name} is {age} years old."

    @PROFILE
    @given(scenario=profile_scenarios())
    def test_a_profile_survives_a_dump_and_validate_round_trip(self, scenario):
        attributes, parts = scenario
        profile = DemographicProfile(attributes=attributes, template=template_from_parts(parts))
        restored = DemographicProfile.model_validate(profile.model_dump())
        assert restored == profile
        assert str(restored) == str(profile)

    @PROFILE
    @given(value=st.text(max_size=10))
    def test_braces_in_attribute_values_are_not_interpreted(self, value):
        profile = DemographicProfile(attributes={"name": value}, template="[{name}]")
        assert str(profile) == f"[{value}]"


# =============================================================================
# Questionnaire
# =============================================================================


class TestQuestionnaire:
    @PROFILE
    @given(questionnaire=questionnaires())
    def test_a_dump_validates_back_to_an_equal_questionnaire(self, questionnaire):
        assert Questionnaire.model_validate(questionnaire.model_dump()) == questionnaire

    @PROFILE
    @given(questionnaire=questionnaires())
    def test_the_json_round_trip_is_lossless(self, questionnaire):
        as_json = questionnaire.model_dump_json()
        assert Questionnaire.model_validate_json(as_json) == questionnaire
        assert Questionnaire.model_validate(json.loads(as_json)) == questionnaire

    @PROFILE
    @given(questionnaire=questionnaires())
    def test_the_number_of_questions_is_the_number_of_items(self, questionnaire):
        items = questionnaire.instruction_items
        assert questionnaire.get_number_of_questions() == (0 if items is None else len(items))

    @PROFILE
    @given(questionnaire=questionnaires(), more=operations)
    def test_a_loaded_questionnaire_accepts_new_answers(self, questionnaire, more):
        restored = Questionnaire.model_validate(questionnaire.model_dump())
        for item in restored.instruction_items or []:
            model = plain(item.get_all_answers())
            for operation in more:
                item.update_answer(*operation)
                apply_naively(model, operation)
            assert plain(item.get_all_answers()) == model

    @PROFILE
    @given(
        name=words,
        instruction=words,
        questions=st.lists(words, max_size=4),
        texts=st.lists(words, max_size=4, unique=True),
        with_defaults=st.booleans(),
    )
    def test_print_questionnaire_shows_everything_that_was_put_in(
        self, name, instruction, questions, texts, with_defaults
    ):
        options = {str(i): {"text": f"option-{text}"} for i, text in enumerate(texts)}
        questionnaire = Questionnaire(
            name=name,
            general_instruction=instruction,
            default_answer_options=options if with_defaults else None,
            instruction_items=[
                {"question": question, "answer_options": options} for question in questions
            ],
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            questionnaire.print_questionnaire()
        printed = buffer.getvalue()
        assert f"Name: {name}" in printed
        assert f"General Instruction: {instruction}" in printed
        assert printed.count("- Question: ") == len(questions)
        for question in questions:
            assert f"- Question: {question}" in printed
        for text in texts:
            expected_copies = len(questions) + (1 if with_defaults else 0)
            shown = re.findall(rf"option-{text}(?![A-Za-z])", printed)  # not "option-AA" for "A"
            assert len(shown) == expected_copies
