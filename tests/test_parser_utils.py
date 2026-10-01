"""Unit tests for ``rupsycho.parsers.parser_utils``.

The helpers in this module do the heavy lifting of the cleaners and judges, so they are tested
with large input/expected tables. Tests marked ``xfail(strict=True)`` describe the *correct*
behaviour of a known defect: they fail today and start failing loudly (XPASS) as soon as the
defect is fixed, at which point the marker has to be removed.
"""

import importlib.machinery
import importlib.util
import json
import re
import sys
import types
import warnings
from pathlib import Path

import pytest

from rupsycho.parsers.parser_utils import (
    check_age,
    check_gender,
    check_multiple_choice_answers,
    check_span,
    mk_age_keywords,
    process_completion,
    prompt_cleaner,
    split_on_symbols,
)

# ---------------------------------------------------------------------------
# ``num2words`` stand-in
# ---------------------------------------------------------------------------
# ``mk_age_keywords`` imports ``num2words`` lazily. The package is not a declared dependency
# (see ``test_num2words_is_declared_as_a_dependency``), so it is missing from CI environments.
# The stand-in below spells the numbers 0-999 exactly like ``num2words.num2words`` does (checked
# against the real library), which is all the age helpers need.

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def _spell_number(n: int) -> str:
    """English cardinal for ``0 <= n < 1000`` (``21`` -> ``twenty-one``, ``101`` -> ``one hundred and one``)."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, units = divmod(n, 10)
        return _TENS[tens] + (f"-{_UNITS[units]}" if units else "")
    hundreds, rest = divmod(n, 100)
    return f"{_UNITS[hundreds]} hundred" + (f" and {_spell_number(rest)}" if rest else "")


@pytest.fixture
def fake_num2words(monkeypatch):
    """Make ``from num2words import num2words`` work (uses the real package when installed)."""
    if importlib.util.find_spec("num2words") is not None:
        return
    module = types.ModuleType("num2words")
    module.num2words = _spell_number  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec("num2words", loader=None)
    monkeypatch.setitem(sys.modules, "num2words", module)


# ===========================================================================
# check_span
# ===========================================================================

SPAN_TEXT = "I choose option 3"  # 17 characters
SPAN_FILTERS = {"number": ["3"], "word": ["choose"], "absent": ["xyz"]}


@pytest.mark.parametrize(
    ("span", "ratio", "number", "word"),
    [
        pytest.param(None, None, True, True, id="whole-text"),
        pytest.param((0, 8), None, False, True, id="span-start"),
        pytest.param((9, 17), None, True, False, id="span-end"),
        pytest.param((5, 100), None, True, False, id="span-end-beyond-the-text"),
        pytest.param((16, 17), None, True, False, id="single-character-span"),
        pytest.param((0, 0), None, False, False, id="empty-span"),
        pytest.param(None, 0.5, False, True, id="first-half"),
        pytest.param(None, -0.5, True, False, id="last-half"),
        pytest.param(None, 1.0, True, True, id="ratio-one"),
        pytest.param(None, -1.0, True, True, id="ratio-minus-one"),
        pytest.param(None, 2.0, True, True, id="ratio-beyond-one"),
        pytest.param(None, 0.01, False, False, id="tiny-positive-ratio-is-empty"),
        pytest.param((0, 8), -0.5, False, True, id="span-wins-over-ratio"),
    ],
)
def test_check_span_windows(span, ratio, number, word):
    result = check_span(SPAN_TEXT, SPAN_FILTERS, span=span, ratio=ratio)
    assert result == {"number": number, "word": word, "absent": False}


@pytest.mark.parametrize(
    ("text", "filters", "expected"),
    [
        pytest.param("I CHOOSE 3", {"w": ["choose"]}, {"w": True}, id="text-is-lowercased"),
        pytest.param("i choose 3", {"w": ["CHOOSE"]}, {"w": True}, id="answers-are-lowercased"),
        pytest.param("yep", {"y": ["yes", "yep"], "n": ["no"]}, {"y": True, "n": False}, id="any"),
        pytest.param("anything", {"empty": []}, {"empty": False}, id="category-without-answers"),
        pytest.param("anything", {}, {}, id="no-categories"),
        pytest.param("", {"a": ["x"]}, {"a": False}, id="empty-text"),
        pytest.param(12345, {"a": ["34"], "b": ["9"]}, {"a": True, "b": False}, id="non-str-text"),
        pytest.param(
            "Grüße 😊", {"a": ["grüße"], "b": ["😊"]}, {"a": True, "b": True}, id="unicode"
        ),
    ],
)
def test_check_span_matching(text, filters, expected):
    assert check_span(text, filters) == expected


@pytest.mark.xfail(
    strict=True,
    reason="check_span: text[-0:] returns the whole text when abs(ratio) * len(text) < 1",
)
@pytest.mark.parametrize("ratio", [-0.01, -0.05])
def test_check_span_tiny_negative_ratio_selects_nothing(ratio):
    # mirrors the (correct) tiny positive ratio above: a window of zero characters is empty
    result = check_span(SPAN_TEXT, SPAN_FILTERS, ratio=ratio)
    assert result == {"number": False, "word": False, "absent": False}


# ===========================================================================
# prompt_cleaner
# ===========================================================================

KNIGHT_PROMPT = "Once upon a time, there was a brave knight."


def test_prompt_cleaner_removes_an_echoed_prompt():
    completion = KNIGHT_PROMPT + " He fought valiantly against the dragon."
    result = prompt_cleaner(KNIGHT_PROMPT, completion)
    assert result == {
        "completion": "he fought valiantly against the dragon.",
        "similarity_score": 0.683,
        "similarity_threshold": 0.5,
    }


@pytest.mark.parametrize("fast", [True, False])
def test_prompt_cleaner_fast_and_exact_ratio_agree_on_a_plain_echo(fast):
    result = prompt_cleaner("Question: what? Answer:", "Question: what? Answer: 3", fast=fast)
    assert result["completion"] == "3"


@pytest.mark.parametrize(
    ("prompt", "completion", "expected", "score"),
    [
        pytest.param("abc", "abc", "", 1.0, id="identical"),
        pytest.param("abc", "  ABC  ", "", 1.0, id="case-and-surrounding-whitespace-ignored"),
        pytest.param("", "", "", 1.0, id="both-empty"),
        pytest.param("abc", "", "", 0.0, id="empty-completion"),
        pytest.param("", "abc", "abc", 0.0, id="empty-prompt"),
        pytest.param("Answer the question", "3", "3", 0.1, id="unrelated-and-much-shorter"),
    ],
)
def test_prompt_cleaner_edge_cases(prompt, completion, expected, score):
    result = prompt_cleaner(prompt, completion)
    assert result["completion"] == expected
    assert result["similarity_score"] == score


def test_prompt_cleaner_keeps_a_completion_below_the_threshold():
    result = prompt_cleaner(
        "Question: what? Answer:", "Question: what? Answer: 3", similarity_threshold=0.99
    )
    # the completion is lower-cased and stripped but nothing is removed
    assert result["completion"] == "question: what? answer: 3"
    assert result["similarity_threshold"] == 0.99


def test_prompt_cleaner_junk_characters_are_ignored_by_the_matcher():
    result = prompt_cleaner("Question: what? Answer:", "Question: what? Answer: 3", junk=" ?:")
    assert result["completion"] == "3"


@pytest.mark.parametrize("threshold", [0, 0.0, -1])
def test_prompt_cleaner_non_positive_threshold_is_replaced_with_a_warning(threshold):
    with pytest.warns(UserWarning, match="similarity_threshold must be greater than 0") as record:
        result = prompt_cleaner(
            "Question: what? Answer:", "Question: what? Answer: 3", similarity_threshold=threshold
        )
    assert result["similarity_threshold"] == 0.001
    assert result["completion"] == "3"
    # stacklevel=2: the warning is attributed to the caller and not to parser_utils.py
    assert record[0].filename == __file__


def test_prompt_cleaner_does_not_warn_for_a_valid_threshold():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prompt_cleaner("abc", "abc", similarity_threshold=0.5)


def test_prompt_cleaner_returns_exactly_three_keys():
    result = prompt_cleaner("a", "b")
    assert set(result) == {"completion", "similarity_score", "similarity_threshold"}


@pytest.mark.xfail(
    strict=True,
    reason="prompt_cleaner(fast=True) strips len(prompt) leading characters of any similar-length text",
)
@pytest.mark.parametrize(
    ("prompt", "completion", "expected"),
    [
        ("Please choose one", "Agree strongly, 5!", "agree strongly, 5!"),
        (
            "Question: what? Answer:",
            "I think the answer is 3, because it is easy.",
            "i think the answer is 3, because it is easy.",
        ),
    ],
)
def test_prompt_cleaner_leaves_completions_that_do_not_echo_the_prompt_intact(
    prompt, completion, expected
):
    assert prompt_cleaner(prompt, completion)["completion"] == expected


# ===========================================================================
# process_completion
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "pattern_name", "expected"),
    [
        ("1. Strongly agree", "numeric_alpha", ["1", ".", "Strongly agree"]),
        ("1, Strongly agree", "numeric_alpha", ["1", ",", "Strongly agree"]),
        ("12.  padded  ", "numeric_alpha", ["12", ".", "padded"]),
        ("1. Never\n2. Always", "numeric_alpha", ["1", ".", "Never", "2", ".", "Always"]),
        ("a. Strongly agree", "alpha_alpha", ["a", ".", "Strongly agree"]),
        ("3. 4", "numeric_numeric", ["3", ".", "4"]),
        ("3.4", "numeric_numeric", ["3", ".", "4"]),
        ("no enumeration here", "numeric_alpha", []),
        ("", "numeric_alpha", []),
        ("1. Strongly agree", "numeric_numeric", []),
    ],
)
def test_process_completion_default_patterns(text, pattern_name, expected):
    assert process_completion(text, pattern_name=pattern_name) == expected


def test_process_completion_unknown_default_pattern():
    with pytest.raises(ValueError, match="'bogus' not found in the default regex dictionary"):
        process_completion("1. x", pattern_name="bogus")


def test_process_completion_requires_some_pattern():
    with pytest.raises(ValueError, match="You must provide either pattern_name"):
        process_completion("1. x")


def test_process_completion_patterns_from_a_json_file(tmp_path):
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps({"key_value": r"(\w+)=(\d+)", "unused": "x"}), encoding="utf-8")

    assert process_completion("a=1 b=22", "key_value", regex_dict_path=str(path)) == [
        "a",
        "1",
        "b",
        "22",
    ]
    assert process_completion("nothing", "key_value", regex_dict_path=str(path)) == []


def test_process_completion_unknown_name_in_a_json_file(tmp_path):
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps({"key_value": r"(\w+)=(\d+)"}), encoding="utf-8")

    with pytest.raises(ValueError, match="'other' not found in the provided regex dictionary"):
        process_completion("a=1", "other", regex_dict_path=str(path))
    with pytest.raises(ValueError, match="'None' not found in the provided regex dictionary"):
        process_completion("a=1", regex_dict_path=str(path))


def test_process_completion_missing_json_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_completion("a=1", "key_value", regex_dict_path=str(tmp_path / "missing.json"))


def test_process_completion_user_pattern_wins_over_everything_else(tmp_path):
    # neither the (bogus) name nor the (missing) file is looked at
    result = process_completion(
        "1. Strongly agree 2. Other",
        pattern_name="bogus",
        regex_dict_path=str(tmp_path / "missing.json"),
        user_input_pattern=r"(\d+)\. (\w+)",
    )
    assert result == ["1", "Strongly", "2", "Other"]


def test_process_completion_user_pattern_without_a_match():
    assert process_completion("1. Strongly agree", user_input_pattern=r"zzz") == []


def test_process_completion_user_pattern_with_a_single_group_and_one_match():
    assert process_completion("choose 12", user_input_pattern=r"(\d+)") == ["12"]


@pytest.mark.xfail(
    strict=True,
    reason="process_completion returns [('1', 'Strongly')] (unflattened tuples) for exactly one match",
)
def test_process_completion_user_pattern_one_match_with_two_groups_is_flat():
    assert process_completion("1. Strongly agree", user_input_pattern=r"(\d+)\. (\w+)") == [
        "1",
        "Strongly",
    ]


@pytest.mark.xfail(
    strict=True,
    reason="process_completion iterates over the characters of every match of a one-group pattern",
)
def test_process_completion_user_pattern_one_group_two_matches():
    assert process_completion("12 and 34", user_input_pattern=r"(\d+)") == ["12", "34"]


# ===========================================================================
# check_multiple_choice_answers
# ===========================================================================

BFI = [
    "1. Disagree strongly",
    "2. Disagree a little",
    "3. Neither agree nor disagree",
    "4. Agree a little",
    "5. Agree strongly",
]


def counts_for(options, hits):
    """Expected result dict: ``hits`` maps option index -> number of detections, the rest is 0."""
    return {option: hits.get(i, 0) for i, option in enumerate(options)}


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        pytest.param("3", {2: 1}, id="bare-number"),
        pytest.param("I choose 3", {2: 1}, id="number-in-a-sentence"),
        pytest.param("I choose 3.", {2: 1}, id="trailing-period"),
        pytest.param("(3)", {2: 1}, id="parentheses"),
        pytest.param("[5]", {4: 1}, id="brackets"),
        pytest.param("5!!!", {4: 1}, id="exclamation-marks"),
        pytest.param("3 3 3", {2: 3}, id="occurrences-are-counted"),
        pytest.param("Agree strongly", {4: 1}, id="option-text"),
        pytest.param("AGREE STRONGLY", {4: 1}, id="option-text-upper-case"),
        pytest.param("i disagree strongly", {0: 1}, id="text-embedded-in-a-sentence"),
        pytest.param("Neither agree nor disagree", {2: 1}, id="long-option-text"),
        pytest.param("3. Neither agree nor disagree", {2: 2}, id="number-and-text-count-twice"),
        pytest.param(
            "The answer is 3. Neither agree nor disagree", {2: 2}, id="prefix-before-both"
        ),
        pytest.param("Disagree a little", {1: 1}, id="disagree-is-not-agree"),
        pytest.param("Agree a little", {3: 1}, id="agree-is-not-disagree"),
        pytest.param("I choose 1 and 2", {0: 1, 1: 1}, id="two-numbers"),
        pytest.param("I choose 11", {}, id="eleven-is-not-one"),
        pytest.param("12345", {}, id="long-number"),
        pytest.param("", {}, id="empty"),
        pytest.param("   \n\t ", {}, id="only-whitespace"),
        pytest.param("no option mentioned", {}, id="nothing-matches"),
        pytest.param("option 3 \U0001f60a", {2: 1}, id="emoji-is-ignored"),
        pytest.param("\U0001f60a\U0001f60a", {}, id="only-emoji"),
        pytest.param("5.", {4: 1}, id="enumeration-with-period"),
        pytest.param("5: agree strongly", {4: 2}, id="enumeration-with-colon"),
    ],
)
def test_check_multiple_choice_answers_bfi(text, hits):
    assert check_multiple_choice_answers(text, BFI) == counts_for(BFI, hits)


@pytest.mark.parametrize(
    ("text", "ignore_case", "hits"),
    [
        pytest.param("AGREE STRONGLY", True, {4: 1}, id="insensitive"),
        pytest.param("AGREE STRONGLY", False, {}, id="sensitive-upper-case-text"),
        pytest.param("agree strongly", False, {}, id="sensitive-lower-case-text"),
        pytest.param("Agree strongly", False, {4: 1}, id="sensitive-exact-case"),
        pytest.param("5", False, {4: 1}, id="numbers-have-no-case"),
    ],
)
def test_check_multiple_choice_answers_ignore_case(text, ignore_case, hits):
    assert check_multiple_choice_answers(text, BFI, ignore_case) == counts_for(BFI, hits)


LETTERS = ["A. Never", "B. Sometimes", "C. Always"]


@pytest.mark.parametrize(
    ("text", "ignore_case", "hits"),
    [
        pytest.param("B", True, {1: 1}, id="letter"),
        pytest.param("b", True, {1: 1}, id="lower-case-letter"),
        pytest.param("b", False, {}, id="lower-case-letter-case-sensitive"),
        pytest.param("B", False, {1: 1}, id="letter-case-sensitive"),
        pytest.param("Answer: B", True, {1: 1}, id="letter-after-a-colon"),
        pytest.param("sometimes", True, {1: 1}, id="text"),
        pytest.param("C. Always", True, {2: 2}, id="letter-and-text"),
        pytest.param("Always", True, {2: 1}, id="only-text"),
    ],
)
def test_check_multiple_choice_answers_letters(text, ignore_case, hits):
    assert check_multiple_choice_answers(text, LETTERS, ignore_case) == counts_for(LETTERS, hits)


@pytest.mark.parametrize("separator", [".", " = ", "=", ": ", ":", " - ", "-", " "])
def test_check_multiple_choice_answers_enumeration_styles(separator):
    options = [f"1{separator}Never", f"2{separator}Sometimes", f"3{separator}Always"]
    assert check_multiple_choice_answers("2", options) == counts_for(options, {1: 1})
    assert check_multiple_choice_answers("sometimes", options) == counts_for(options, {1: 1})
    assert check_multiple_choice_answers("2 sometimes", options) == counts_for(options, {1: 2})


def test_check_multiple_choice_answers_option_that_contains_its_own_punctuation():
    # the model answer is stripped from punctuation as well, so an option matches itself
    options = ["1. Strongly disagree", "3. neutral (neither agree nor disagree)"]
    assert check_multiple_choice_answers(options[1], options) == counts_for(options, {1: 2})


NUMBERS_ONLY = ["1.", "11.", "111."]


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("1", {0: 1}),
        ("11", {1: 1}),
        ("111", {2: 1}),
        ("1 11 111", {0: 1, 1: 1, 2: 1}),
        ("1111", {}),
        ("option 11, definitely", {1: 1}),
    ],
)
def test_check_multiple_choice_answers_options_that_are_prefixes_of_each_other(text, hits):
    assert check_multiple_choice_answers(text, NUMBERS_ONLY) == counts_for(NUMBERS_ONLY, hits)


def test_check_multiple_choice_answers_result_keeps_the_original_options():
    result = check_multiple_choice_answers("3", BFI)
    assert list(result) == BFI


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(
            ["1. a.b", "2. a*b", "3. (a|b)", "4. [x]", "5. a\\b", "6. ^$"], id="regex-syntax"
        ),
        pytest.param(["", " ", "."], id="degenerate"),
        pytest.param([], id="no-options"),
    ],
)
def test_check_multiple_choice_answers_never_fails_on_odd_options(options):
    result = check_multiple_choice_answers("a.b a*b (a|b) [x] 1 2 3", options)
    assert set(result) == set(options)


def test_check_multiple_choice_answers_special_characters_in_options_are_matched_literally():
    options = ["1. 5+ times", "2. [other]"]
    assert check_multiple_choice_answers("5+ times", options) == counts_for(options, {0: 1})
    assert check_multiple_choice_answers("other", options) == counts_for(options, {1: 1})


def test_check_multiple_choice_answers_very_long_text():
    text = "lorem ipsum, dolor sit amet. " * 20_000 + "I pick 5" + " consectetur" * 20_000
    assert check_multiple_choice_answers(text, BFI) == counts_for(BFI, {4: 1})


NEUTRAL_VARIANTS = [
    str.upper,
    str.lower,
    lambda text: f"  {text}  ",
    lambda text: f"\n{text}\t",
    lambda text: f"{text}!!!",
    lambda text: f"...{text}",
]


@pytest.mark.parametrize(
    "text",
    [
        "3",
        "I choose 3",
        "Agree strongly",
        "5: agree strongly",
        "I choose 1 and 2",
        "Neither agree nor disagree",
        "disagree a little",
        "no option mentioned",
        "",
    ],
)
def test_check_multiple_choice_answers_ignores_case_padding_and_edge_punctuation(text):
    expected = check_multiple_choice_answers(text, BFI)

    for variant in NEUTRAL_VARIANTS:
        assert check_multiple_choice_answers(variant(text), BFI) == expected, variant(text)


@pytest.mark.parametrize("text", ["3", "5: agree strongly", "I choose 1 and 2", "nothing", ""])
def test_check_multiple_choice_answers_does_not_depend_on_the_order_of_the_options(text):
    forward = check_multiple_choice_answers(text, BFI)
    backward = check_multiple_choice_answers(text, BFI[::-1])

    assert forward == backward
    assert list(backward) == BFI[::-1]


@pytest.mark.xfail(
    strict=True,
    reason="punctuation is deleted instead of replaced by a space, which glues tokens together",
)
@pytest.mark.parametrize(
    ("text", "hits"),
    [
        pytest.param('{"answer":"3"}', {2: 1}, id="compact-json"),
        pytest.param("answer:3", {2: 1}, id="colon-without-space"),
        pytest.param("Agree  strongly", {4: 1}, id="double-space"),
        pytest.param("Agree\nstrongly", {4: 1}, id="line-break"),
    ],
)
def test_check_multiple_choice_answers_tokens_separated_by_punctuation_or_whitespace(text, hits):
    assert check_multiple_choice_answers(text, BFI) == counts_for(BFI, hits)


# ===========================================================================
# mk_age_keywords / check_age
# ===========================================================================


@pytest.mark.usefixtures("fake_num2words")
class TestMkAgeKeywords:
    def test_has_one_entry_per_age(self):
        assert len(mk_age_keywords()) == 101
        assert len(mk_age_keywords(0)) == 1
        assert len(mk_age_keywords(5)) == 6

    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (0, ["0", "nil", "nought", "oh", "zero"]),
            (1, ["1", "one"]),
            (13, ["13", "thirteen"]),
            (20, ["20", "twenty"]),
            (21, ["21", "twenty one", "twenty-one", "twentyone"]),
            (44, ["44", "forty four", "forty-four", "fortyfour"]),
            (100, ["100", "one hundred"]),
        ],
    )
    def test_spellings_of_an_age(self, age, expected):
        assert mk_age_keywords(100)[age] == expected

    def test_entries_are_ordered_and_start_with_the_digits(self):
        keywords = mk_age_keywords(100)
        assert [entry[0] for entry in keywords] == [str(i) for i in range(101)]

    def test_no_spelling_is_shared_between_two_ages(self):
        seen: dict[str, str] = {}
        for entry in mk_age_keywords(100):
            for form in entry:
                assert form not in seen, f"{form!r} belongs to {seen[form]} and {entry[0]}"
                seen[form] = entry[0]

    def test_ages_above_one_hundred_use_and(self):
        assert mk_age_keywords(120)[120] == ["120", "one hundred and twenty"]


@pytest.mark.usefixtures("fake_num2words")
class TestCheckAge:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("I am 25 years old", "25", id="digits"),
            pytest.param("25", "25", id="bare-digits"),
            pytest.param("Age: 34", "34", id="label"),
            pytest.param("I'm 30.", "30", id="punctuation"),
            pytest.param("I am 0", "0", id="zero-digits"),
            pytest.param("I am zero", "0", id="zero-word"),
            pytest.param("100", "100", id="upper-limit"),
            pytest.param("thirty", "30", id="word"),
            pytest.param("I am twenty-five years old", "25", id="hyphenated-word"),
            pytest.param("Twenty-one", "21", id="capitalised-word"),
            pytest.param("TWENTY-ONE", "21", id="upper-case-word"),
            pytest.param("ninety-nine", "99", id="ninety-nine"),
            pytest.param("born in 1990, age 34", "34", id="years-are-not-ages"),
            pytest.param("I am 18 or 19", "18", id="first-number-wins"),
            pytest.param("Er \U0001f60a 42 \U0001f60a", "42", id="emoji-around-the-number"),
            pytest.param("no number here", "inconclusive", id="no-number"),
            pytest.param("", "inconclusive", id="empty"),
            pytest.param("   ", "inconclusive", id="whitespace"),
            pytest.param("I am 120 years old", "inconclusive", id="above-max-age"),
        ],
    )
    def test_default_max_age(self, text, expected):
        assert check_age(text, 100) == expected

    def test_max_age_limits_the_search(self):
        assert check_age("I am 45", max_age=30) == "inconclusive"
        assert check_age("I am thirty", max_age=30) == "30"
        assert check_age("I am 120", max_age=150) == "120"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Twenty", "inconclusive"),
            ("twenty", "20"),
            ("20", "20"),
            ("TWENTY-ONE", "inconclusive"),
        ],
    )
    def test_case_sensitive_search(self, text, expected):
        assert check_age(text, 100, ignore_case=False) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="check_age compares single words, so multi-word spellings resolve to their first word",
    )
    @pytest.mark.parametrize(
        ("text", "max_age", "expected"),
        [
            pytest.param("I am twenty five years old", 100, "25", id="twenty-five"),
            pytest.param("I am forty four", 100, "44", id="forty-four"),
            pytest.param("one hundred", 100, "100", id="one-hundred"),
            pytest.param("one hundred and twenty", 150, "120", id="one-hundred-and-twenty"),
        ],
    )
    def test_numbers_spelled_with_a_space(self, text, max_age, expected):
        assert check_age(text, max_age) == expected

    @pytest.mark.xfail(strict=True, reason="check_age treats the interjection 'Oh' as the age 0")
    def test_interjection_oh_is_not_an_age(self):
        assert check_age("Oh, I'm 25 years old", 100) == "25"

    @pytest.mark.xfail(
        strict=True,
        reason="punctuation is deleted instead of replaced by a space: 18-year-old -> 18yearold",
    )
    def test_number_glued_to_other_words_by_a_hyphen(self):
        assert check_age("I am an 18-year-old student", 100) == "18"


# ===========================================================================
# check_gender
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("I am a man", "male", id="man"),
        pytest.param("I'm a male", "male", id="male-with-contraction"),
        pytest.param("I am a boy and a guy", "male", id="two-male-keywords"),
        pytest.param("Mr Smith", "male", id="mr"),
        pytest.param("MALE", "male", id="upper-case"),
        pytest.param("I am a woman", "female", id="woman"),
        pytest.param("I am female", "female", id="female-is-not-male"),
        pytest.param("Female.", "female", id="female-with-period"),
        pytest.param("Ms. Smith", "female", id="ms"),
        pytest.param("I'm her", "female", id="pronoun"),
        pytest.param("I am a trans man", "male", id="trans-man"),
        pytest.param("I am a trans woman", "female", id="trans-woman"),
        pytest.param("I identify as non-binary", "other", id="non-binary"),
        pytest.param("I am nonbinary", "other", id="nonbinary"),
        pytest.param("none of your business", "other", id="none"),
        pytest.param("He said she is a girl", "female", id="female-wins-two-to-one"),
        pytest.param("he him his sir said she", "male", id="male-wins-four-to-one"),
        pytest.param("man man man", "male", id="a-keyword-counts-once"),
        pytest.param("man woman", "inconclusive", id="one-each"),
        pytest.param("He and she", "inconclusive", id="pronouns-cancel-out"),
        pytest.param("I don't have a gender", "not present", id="no-keyword"),
        pytest.param("", "not present", id="empty"),
        pytest.param("   ", "not present", id="whitespace"),
        pytest.param("\U0001f60a", "not present", id="emoji"),
        pytest.param("they/them", "not present", id="they-them"),
    ],
)
def test_check_gender(text, expected):
    assert check_gender(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("MALE", "not present"), ("male", "male"), ("Female", "not present"), ("female", "female")],
)
def test_check_gender_case_sensitive(text, expected):
    assert check_gender(text, ignore_case=False) == expected


@pytest.mark.xfail(
    strict=True, reason="punctuation is deleted instead of replaced by a space: she/her -> sheher"
)
@pytest.mark.parametrize(("text", "expected"), [("She/her", "female"), ("he/him", "male")])
def test_check_gender_pronoun_pairs(text, expected):
    assert check_gender(text) == expected


# ===========================================================================
# split_on_symbols
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", ["hello", "world"]),
        ("abc123def", ["abc", "def"]),
        ("a_b-c", ["a", "b", "c"]),
        ("x--y__z!!", ["x", "y", "z"]),
        ("  ", []),
        ("", []),
        ("3.14", []),
        ("über straße 5 \U0001f60a ok", ["über", "straße", "ok"]),
        ("CamelCase stays", ["CamelCase", "stays"]),
    ],
)
def test_split_on_symbols(text, expected):
    assert split_on_symbols(text) == expected


# ===========================================================================
# packaging
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="num2words is imported by mk_age_keywords/check_age but not declared in pyproject.toml",
)
def test_num2words_is_declared_as_a_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not available")
    assert re.search(r"""["']num2words\b""", pyproject.read_text(encoding="utf-8"))
