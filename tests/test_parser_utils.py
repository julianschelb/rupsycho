"""Unit tests for ``rupsycho.parsers.parser_utils``.

The helpers in this module do the heavy lifting of the cleaners and judges, so they are tested
with large input/expected tables. Tests marked ``xfail(strict=True)`` describe the *correct*
behaviour of a known defect: they fail today and start failing loudly (XPASS) as soon as the
defect is fixed, at which point the marker has to be removed.
"""

import builtins
import json
import re
import sys
import warnings

import pytest

from rupsycho.parsers.parser_utils import (
    check_age,
    check_gender,
    check_multiple_choice_answers,
    check_span,
    mk_age_keywords,
    number_to_words,
    process_completion,
    prompt_cleaner,
    split_on_symbols,
)

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


@pytest.mark.parametrize("ratio", [-0.01, -0.05])
def test_check_span_tiny_negative_ratio_selects_nothing(ratio):
    # mirrors the tiny positive ratio above: a window of zero characters is empty (and must not
    # fall back to ``text[-0:]``, which is the whole text)
    result = check_span(SPAN_TEXT, SPAN_FILTERS, ratio=ratio)
    assert result == {"number": False, "word": False, "absent": False}


WINDOW_TEXT = "0123456789ABCDEFGHIJ"  # 20 characters, all different


@pytest.mark.parametrize(
    ("ratio", "window", "one_character_more"),
    [
        pytest.param(0.25, "01234", "012345", id="first-quarter"),
        pytest.param(0.5, "0123456789", "0123456789A", id="first-half"),
        pytest.param(0.75, "0123456789ABCDE", "0123456789ABCDEF", id="first-three-quarters"),
        pytest.param(-0.25, "FGHIJ", "EFGHIJ", id="last-quarter"),
        pytest.param(-0.5, "ABCDEFGHIJ", "9ABCDEFGHIJ", id="last-half"),
        pytest.param(-0.75, "56789ABCDEFGHIJ", "456789ABCDEFGHIJ", id="last-three-quarters"),
    ],
)
def test_check_span_ratio_window_has_exactly_the_requested_size(ratio, window, one_character_more):
    # a positive ratio keeps the beginning of the text, a negative one its end; one character
    # more than the window does not fit into it
    result = check_span(
        WINDOW_TEXT, {"window": [window], "too_long": [one_character_more]}, ratio=ratio
    )
    assert result == {"window": True, "too_long": False}


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


# Completions that merely share their first characters (or a few words) with the prompt. They
# have about the length of the prompt: a decision based on the lengths alone (the quick ratio)
# would cut ``len(prompt)`` characters off them.
NOT_AN_ECHO = [
    pytest.param("Question: what?", "Quite sure it is 3", "quite sure it is 3", id="shared-start"),
    pytest.param("Answer the question", "A: three", "a: three", id="answer-vanishes"),
    pytest.param("Rate yourself", "Really? I'd say 4", "really? i'd say 4", id="shared-letter"),
    pytest.param(
        "Choose a number",
        "Certainly 3 because I am cautious",
        "certainly 3 because i am cautious",
        id="longer-completion",
    ),
]


@pytest.mark.parametrize("fast", [True, False], ids=["quick-ratio", "exact-ratio"])
@pytest.mark.parametrize(("prompt", "completion", "expected"), NOT_AN_ECHO)
def test_prompt_cleaner_keeps_completions_that_are_not_an_echo(prompt, completion, expected, fast):
    assert prompt_cleaner(prompt, completion, fast=fast)["completion"] == expected


def test_prompt_cleaner_cuts_where_the_echo_ends_in_the_completion():
    echoes = {
        # (prompt, completion): the answer that follows the echoed prompt
        ("Question: what? Answer:", "Question: what is it? Answer: 3"): "3",
        ("Rate yourself.  Choose 1 to 5.  Answer:", "Rate yourself. Choose 1 to 5. Answer: 4"): "4",
        ("Question: what?\n\nAnswer:", "Question: what?\nAnswer: 3"): "3",
    }

    cleaned = {
        pair: prompt_cleaner(*pair, similarity_threshold=0.7)["completion"] for pair in echoes
    }

    assert cleaned == echoes


ECHO_AND_OTHER_PAIRS = [
    pytest.param("Question: what? Answer:", "Question: what? Answer: 3", id="echo"),
    pytest.param("Question: what? Answer:", "Question: what is it? Answer: 3", id="edited-echo"),
    pytest.param("Question: what? Answer:", "Answer: 3", id="echo-of-the-tail"),
    pytest.param("Question: what? Answer:", "I think the answer is 3.", id="no-echo"),
    pytest.param("abcdef", "x", id="much-shorter"),
    pytest.param("abcdef", "azzzzz", id="same-length-same-first-letter"),
    pytest.param("", "", id="both-empty"),
    pytest.param("abc", "", id="empty-completion"),
    pytest.param(KNIGHT_PROMPT, KNIGHT_PROMPT + " He fought valiantly.", id="long-echo"),
]


@pytest.mark.parametrize(("prompt", "completion"), ECHO_AND_OTHER_PAIRS)
def test_prompt_cleaner_quick_mode_only_shortcuts_the_exact_decision(prompt, completion):
    quick = prompt_cleaner(prompt, completion, fast=True)
    exact = prompt_cleaner(prompt, completion, fast=False)

    # the decision to strip is exact either way ...
    assert quick["completion"] == exact["completion"]
    # ... the quick score is an upper bound of the exact one, and equal to it whenever the bound
    # reaches the threshold (only then the exact value is computed)
    assert quick["similarity_score"] >= exact["similarity_score"]
    if quick["similarity_score"] >= quick["similarity_threshold"]:
        assert quick["similarity_score"] == exact["similarity_score"]


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


def test_process_completion_reads_the_pattern_file_as_utf8_whatever_the_locale(
    tmp_path, monkeypatch
):
    path = tmp_path / "patterns.json"
    path.write_bytes(json.dumps({"umlaut": "(ü+)"}, ensure_ascii=False).encode("utf-8"))
    real_open = builtins.open

    def cp1252_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        # what open() does on a Windows machine when no encoding is given
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", cp1252_open)

    assert process_completion("grüße üü", "umlaut", regex_dict_path=str(path)) == ["ü", "üü"]


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


def test_process_completion_user_pattern_one_match_with_two_groups_is_flat():
    assert process_completion("1. Strongly agree", user_input_pattern=r"(\d+)\. (\w+)") == [
        "1",
        "Strongly",
    ]


def test_process_completion_user_pattern_one_group_two_matches():
    assert process_completion("12 and 34", user_input_pattern=r"(\d+)") == ["12", "34"]


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        pytest.param("12 and 34", r"\d+", ["12", "34"], id="no-group-whole-matches"),
        pytest.param("a=1 b=22", r"(\w)=(\d+)", ["a", "1", "b", "22"], id="two-groups-two-matches"),
        pytest.param("x=1", r"(\w)=(\d+)", ["x", "1"], id="two-groups-one-match"),
        pytest.param("  12  ", r"(\s*\d+\s*)", ["12"], id="items-are-stripped"),
        pytest.param("a", r"(a)(b)?", ["a", ""], id="optional-group-that-did-not-match"),
    ],
)
def test_process_completion_flattens_matches_of_any_shape(text, pattern, expected):
    assert process_completion(text, user_input_pattern=pattern) == expected


def test_process_completion_invalid_user_pattern():
    with pytest.raises(re.error):
        process_completion("text", user_input_pattern="([")


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


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        pytest.param('{"answer":"3"}', {2: 1}, id="compact-json"),
        pytest.param("answer:3", {2: 1}, id="colon-without-space"),
        pytest.param("Agree  strongly", {4: 1}, id="double-space"),
        pytest.param("Agree\nstrongly", {4: 1}, id="line-break"),
        pytest.param("Agree\t\tstrongly", {4: 1}, id="tabs"),
        pytest.param("Agree,strongly", {4: 1}, id="comma-between-the-words"),
        pytest.param("Agree - strongly", {4: 1}, id="dash-between-the-words"),
        pytest.param("agree (strongly)", {4: 1}, id="parentheses-inside-the-option"),
        pytest.param("Agree... strongly", {4: 1}, id="ellipsis-inside-the-option"),
    ],
)
def test_check_multiple_choice_answers_tokens_separated_by_punctuation_or_whitespace(text, hits):
    assert check_multiple_choice_answers(text, BFI) == counts_for(BFI, hits)


def test_check_multiple_choice_answers_punctuation_never_glues_tokens_together():
    # "5.agree" must not become the single token "5agree" (which would match nothing)
    assert check_multiple_choice_answers("5.agree strongly", BFI) == counts_for(BFI, {4: 2})
    assert check_multiple_choice_answers("1,2", BFI) == counts_for(BFI, {0: 1, 1: 1})
    assert check_multiple_choice_answers("1/5", BFI) == counts_for(BFI, {0: 1, 4: 1})


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        pytest.param("answer_3", {2: 1}, id="number-after-an-underscore"),
        pytest.param("3_answer", {2: 1}, id="number-before-an-underscore"),
        pytest.param("agree_strongly", {4: 1}, id="underscore-inside-an-option"),
        pytest.param("__5__", {4: 1}, id="number-between-underscores"),
        pytest.param("choice_1_or_2", {0: 1, 1: 1}, id="snake-case-sentence"),
    ],
)
def test_check_multiple_choice_answers_underscores_separate_tokens(text, hits):
    # "_" counts as a word character for regular expressions, so "answer_3" has no word boundary
    assert check_multiple_choice_answers(text, BFI) == counts_for(BFI, hits)


def test_check_multiple_choice_answers_options_with_underscores_match_their_own_text():
    options = ["1. snake_case", "2. camelCase"]

    assert check_multiple_choice_answers("snake_case", options) == counts_for(options, {0: 1})
    assert check_multiple_choice_answers("snake case", options) == counts_for(options, {0: 1})
    assert check_multiple_choice_answers("SNAKE-CASE", options) == counts_for(options, {0: 1})


# Options whose text is contained in the text of another option
LIKERT = ["1. Strongly disagree", "2. Disagree", "3. Neutral", "4. Agree", "5. Strongly agree"]


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        pytest.param("Strongly agree", {4: 1}, id="strongly-agree"),
        pytest.param("Strongly disagree", {0: 1}, id="strongly-disagree"),
        pytest.param("agree", {3: 1}, id="agree"),
        pytest.param("disagree", {1: 1}, id="disagree"),
        pytest.param("I strongly agree!", {4: 1}, id="sentence-with-punctuation"),
        pytest.param("STRONGLY   AGREE", {4: 1}, id="shouting-with-extra-spaces"),
        pytest.param("5. Strongly agree", {4: 2}, id="number-and-text-of-the-longer-option"),
        pytest.param("4. Agree", {3: 2}, id="number-and-text-of-the-shorter-option"),
        pytest.param("agree and strongly agree", {3: 1, 4: 1}, id="both-mentioned-separately"),
        pytest.param("strongly agree, agree", {3: 1, 4: 1}, id="longer-first-then-shorter"),
        pytest.param("agree, strongly agree", {3: 1, 4: 1}, id="shorter-first-then-longer"),
        pytest.param("strongly agree strongly agree", {4: 2}, id="longer-option-twice"),
        pytest.param("disagree agree", {1: 1, 3: 1}, id="two-neighbours"),
    ],
)
def test_check_multiple_choice_answers_overlapping_options(text, hits):
    # a hit inside the longer match of another option counts for that option only
    assert check_multiple_choice_answers(text, LIKERT) == counts_for(LIKERT, hits)


def test_check_multiple_choice_answers_overlapping_options_do_not_depend_on_their_order():
    for text in ("Strongly agree", "agree", "agree, strongly disagree"):
        forward = check_multiple_choice_answers(text, LIKERT)
        backward = check_multiple_choice_answers(text, LIKERT[::-1])
        assert forward == backward  # dictionaries compare without regard to their order


def test_check_multiple_choice_answers_options_with_the_same_text_both_count():
    options = ["1. yes", "2. yes"]
    assert check_multiple_choice_answers("yes", options) == {"1. yes": 1, "2. yes": 1}


def test_check_multiple_choice_answers_apostrophes_in_options_are_matched_like_in_the_text():
    options = ["1. Don't know", "2. Sure"]
    assert check_multiple_choice_answers("I don't know", options) == counts_for(options, {0: 1})
    assert check_multiple_choice_answers("DON'T KNOW!", options) == counts_for(options, {0: 1})


# ===========================================================================
# mk_age_keywords / check_age
# ===========================================================================


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

    @pytest.mark.parametrize(
        ("text", "max_age", "expected"),
        [
            pytest.param("I am twenty five years old", 100, "25", id="twenty-five"),
            pytest.param("I am forty four", 100, "44", id="forty-four"),
            pytest.param("one hundred", 100, "100", id="one-hundred"),
            pytest.param("one hundred and twenty", 150, "120", id="one-hundred-and-twenty"),
            pytest.param("one hundred and one", 150, "101", id="one-hundred-and-one"),
            pytest.param("TWENTY  FIVE", 100, "25", id="shouting-with-two-spaces"),
            pytest.param("twenty\nfive", 100, "25", id="line-break-between-the-words"),
            pytest.param("twenty, five", 100, "25", id="comma-between-the-words"),
            pytest.param("twenty or thirty", 100, "20", id="first-number-wins"),
            pytest.param("twentyfive", 100, "25", id="written-together"),
        ],
    )
    def test_numbers_spelled_with_a_space(self, text, max_age, expected):
        assert check_age(text, max_age) == expected

    def test_the_longest_spelling_wins_over_its_first_word(self):
        # "twenty" is an age, but "twenty five" is the longer match
        assert check_age("twenty five", 100) == "25"
        assert check_age("twenty", 100) == "20"
        assert check_age("twenty fiver", 100) == "20"  # "fiver" is not "five"

    def test_interjection_oh_is_not_an_age(self):
        assert check_age("Oh, I'm 25 years old", 100) == "25"

    @pytest.mark.parametrize("text", ["Oh", "oh!", "OH my", "Oh, well"])
    def test_interjection_oh_alone_is_inconclusive(self, text):
        assert check_age(text, 100) == "inconclusive"

    @pytest.mark.parametrize("text", ["nil", "nought", "zero", "0"])
    def test_other_spellings_of_zero_are_still_the_age_zero(self, text):
        assert check_age(text, 100) == "0"

    def test_number_glued_to_other_words_by_a_hyphen(self):
        assert check_age("I am an 18-year-old student", 100) == "18"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I'm 25/26", "25"),
            ("age=34", "34"),
            ("(42)", "42"),
            ("thirty-year-old", "30"),
            ("twenty-five-year-old", "25"),
        ],
    )
    def test_symbols_separate_words(self, text, expected):
        assert check_age(text, 100) == expected

    def test_long_spellings_of_ages_above_120(self):
        spelled = {
            ("one hundred and twenty-one", 150): "121",
            ("one hundred and thirty four", 150): "134",
            ("nine hundred and ninety-nine", 999): "999",
        }

        found = {key: check_age(*key) for key in spelled}

        assert found == spelled


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
        pytest.param("man man man", "male", id="repeated-keyword"),
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She/her", "female"),
        ("he/him", "male"),
        ("she / her", "female"),
        ("I use he/him/his pronouns", "male"),
        ("(she/her)", "female"),
        ("she,her", "female"),
        ("they/them", "not present"),
        ("she/they", "female"),
        ("he/she", "inconclusive"),
    ],
)
def test_check_gender_pronoun_pairs(text, expected):
    # symbols other than hyphens and apostrophes separate the words: "she/her" is not "sheher"
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
# number_to_words: the age helpers need no third-party package
# ===========================================================================


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (1, "one"),
        (9, "nine"),
        (10, "ten"),
        (11, "eleven"),
        (13, "thirteen"),
        (19, "nineteen"),
        (20, "twenty"),
        (21, "twenty-one"),
        (40, "forty"),
        (44, "forty-four"),
        (99, "ninety-nine"),
        (100, "one hundred"),
        (101, "one hundred and one"),
        (110, "one hundred and ten"),
        (120, "one hundred and twenty"),
        (342, "three hundred and forty-two"),
        (900, "nine hundred"),
        (999, "nine hundred and ninety-nine"),
    ],
)
def test_number_to_words(number, words):
    assert number_to_words(number) == words


def test_number_to_words_has_one_distinct_spelling_per_number():
    spellings = [number_to_words(n) for n in range(1000)]

    assert len(set(spellings)) == 1000
    assert all(re.fullmatch(r"[a-z]+([ -][a-z]+)*", spelling) for spelling in spellings)


def test_number_to_words_agrees_with_the_num2words_package():
    # the library used to provide this; it is no longer needed but is the reference spelling
    num2words = pytest.importorskip("num2words").num2words

    assert [number_to_words(n) for n in range(1000)] == [num2words(n) for n in range(1000)]


@pytest.mark.parametrize("number", [-1, 1000, 12345])
def test_number_to_words_rejects_numbers_outside_zero_to_999(number):
    with pytest.raises(ValueError, match="0-999"):
        number_to_words(number)


def test_the_age_helpers_do_not_import_num2words(monkeypatch):
    # ``None`` in sys.modules makes ``import num2words`` raise ImportError
    monkeypatch.setitem(sys.modules, "num2words", None)

    assert mk_age_keywords(21)[21] == ["21", "twenty one", "twenty-one", "twentyone"]
    assert check_age("I am twenty-one", 100) == "21"
    assert check_age("I am 21", 100) == "21"
