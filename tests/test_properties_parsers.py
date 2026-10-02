"""Property-based tests of the output parsers (cleaners, validators, judges, parser utils).

Oracle strategy
---------------
Every property compares a parser with one of two things, never with its own code:

* an independent, deliberately naive re-implementation from ``tests/helpers.py`` (a hand-written
  scanner instead of a regular expression, ``str.lstrip().lower()`` instead of ``re.sub``, the
  pinned prefix lists of the validators, a ``decide`` rule for the judges), or
* an algebraic invariant that every sensible implementation must satisfy (idempotence, a cleaner
  only ever deletes, the verdict does not depend on the order of the options, padding the text
  with whitespace or punctuation does not change it).

Texts are generated *structurally* where the contract is about structure: a multiple-choice text
is a list of known words with a known option mentioned in it, so the expected answer is known by
construction. Where the library violates an intended invariant the property is still written for
the correct behaviour and marked ``xfail(strict=True)`` with a ``@example`` that reproduces the bug
deterministically; fixing the bug turns the test into an unexpected pass, which fails the suite
and says "remove the marker".
"""

from __future__ import annotations

import json
import math
import string
import warnings
from difflib import SequenceMatcher

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage

from rupsycho.parsers import parser_utils
from rupsycho.parsers import validators as validators_module
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner
from rupsycho.parsers.judges import DemographicsJudge, ModelBasedAnswerJudge, MultipleChoiceJudge
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
from rupsycho.parsers.validators import (
    ApologiesValidatorParser,
    BeingAiValidatorParser,
    RefusalValidatorParser,
    ValidatorParser,
    normalize,
)

from .helpers import (
    DEFAULT_COMPLETION_PATTERNS,
    GENDER_KEYWORDS,
    GOLDEN_APOLOGIES,
    GOLDEN_BEING_AI,
    GOLDEN_REFUSAL_ANYWHERE,
    GOLDEN_REFUSAL_PREFIXES,
    INCONCLUSIVE,
    NEUTRAL_WORDS,
    NOT_PRESENT,
    PUNCTUATION_WITHOUT_UNDERSCORE,
    REGEX_EXTRACTION_ORACLES,
    TYPOGRAPHIC_TO_ASCII,
    WHITESPACE,
    Option,
    apply_case_mask,
    ascii_lower,
    ascii_upper,
    completion_text,
    decide,
    expected_age_entry,
    naive_basic_clean,
    naive_entropy,
    naive_groups,
    naive_letter_runs,
    naive_normalise,
    naive_number_to_words,
    naive_option_counts,
    naive_span_text,
    naive_verdict,
    naive_words_to_int,
    normalise_for_prompt_cleaner,
)

PROFILE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
LIGHT = settings(PROFILE, max_examples=25)

# =============================================================================
# Strategies
# =============================================================================

TEXT_FRAGMENTS = [
    "a",
    "Z",
    "0",
    "7",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\x0b",
    "\x0c",
    "\x1f",
    ".",
    ",",
    "!",
    "_",
    "-",
    "(",
    ")",
    "'",
    '"',
    "\xe9",
    "\xdf",
    "\U00000130",
    "\U0000017f",
    "\U0001f60a",
    "\xa0",
    "\U00002003",
    "\U0000200b",
    "\U00003000",
    "\U00000663",
    "\U00002019",
    "\U0000201c",
    "\U0000201d",
    "\U00002018",
    "Hello",
    "world",
]

# Arbitrary Unicode and fragment soups full of whitespace, punctuation, emoji and odd letters.
messy_text = st.one_of(
    st.text(max_size=60),
    st.lists(st.sampled_from(TEXT_FRAGMENTS), max_size=24).map("".join),
)
leading_noise = st.lists(st.sampled_from(WHITESPACE), max_size=4).map("".join)
PAD_ALPHABET = "".join(WHITESPACE) + PUNCTUATION_WITHOUT_UNDERSCORE
case_mask = st.lists(st.booleans(), min_size=1, max_size=8)

CURLY_APOSTROPHE = "\U00002019"

_CLEAN_ALPHABET = string.ascii_letters + string.digits + " .,?!"
short_texts = st.text(alphabet=_CLEAN_ALPHABET, min_size=1, max_size=30)


# =============================================================================
# BasicCleaner
# =============================================================================


class TestBasicCleaner:
    @PROFILE
    @given(text=messy_text)
    def test_output_has_the_documented_shape(self, text):
        result = BasicCleaner().parse(text)
        assert result.isascii()
        assert "\n" not in result and "\r" not in result
        assert result == result.strip()
        assert "  " not in result
        assert all(not char.isspace() or char == " " for char in result)

    @PROFILE
    @given(text=messy_text)
    def test_is_idempotent(self, text):
        cleaner = BasicCleaner()
        once = cleaner.parse(text)
        assert cleaner.parse(once) == once

    @PROFILE
    @given(text=messy_text)
    def test_equals_the_naive_cleaner(self, text):
        assert BasicCleaner().parse(text) == naive_basic_clean(text)

    @PROFILE
    @given(text=messy_text)
    def test_only_whitespace_and_non_ascii_characters_are_touched(self, text):
        """Every other character survives, unchanged and in order."""
        straightened = [TYPOGRAPHIC_TO_ASCII.get(char, char) for char in text]
        survivors = [char for char in straightened if char.isascii() and not char.isspace()]
        assert [char for char in BasicCleaner().parse(text) if char != " "] == survivors

    @PROFILE
    @given(
        words=st.lists(st.text(alphabet=string.ascii_letters, min_size=1, max_size=6), max_size=5)
    )
    def test_typographic_quotes_and_no_break_spaces_keep_the_words_apart(self, words):
        """Curly quotes become ASCII quotes and a no-break space still separates words."""
        text = "\xa0".join(f"\U0000201c{word}\U0000201d" for word in words)
        assert BasicCleaner().parse(text) == " ".join(f'"{word}"' for word in words)

    @PROFILE
    @given(words=st.lists(st.text(alphabet=string.printable.strip(), min_size=1), max_size=8))
    def test_tidy_text_is_a_fixed_point(self, words):
        tidy = " ".join(words)
        assert BasicCleaner().parse(tidy) == tidy

    @PROFILE
    @given(text=messy_text)
    def test_invoke_agrees_with_parse(self, text):
        assert BasicCleaner().invoke(text) == BasicCleaner().parse(text)

    @PROFILE
    @given(text=messy_text)
    def test_the_legacy_basic_parser_is_the_same_cleaner(self, text):
        legacy = pytest.importorskip("rupsycho.parsers.parser")
        assert legacy.BasicParser().parse(text) == BasicCleaner().parse(text)


# =============================================================================
# PromptRemovalCleaner / prompt_cleaner
# =============================================================================

thresholds = st.floats(min_value=0.05, max_value=1.0, allow_nan=False)


class TestPromptRemovalCleaner:
    @pytest.mark.parametrize(
        ("prompt", "completion"),
        [("abc", "cde"), ("Once upon a time", "Once upon a time, the end"), ("x", "y")],
    )
    def test_parse_returns_text(self, prompt, completion):
        parser = PromptRemovalCleaner(prompt=prompt)
        assert isinstance(parser.parse(completion), str)
        assert isinstance(parser.invoke(completion), str)

    @PROFILE
    @given(prompt=short_texts, completion=short_texts, threshold=thresholds, fast=st.booleans())
    def test_never_returns_more_than_the_input(self, prompt, completion, threshold, fast):
        cleaned = completion_text(PromptRemovalCleaner(prompt, threshold, fast).parse(completion))
        assert len(cleaned) <= len(normalise_for_prompt_cleaner(completion))

    @PROFILE
    @given(prompt=short_texts, completion=messy_text, threshold=thresholds, fast=st.booleans())
    def test_only_ever_removes_a_prefix(self, prompt, completion, threshold, fast):
        cleaned = completion_text(PromptRemovalCleaner(prompt, threshold, fast).parse(completion))
        assert normalise_for_prompt_cleaner(completion).endswith(cleaned)

    @PROFILE
    @given(
        prompt=st.text(alphabet="abcdefghijklm", min_size=1, max_size=20),
        completion=st.text(alphabet="nopqrstuvwxyz", max_size=20),
        threshold=thresholds,
        fast=st.booleans(),
    )
    def test_texts_without_a_common_character_are_returned_unchanged(
        self, prompt, completion, threshold, fast
    ):
        cleaned = completion_text(PromptRemovalCleaner(prompt, threshold, fast).parse(completion))
        assert cleaned == normalise_for_prompt_cleaner(completion)

    @PROFILE
    @given(
        prompt=st.text(alphabet=_CLEAN_ALPHABET, min_size=1, max_size=30).filter(str.strip),
        separator=st.sampled_from(["", " ", "\n", "  ", ": "]),
        answer=st.text(alphabet=_CLEAN_ALPHABET, max_size=10),
        fast=st.booleans(),
    )
    def test_an_echoed_prompt_is_removed(self, prompt, separator, answer, fast):
        """The use case: the model repeats the prompt and then answers."""
        completion = prompt + separator + answer
        cleaned = completion_text(PromptRemovalCleaner(prompt, 0.01, fast).parse(completion))
        assert cleaned == (separator + answer).lower().strip()

    @PROFILE
    @given(
        prompt=st.text(alphabet="abc", min_size=1, max_size=8),
        completion=st.text(alphabet="abc", max_size=8),
        threshold=st.floats(min_value=0.3, max_value=1.0, allow_nan=False),
    )
    def test_a_completion_below_the_exact_similarity_threshold_is_returned_unchanged(
        self, prompt, completion, threshold
    ):
        """With ``fast=False`` the documented threshold semantics hold."""
        normalised = normalise_for_prompt_cleaner(completion)
        ratio = SequenceMatcher(a=normalise_for_prompt_cleaner(prompt), b=normalised).ratio()
        assume(ratio < threshold)
        cleaned = completion_text(PromptRemovalCleaner(prompt, threshold, False).parse(completion))
        assert cleaned == normalised

    @PROFILE
    @given(
        prompt=st.text(alphabet="abc", min_size=1, max_size=8),
        completion=st.text(alphabet="abc", max_size=8),
        threshold=st.floats(min_value=0.3, max_value=1.0, allow_nan=False),
        fast=st.booleans(),
    )
    @example(prompt="abc", completion="cde", threshold=0.5, fast=True)
    def test_a_dissimilar_completion_is_returned_unchanged(
        self, prompt, completion, threshold, fast
    ):
        normalised = normalise_for_prompt_cleaner(completion)
        ratio = SequenceMatcher(a=normalise_for_prompt_cleaner(prompt), b=normalised).ratio()
        assume(ratio < threshold)
        cleaned = completion_text(PromptRemovalCleaner(prompt, threshold, fast).parse(completion))
        assert cleaned == normalised

    @PROFILE
    @given(
        prompt=short_texts,
        completion=short_texts,
        threshold=st.floats(max_value=0.0, allow_nan=False),
    )
    def test_a_non_positive_threshold_is_replaced_with_a_warning(
        self, prompt, completion, threshold
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = prompt_cleaner(prompt, completion, similarity_threshold=threshold)
        assert result["similarity_threshold"] == 0.001
        assert any("similarity_threshold" in str(w.message) for w in caught)

    @PROFILE
    @given(prompt=messy_text, completion=messy_text, fast=st.booleans())
    def test_reports_a_rounded_similarity_between_zero_and_one(self, prompt, completion, fast):
        result = prompt_cleaner(prompt, completion, fast=fast)
        assert 0.0 <= result["similarity_score"] <= 1.0
        assert result["similarity_score"] == round(result["similarity_score"], 3)


# =============================================================================
# RegexExtractorCleaner
# =============================================================================

search_text = st.lists(
    st.sampled_from(
        ["answer:", "Answer:", "answer: ", " ", "\n", "12", "7", "x", "[", "]", "[4]", "[a b]"]
        + ["abc", "\xe9", "\U00000663", "-", ":", "[\n]"]
    ),
    max_size=14,
).map("".join)


class TestRegexExtractorCleaner:
    @PROFILE
    @given(pattern=st.sampled_from(sorted(REGEX_EXTRACTION_ORACLES)), text=search_text)
    def test_returns_what_a_hand_written_scanner_finds(self, pattern, text):
        found = REGEX_EXTRACTION_ORACLES[pattern](text)
        result = RegexExtractorCleaner(pattern=pattern).parse(text)
        assert result == (text if found is None else found)

    @PROFILE
    @given(pattern=st.sampled_from(sorted(REGEX_EXTRACTION_ORACLES)), text=messy_text)
    def test_the_result_is_always_a_substring_of_the_input(self, pattern, text):
        assert RegexExtractorCleaner(pattern=pattern).parse(text) in text

    @PROFILE
    @given(prefix=st.text(alphabet="xyz ", max_size=5), digits=st.integers(0, 10**6))
    def test_a_pattern_without_a_group_raises_the_documented_error(self, prefix, digits):
        parser = RegexExtractorCleaner(pattern=r"\d+")
        with pytest.raises(OutputParserException):
            parser.parse(f"{prefix}{digits}")
        assert parser.parse(prefix) == prefix  # nothing to extract: the text comes back

    def test_a_group_that_did_not_participate_falls_back_to_the_text(self):
        for pattern, text in ((r"(a)?b", "b"), (r"(?:(x)|y)z", "yz")):
            assert RegexExtractorCleaner(pattern=pattern).parse(text) == text, pattern


# =============================================================================
# Validators
# =============================================================================

VALIDATORS = {
    "apologies": (ApologiesValidatorParser, validators_module.APOLOGIES_HINTS, ()),
    "being_ai": (BeingAiValidatorParser, validators_module.BEING_AI_HINTS, ()),
    "refusal": (
        RefusalValidatorParser,
        validators_module.REFUSAL_HINTS,
        validators_module.CATCH_ALL_HINTS,
    ),
}
"""name -> (validator class, prefix hints, anywhere hints) as the library currently defines them."""

_ALL_PREFIXES = sorted({h for _, hints, _ in VALIDATORS.values() for h in hints})
printable_text = st.text(alphabet=string.printable, max_size=40)
candidate_text = st.one_of(
    printable_text,
    st.builds(
        lambda noise, hint, tail: noise + hint + tail,
        st.sampled_from(["", "", "x ", "okay. "]),
        st.sampled_from(_ALL_PREFIXES),
        printable_text,
    ),
)
validator_names = st.sampled_from(sorted(VALIDATORS))


class TestValidators:
    def test_the_documented_prefix_lists_are_still_present(self):
        """Hints may be added, but a documented one must not vanish unnoticed."""
        assert set(GOLDEN_APOLOGIES) <= set(validators_module.APOLOGIES_HINTS)
        assert set(GOLDEN_BEING_AI) <= set(validators_module.BEING_AI_HINTS)
        assert set(GOLDEN_REFUSAL_PREFIXES) <= set(validators_module.REFUSAL_HINTS)
        assert set(GOLDEN_REFUSAL_ANYWHERE) <= set(validators_module.CATCH_ALL_HINTS)

    @PROFILE
    @given(text=messy_text)
    def test_normalize_strips_leading_whitespace_and_lower_cases(self, text):
        assert normalize(text) == naive_normalise(text)
        assert normalize(normalize(text)) == normalize(text)

    @PROFILE
    @given(text=candidate_text, name=validator_names, curly=st.booleans())
    def test_each_validator_equals_the_naive_verdict(self, text, name, curly):
        text = text.replace("'", CURLY_APOSTROPHE) if curly else text
        validator_class, prefixes, anywhere = VALIDATORS[name]
        verdict = naive_verdict(text, prefixes, anywhere)
        assert validator_class().parse(text) == {
            "text": text,
            "validation_status": verdict,
            "details": {name: verdict == "invalid"},
        }

    @PROFILE
    @given(text=candidate_text, curly=st.booleans())
    def test_the_combined_validator_is_the_conjunction_of_the_three(self, text, curly):
        text = text.replace("'", CURLY_APOSTROPHE) if curly else text
        verdicts = {
            name: naive_verdict(text, prefixes, anywhere)
            for name, (_, prefixes, anywhere) in VALIDATORS.items()
        }
        result = ValidatorParser().parse(text)
        assert result["text"] == text
        assert result["details"] == {name: v == "invalid" for name, v in verdicts.items()}
        assert result["validation_status"] in ("valid", "invalid")
        assert result["validation_status"] == (
            "valid" if all(v == "valid" for v in verdicts.values()) else "invalid"
        )

    @PROFILE
    @given(
        name=validator_names,
        data=st.data(),
        noise=leading_noise,
        mask=case_mask,
        tail=printable_text,
        curly=st.booleans(),
    )
    def test_a_known_prefix_always_invalidates(self, name, data, noise, mask, tail, curly):
        """Also with a typographic apostrophe, which models and word processors produce."""
        validator_class, prefixes, _ = VALIDATORS[name]
        hint = data.draw(st.sampled_from(prefixes))
        hint = hint.replace("'", CURLY_APOSTROPHE) if curly else hint
        text = noise + apply_case_mask(hint, mask) + tail
        assert validator_class().parse(text)["validation_status"] == "invalid"
        assert ValidatorParser().parse(text)["validation_status"] == "invalid"

    @PROFILE
    @given(data=st.data(), before=printable_text, after=printable_text, mask=case_mask)
    def test_a_catch_all_phrase_invalidates_anywhere_in_the_text(self, data, before, after, mask):
        phrase = data.draw(st.sampled_from(validators_module.CATCH_ALL_HINTS))
        text = before + apply_case_mask(phrase, mask) + after
        assert RefusalValidatorParser().parse(text)["validation_status"] == "invalid"
        assert ValidatorParser().parse(text)["details"]["refusal"] is True

    @PROFILE
    @given(
        digit=st.sampled_from(string.digits),
        rest=st.text(alphabet=string.digits + " .,:;-()", max_size=20),
        noise=leading_noise,
    )
    def test_numeric_answers_are_always_valid(self, digit, rest, noise):
        text = noise + digit + rest
        assert ValidatorParser().parse(text)["validation_status"] == "valid"

    @PROFILE
    @given(
        number=st.integers(1, 9),
        words=st.lists(st.sampled_from(["agree", "disagree", "strongly", "neutral"]), max_size=3),
    )
    def test_numbered_likert_answers_are_valid(self, number, words):
        text = f"{number}. {' '.join(words)}"
        assert ValidatorParser().parse(text)["validation_status"] == "valid"

    @PROFILE
    @given(text=candidate_text, noise=leading_noise, mask=case_mask)
    def test_leading_whitespace_and_ascii_case_never_change_the_verdict(self, text, noise, mask):
        parser = ValidatorParser()
        expected = parser.parse(text)["validation_status"]
        assert parser.parse(noise + text)["validation_status"] == expected
        assert parser.parse(apply_case_mask(text, mask))["validation_status"] == expected
        assert parser.parse(ascii_upper(text))["validation_status"] == expected
        assert parser.parse(ascii_lower(text))["validation_status"] == expected

    @PROFILE
    @given(text=candidate_text)
    def test_invoke_agrees_with_parse(self, text):
        assert ValidatorParser().invoke(text) == ValidatorParser().parse(text)


# =============================================================================
# MultipleChoiceJudge and check_multiple_choice_answers
# =============================================================================

OPTION_WORDS = (
    "agree",
    "disagree",
    "strongly",
    "little",
    "neutral",
    "neither",
    "nor",
    "somewhat",
    "often",
    "rarely",
    "never",
    "always",
    "sometimes",
    "mostly",
    "hardly",
)
assert not set(OPTION_WORDS) & set(NEUTRAL_WORDS)  # the filler must never form an option

LABELS = (*range(1, 10), *"abcde")
"""Enumerations an option can carry: a number or a single letter ('3. ...', 'b. ...')."""

options_st = st.builds(
    Option,
    number=st.one_of(st.none(), st.sampled_from(LABELS)),
    phrase=st.lists(st.sampled_from(OPTION_WORDS), min_size=1, max_size=3).map(tuple),
)
option_lists = st.lists(options_st, min_size=1, max_size=5, unique_by=Option.render)


@st.composite
def nested_option_lists(draw) -> list[Option]:
    """Options whose phrases contain one another ("agree" / "strongly agree" / "agree a little")."""
    phrases = [tuple(draw(st.lists(st.sampled_from(OPTION_WORDS), min_size=1, max_size=3)))]
    for _ in range(draw(st.integers(1, 3))):
        base = draw(st.sampled_from(phrases))
        if len(base) > 1 and draw(st.booleans()):  # a contiguous part of an existing phrase
            start = draw(st.integers(0, len(base) - 1))
            phrase = base[start : draw(st.integers(start + 1, len(base)))]
        else:  # an existing phrase with words added in front or behind
            extra = tuple(draw(st.lists(st.sampled_from(OPTION_WORDS), min_size=1, max_size=2)))
            phrase = (*extra, *base) if draw(st.booleans()) else (*base, *extra)
        if phrase not in phrases:
            phrases.append(phrase)
    labelled = draw(st.booleans())
    labels = draw(st.permutations(LABELS))
    return [
        Option(labels[index] if labelled else None, phrase) for index, phrase in enumerate(phrases)
    ]


@st.composite
def disjoint_options(draw, min_size: int = 2, max_size: int = 4) -> list[Option]:
    """Options that share no word and no label, all labelled or all unlabelled."""
    size = draw(st.integers(min_size, max_size))
    words = draw(st.permutations(OPTION_WORDS))
    labels = draw(st.permutations(LABELS))
    labelled = draw(st.booleans())
    options, cursor = [], 0
    for index in range(size):
        length = draw(st.integers(1, 2))
        phrase = tuple(words[cursor : cursor + length])
        cursor += length
        options.append(Option(labels[index] if labelled else None, phrase))
    return options


@st.composite
def scenarios(draw) -> tuple[list[Option], str, list[str], list[list[bool]]]:
    """Options plus a sentence built from their own words, labels and whole phrases.

    Punctuation is attached to the *ends* of tokens only, so deleting it leaves single spaces
    between the words and the token list is exactly what the judge should see.

    Returns:
        ``(options, text, tokens, option_masks)``; the masks give each option a random case when
        the judge ignores case.
    """
    options = draw(st.one_of(option_lists, nested_option_lists()))
    chunks: list[tuple[str, ...]] = [(word,) for option in options for word in option.phrase]
    chunks += [option.phrase for option in options]
    chunks += [(str(option.number),) for option in options if option.number is not None]
    chunks += [(word,) for word in NEUTRAL_WORDS[:2]]
    picked = draw(st.lists(st.sampled_from(chunks), max_size=6))
    pieces, tokens = [], []
    for token in (token for chunk in picked for token in chunk):
        shown = apply_case_mask(token, draw(case_mask))
        opening = draw(st.sampled_from(["", "", "(", "*", '"']))
        closing = draw(st.sampled_from(["", "", ",", ".", "!", ")", ":", "*"]))
        pieces.append(opening + shown + closing)
        tokens.append(shown)
    masks = [draw(case_mask) for _ in options]
    return options, " ".join(pieces), tokens, masks


def shown_options(options, masks, ignore_case):
    """The option strings handed to the judge (mixed case only if the judge ignores case)."""
    if not ignore_case:
        return [option.render() for option in options]
    return [apply_case_mask(option.render(), mask) for option, mask in zip(options, masks)]


class TestMultipleChoiceJudge:
    @PROFILE
    @given(scenario=scenarios(), ignore_case=st.booleans())
    def test_counts_equal_the_token_oracle(self, scenario, ignore_case):
        options, text, tokens, masks = scenario
        given_options = shown_options(options, masks, ignore_case)
        counts = check_multiple_choice_answers(text, given_options, ignore_case)
        scores = naive_option_counts(tokens, options, ignore_case=ignore_case)
        assert counts == dict(zip(given_options, scores))

    @PROFILE
    @given(scenario=scenarios(), ignore_case=st.booleans())
    def test_the_verdict_is_the_unique_maximum_of_the_counts(self, scenario, ignore_case):
        options, text, tokens, masks = scenario
        given_options = shown_options(options, masks, ignore_case)
        judge = MultipleChoiceJudge(given_options, ignore_case=ignore_case)
        scores = naive_option_counts(tokens, options, ignore_case=ignore_case)
        assert judge.parse(text) == decide(dict(zip(given_options, scores)))

    @PROFILE
    @given(
        options=st.one_of(
            st.lists(options_st, min_size=1, max_size=5, unique_by=lambda o: o.phrase),
            nested_option_lists(),
        ),
        ignore_case=st.booleans(),
        data=st.data(),
    )
    def test_every_option_identifies_itself_even_if_options_overlap(
        self, options, ignore_case, data
    ):
        """The text of an option is judged as that option (the 'self-match' of the judge).

        Options that differ only in their label ("agree" / "1. agree") are ambiguous by nature
        and excluded.
        """
        rendered = [option.render() for option in options]
        judge = MultipleChoiceJudge(rendered, ignore_case=ignore_case)
        chosen = data.draw(st.sampled_from(rendered))
        assert judge.parse(chosen) == chosen

    @PROFILE
    @given(
        options=st.lists(st.text(max_size=12), min_size=1, max_size=5, unique=True),
        text=messy_text,
        ignore_case=st.booleans(),
    )
    def test_the_verdict_is_an_option_or_a_documented_sentinel(self, options, text, ignore_case):
        verdict = MultipleChoiceJudge(options, ignore_case=ignore_case).parse(text)
        assert verdict in {*options, NOT_PRESENT, INCONCLUSIVE}

    @PROFILE
    @given(
        options=disjoint_options(),
        data=st.data(),
        before=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=4),
        after=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=4),
        mask=case_mask,
        ignore_case=st.booleans(),
    )
    def test_a_text_containing_exactly_one_option_returns_it(
        self, options, data, before, after, mask, ignore_case
    ):
        chosen = data.draw(st.sampled_from(options))
        shown = apply_case_mask(chosen.render(), mask) if ignore_case else chosen.render()
        text = " ".join([*before, shown, *after])
        judge = MultipleChoiceJudge([o.render() for o in options], ignore_case=ignore_case)
        assert judge.parse(text) == chosen.render()

    @PROFILE
    @given(
        options=disjoint_options(),
        data=st.data(),
        filler=st.lists(st.sampled_from(NEUTRAL_WORDS), min_size=1, max_size=4),
    )
    def test_the_label_or_the_phrase_alone_identifies_the_option(self, options, data, filler):
        chosen = data.draw(st.sampled_from(options))
        mentions = [" ".join(chosen.phrase)]
        if chosen.number is not None:
            mentions.append(str(chosen.number))
        judge = MultipleChoiceJudge([o.render() for o in options])
        for mention in mentions:
            assert judge.parse(" ".join([*filler, mention])) == chosen.render()

    @PROFILE
    @given(options=disjoint_options(), filler=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=6))
    def test_a_text_without_any_option_is_not_present(self, options, filler):
        judge = MultipleChoiceJudge([o.render() for o in options])
        assert judge.parse(" ".join(filler)) == NOT_PRESENT

    @PROFILE
    @given(options=disjoint_options(), data=st.data())
    def test_two_equally_good_options_are_inconclusive(self, options, data):
        first, second = data.draw(st.permutations(options))[:2]
        text = f"{first.render()} {second.render()}"
        judge = MultipleChoiceJudge([o.render() for o in options])
        assert judge.parse(text) == INCONCLUSIVE

    @PROFILE
    @given(scenario=scenarios(), data=st.data())
    def test_the_verdict_does_not_depend_on_the_order_of_the_options(self, scenario, data):
        options, text, tokens, _ = scenario
        rendered = [option.render() for option in options]
        expected = decide(dict(zip(rendered, naive_option_counts(tokens, options))))
        order = data.draw(st.permutations(range(len(options))))
        judge = MultipleChoiceJudge(["decoy option"])  # the per-call options must win
        assert judge.parse(text, [rendered[i] for i in order]) == expected
        assert judge.parse(text, rendered) == expected

    @PROFILE
    @given(
        options=disjoint_options(),
        data=st.data(),
        filler=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=3),
        left=st.text(alphabet=PAD_ALPHABET, max_size=6),
        right=st.text(alphabet=PAD_ALPHABET, max_size=6),
    )
    def test_padding_with_whitespace_and_punctuation_changes_nothing(
        self, options, data, filler, left, right
    ):
        chosen = data.draw(st.sampled_from(options))
        text = " ".join([*filler, chosen.render()])
        judge = MultipleChoiceJudge([o.render() for o in options])
        assert judge.parse(left + text + right) == judge.parse(text) == chosen.render()

    @PROFILE
    @given(scenario=scenarios(), mask=case_mask)
    def test_ignoring_case_makes_the_verdict_case_independent(self, scenario, mask):
        options, text, _, masks = scenario
        judge = MultipleChoiceJudge(shown_options(options, masks, True), ignore_case=True)
        assert judge.parse(apply_case_mask(text, mask)) == judge.parse(text)

    def test_without_options_the_judge_raises_the_documented_error(self):
        with pytest.raises(OutputParserException):
            MultipleChoiceJudge([]).parse("anything")
        with pytest.raises(OutputParserException):
            MultipleChoiceJudge(["1. yes"]).parse(None)  # type: ignore[arg-type]

    def test_an_underscore_is_ignored_like_other_punctuation(self):
        judge = MultipleChoiceJudge(["1. agree little", "2. never"])
        for wrapper in ("_{}_", "_{}", "{}_"):
            assert judge.parse(wrapper.format("agree little")) == "1. agree little", wrapper

    @PROFILE
    @given(
        gaps=st.lists(
            st.text(alphabet="".join(WHITESPACE) + "-/,;", min_size=1, max_size=3),
            min_size=2,
            max_size=2,
        )
    )
    @example(gaps=["  ", " "])
    @example(gaps=["\n", " "])
    @example(gaps=[" - ", " "])
    def test_layout_between_the_words_of_an_option_changes_nothing(self, gaps):
        judge = MultipleChoiceJudge(["1. agree little often", "2. never"])
        text = f"agree{gaps[0]}little{gaps[1]}often"
        assert judge.parse(text) == "1. agree little often"

    @PROFILE
    @given(
        modifier=st.sampled_from(OPTION_WORDS),
        base=st.sampled_from(OPTION_WORDS),
        padding=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=2),
    )
    @example(modifier="strongly", base="agree", padding=[])
    def test_an_option_equal_to_the_answer_wins_even_if_it_contains_another_option(
        self, modifier, base, padding
    ):
        assume(modifier != base)
        longer, shorter = f"{modifier} {base}", base
        judge = MultipleChoiceJudge([longer, shorter])
        assert judge.parse(" ".join([*padding, longer])) == longer
        assert judge.parse(" ".join([*padding, shorter])) == shorter


# =============================================================================
# DemographicsJudge: gender and age
# =============================================================================

_ALL_GENDER_WORDS = [word for words in GENDER_KEYWORDS.values() for word in words]
assert not set(_ALL_GENDER_WORDS) & set(NEUTRAL_WORDS)


@st.composite
def gender_sentences(draw, lower_only: bool = False):
    """A sentence mixing neutral words with chosen gender keywords.

    Returns:
        ``(text, chosen)`` where ``chosen`` maps each category to the distinct keywords used.
    """
    chosen = {
        category: draw(st.lists(st.sampled_from(words), max_size=3, unique=True))
        for category, words in GENDER_KEYWORDS.items()
    }
    pieces = list(draw(st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=5)))
    for words in chosen.values():
        for word in words:
            shown = word if lower_only else apply_case_mask(word, draw(case_mask))
            closing = draw(st.sampled_from(["", "", ",", ".", "!", "?"]))
            pieces.append(shown + closing)
    return " ".join(draw(st.permutations(pieces))), chosen


class TestGender:
    @PROFILE
    @given(sentence=gender_sentences())
    def test_the_category_with_most_distinct_keywords_wins(self, sentence):
        text, chosen = sentence
        expected = decide({category: len(words) for category, words in chosen.items()})
        assert check_gender(text) == expected

    @PROFILE
    @given(sentence=gender_sentences(lower_only=True))
    def test_case_sensitive_matching_agrees_on_lower_case_text(self, sentence):
        text, chosen = sentence
        expected = decide({category: len(words) for category, words in chosen.items()})
        assert check_gender(text, ignore_case=False) == expected

    @PROFILE
    @given(sentence=gender_sentences(), mask=case_mask)
    def test_the_verdict_is_case_independent_when_case_is_ignored(self, sentence, mask):
        text, _ = sentence
        assert check_gender(apply_case_mask(text, mask)) == check_gender(text)

    @PROFILE
    @given(sentence=gender_sentences(lower_only=True))
    def test_upper_case_keywords_are_not_recognised_when_case_matters(self, sentence):
        text, chosen = sentence
        assume(any(chosen.values()))
        assert check_gender(ascii_upper(text), ignore_case=False) == NOT_PRESENT
        assert check_gender(ascii_upper(text), ignore_case=True) == check_gender(text)

    @PROFILE
    @given(text=messy_text)
    def test_the_verdict_is_always_one_of_five_values(self, text):
        assert check_gender(text) in {"male", "female", "other", NOT_PRESENT, INCONCLUSIVE}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I am non-binary.", "other"),
            ("Transgender man here", "male"),
            ("trans woman", "female"),
        ],
    )
    def test_hyphens_and_two_word_terms(self, text, expected):
        assert check_gender(text) == expected

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "check_gender's docstring promises to count the *occurrences* of related words, but "
            "each keyword counts at most once: 'he he she' scores male 1 / female 1 and is "
            "'inconclusive' instead of 'male'. Fix: sum words.count(keyword) per category (or "
            "correct the docstring)."
        ),
    )
    def test_repeated_keywords_count_every_occurrence(self):
        assert check_gender("he he she") == "male"

    @PROFILE
    @given(text=messy_text, ignore_case=st.booleans())
    def test_the_judge_delegates_to_check_gender(self, text, ignore_case):
        judge = DemographicsJudge(ignore_case=ignore_case)
        assert judge.parse(text, ["gender"]) == check_gender(text, ignore_case=ignore_case)

    @pytest.mark.parametrize("answers", [None, ["height"], ["Gender"], []])
    def test_unknown_question_types_raise_the_documented_error(self, answers):
        with pytest.raises(OutputParserException):
            DemographicsJudge().parse("I am a man", answers)


FRAMES = ("I am {} years old.", "{}", "Honestly, {} I guess", "My age: {}!")
ages = st.integers(0, 100)


def number_to_words(number):
    """``parser_utils.number_to_words``, looked up on use so a rename fails these tests only."""
    return parser_utils.number_to_words(number)


class TestNumberSpelling:
    def test_number_to_words_equals_the_naive_spelling_for_every_supported_number(self):
        for number in range(1000):
            assert number_to_words(number) == naive_number_to_words(number)

    def test_the_independent_parser_reads_every_spelling_back(self):
        for number in range(1000):
            spelling = number_to_words(number)
            for variant in (spelling, spelling.replace("-", " "), spelling.replace("-", "")):
                assert naive_words_to_int(variant) == number, variant

    @PROFILE
    @given(number=st.one_of(st.integers(max_value=-1), st.integers(min_value=1000)))
    def test_numbers_outside_the_supported_range_fail_loudly(self, number):
        with pytest.raises(ValueError):
            number_to_words(number)


class TestAge:
    def test_the_documented_examples_of_the_keyword_table(self):
        table = mk_age_keywords(44)
        assert table[0] == ["0", "nil", "nought", "oh", "zero"]
        assert table[44] == ["44", "forty four", "forty-four", "fortyfour"]

    @PROFILE
    @given(max_age=st.integers(0, 130))
    def test_the_keyword_table_lists_every_age_in_every_spelling(self, max_age):
        table = mk_age_keywords(max_age)
        assert table == [expected_age_entry(age) for age in range(max_age + 1)]
        for age, forms in enumerate(table):
            assert all(naive_words_to_int(form) == age for form in forms[1:]), forms

    def test_every_spelling_of_every_age_is_found(self):
        """Exhaustive: digits, words with spaces, hyphens, run together (but not 'oh')."""
        for age, forms in enumerate(mk_age_keywords(100)):
            for index, form in enumerate(forms):
                if form == "oh":
                    continue  # an interjection, see test_the_interjection_oh_is_not_an_age
                text = FRAMES[index % len(FRAMES)].format(form)
                assert check_age(text, max_age=100) == str(age), (age, text)

    @PROFILE
    @given(age=ages, data=st.data(), frame=st.sampled_from(FRAMES), mask=case_mask)
    def test_a_stated_age_is_found_in_any_case(self, age, data, frame, mask):
        forms = [form for form in mk_age_keywords(100)[age] if form != "oh"]
        spelling = apply_case_mask(data.draw(st.sampled_from(forms)), mask)
        assert check_age(frame.format(spelling), max_age=100) == str(age)

    def test_the_interjection_oh_is_not_an_age(self):
        assert check_age("Oh, I see what you mean", max_age=100) == INCONCLUSIVE
        assert check_age("Oh, I am 25", max_age=100) == "25"

    @PROFILE
    @given(
        age=ages,
        symbol=st.sampled_from(string.punctuation),
        tail=st.sampled_from(["year{0}old", "years", "{0}ish", "{0}"]),
    )
    def test_symbols_separate_words(self, age, symbol, tail):
        text = f"{age}{symbol}" + tail.format(symbol)
        assert check_age(text, max_age=100) == str(age)

    @PROFILE
    @given(
        first=ages,
        second=ages,
        spelling=st.booleans(),
        glue=st.sampled_from(["and", "or", "then", "but"]),
    )
    def test_the_first_number_in_the_text_decides(self, first, second, spelling, glue):
        assume(first != second)
        shown = mk_age_keywords(100)[first][-1] if spelling else str(first)
        assert check_age(f"{shown} {glue} {second}", max_age=100) == str(first)

    @PROFILE
    @given(words=st.lists(st.sampled_from(NEUTRAL_WORDS), max_size=6))
    def test_text_without_a_number_is_inconclusive(self, words):
        assert check_age(" ".join(words), max_age=100) == INCONCLUSIVE

    @PROFILE
    @given(max_age=st.integers(0, 120), age=st.integers(0, 250))
    def test_ages_above_the_limit_are_not_recognised(self, max_age, age):
        expected = str(age) if age <= max_age else INCONCLUSIVE
        assert check_age(f"I am {age} years old", max_age=max_age) == expected

    @PROFILE
    @given(age=ages, max_age=st.integers(0, 100), ignore_case=st.booleans())
    def test_the_judge_uses_its_settings_and_delegates_to_check_age(
        self, age, max_age, ignore_case
    ):
        text = f"I am {age}"
        judge = DemographicsJudge(max_age=max_age, ignore_case=ignore_case)
        expected = check_age(text, max_age=max_age, ignore_case=ignore_case)
        assert judge.parse(text, ["age"]) == expected

    @PROFILE
    @given(age=ages, frame=st.sampled_from(FRAMES), data=st.data())
    def test_case_matters_only_if_it_is_not_ignored(self, age, frame, data):
        words = [form for form in mk_age_keywords(100)[age][1:] if form != "oh"]
        assume(words)
        shouting = ascii_upper(data.draw(st.sampled_from(words)))
        assert check_age(frame.format(shouting), 100, ignore_case=True) == str(age)
        assert check_age(frame.format(shouting), 100, ignore_case=False) == INCONCLUSIVE

    def test_the_age_judge_raises_the_documented_error_without_a_question_type(self):
        with pytest.raises(OutputParserException):
            DemographicsJudge().parse("I am 30")


# =============================================================================
# Parsers inside chains
# =============================================================================

TEXT_PARSERS = {
    "BasicCleaner": lambda: BasicCleaner(),
    "PromptRemovalCleaner": lambda: PromptRemovalCleaner(prompt="Q: what is it?"),
    "RegexExtractorCleaner": lambda: RegexExtractorCleaner(pattern=r"([0-9]+)"),
    "ApologiesValidatorParser": lambda: ApologiesValidatorParser(),
    "BeingAiValidatorParser": lambda: BeingAiValidatorParser(),
    "RefusalValidatorParser": lambda: RefusalValidatorParser(),
    "ValidatorParser": lambda: ValidatorParser(),
    "MultipleChoiceJudge": lambda: MultipleChoiceJudge(["1. agree", "2. disagree"]),
}
parser_names = st.sampled_from(sorted(TEXT_PARSERS))


class TestParsersInChains:
    """The run loop puts a parser behind the model: it must read model output like plain text."""

    @PROFILE
    @given(name=parser_names, text=candidate_text)
    def test_a_chat_message_is_parsed_like_its_text(self, name, text):
        parser = TEXT_PARSERS[name]()
        assert parser.invoke(AIMessage(content=text)) == parser.parse(text)

    @LIGHT
    @given(name=parser_names, text=candidate_text)
    def test_a_parser_behind_a_model_returns_what_parse_returns(self, name, text):
        parser = TEXT_PARSERS[name]()
        chain = FakeListLLM(responses=[text]) | parser
        assert chain.invoke("question") == parser.parse(text)

    @PROFILE
    @given(name=parser_names, bad=st.sampled_from([None, 5, 1.5, ["a"], {"a": 1}]))
    def test_input_that_is_not_text_raises_the_documented_error(self, name, bad):
        with pytest.raises(OutputParserException):
            TEXT_PARSERS[name]().parse(bad)


# =============================================================================
# Other parser_utils helpers
# =============================================================================


FILE_PATTERNS = {
    "numeric_alpha": r"(\d+)(:)(.+)",  # same name as a built-in pattern, different meaning
    "pair": r"([a-z]+)=([a-z]+)",
    "digits": r"(\d+)",
}
"""The patterns of the JSON file below; the file must win over the built-in names."""


@pytest.fixture(scope="module")
def regex_dictionary(tmp_path_factory):
    """A JSON file mapping pattern names to patterns."""
    path = tmp_path_factory.mktemp("regex") / "patterns.json"
    path.write_text(json.dumps(FILE_PATTERNS), encoding="utf-8")
    return str(path)


class TestProcessCompletion:
    @PROFILE
    @given(
        number=st.integers(0, 10**6),
        separator=st.sampled_from([".", ","]),
        answer=st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=20).filter(
            str.strip
        ),
    )
    def test_numeric_alpha_splits_into_number_separator_and_answer(self, number, separator, answer):
        text = f"{number}{separator}{answer}"
        expected = [str(number), separator, answer.strip()]
        assert process_completion(text, pattern_name="numeric_alpha") == expected

    @PROFILE
    @given(
        first=st.integers(0, 999),
        second=st.integers(0, 999),
        separator=st.sampled_from([".", ","]),
        gap=st.text(alphabet=" \t", max_size=3),
    )
    def test_numeric_numeric_splits_into_two_numbers(self, first, second, separator, gap):
        text = f"{first}{separator}{gap}{second}"
        expected = [str(first), separator, str(second)]
        assert process_completion(text, pattern_name="numeric_numeric") == expected

    @PROFILE
    @given(
        word=st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
        separator=st.sampled_from([".", ","]),
        answer=st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=12).filter(
            str.strip
        ),
    )
    def test_alpha_alpha_splits_into_word_separator_and_answer(self, word, separator, answer):
        text = f"{word}{separator}{answer}"
        assert process_completion(text, pattern_name="alpha_alpha") == [
            word,
            separator,
            answer.strip(),
        ]

    @PROFILE
    @given(
        name=st.sampled_from(sorted(DEFAULT_COMPLETION_PATTERNS)),
        text=st.lists(
            st.sampled_from(["1", "23", ".", ",", " ", "a", "bc", "\n", "x y", "7, 8"]),
            max_size=10,
        ).map("".join),
    )
    def test_named_patterns_equal_the_naive_group_extraction(self, name, text):
        expected = naive_groups(text, DEFAULT_COMPLETION_PATTERNS[name])
        assert process_completion(text, pattern_name=name) == expected

    @PROFILE
    @given(
        name=st.sampled_from(sorted(FILE_PATTERNS)),
        text=st.lists(
            st.sampled_from(["1", "23", ":", "=", " ", "a", "bc", "\n", "x y"]), max_size=10
        ).map("".join),
    )
    def test_a_pattern_file_overrides_the_built_in_names(self, name, text, regex_dictionary):
        result = process_completion(text, pattern_name=name, regex_dict_path=regex_dictionary)
        assert result == naive_groups(text, FILE_PATTERNS[name])

    def test_missing_or_unknown_patterns_raise_value_errors(self, regex_dictionary):
        with pytest.raises(ValueError):
            process_completion("1. a")
        with pytest.raises(ValueError):
            process_completion("1. a", pattern_name="nope")
        with pytest.raises(ValueError):
            process_completion("1. a", pattern_name="nope", regex_dict_path=regex_dictionary)

    @PROFILE
    @given(
        pattern=st.sampled_from([r"(\d+)([.,])(.+)", r"(\d+)", r"\d+", r"([a-z]+)-([a-z]+)"]),
        text=st.lists(
            st.sampled_from(["1", "23", ".", ",", " ", "a", "bc", "-", "x y"]), max_size=10
        ).map("".join),
    )
    @example(pattern=r"(\d+)([.,])(.+)", text="1. agree")
    @example(pattern=r"(\d+)", text="12 and 34")
    def test_a_user_pattern_gives_the_groups_of_every_match(self, pattern, text):
        assert process_completion(text, user_input_pattern=pattern) == naive_groups(text, pattern)

    def test_a_one_group_pattern_from_a_file_gives_whole_matches(self, regex_dictionary):
        found = process_completion(
            "12 and 34", pattern_name="digits", regex_dict_path=regex_dictionary
        )
        assert found == ["12", "34"]


class TestCheckSpan:
    @PROFILE
    @given(
        text=messy_text,
        categories=st.dictionaries(
            st.sampled_from(["a", "b", "c"]),
            st.lists(st.text(alphabet="abHeloW ", min_size=1, max_size=4), max_size=3),
            max_size=3,
        ),
    )
    def test_without_span_or_ratio_the_whole_text_is_searched(self, text, categories):
        expected = {
            category: any(answer.lower() in text.lower() for answer in answers)
            for category, answers in categories.items()
        }
        assert check_span(text, categories) == expected

    @PROFILE
    @given(
        text=st.text(alphabet="abHeloW ", max_size=20),
        answers=st.lists(
            st.text(alphabet="abHeloW ", min_size=1, max_size=3), min_size=1, max_size=3
        ),
        start=st.integers(0, 25),
        length=st.integers(0, 25),
    )
    def test_a_span_restricts_the_search_to_that_slice(self, text, answers, start, length):
        span = (start, start + length)
        expected = any(a.lower() in naive_span_text(text, span=span).lower() for a in answers)
        assert check_span(text, {"x": answers}, span=span) == {"x": expected}

    @PROFILE
    @given(
        text=st.text(alphabet="abHeloW ", max_size=20),
        answers=st.lists(
            st.text(alphabet="abHeloW ", min_size=1, max_size=3), min_size=1, max_size=3
        ),
        ratio=st.floats(min_value=0.05, max_value=1.0, allow_nan=False),
    )
    def test_a_positive_ratio_searches_the_beginning(self, text, answers, ratio):
        expected = any(a.lower() in naive_span_text(text, ratio=ratio).lower() for a in answers)
        assert check_span(text, {"x": answers}, ratio=ratio) == {"x": expected}

    @PROFILE
    @given(
        text=st.text(alphabet="abHeloW ", max_size=20),
        answers=st.lists(
            st.text(alphabet="abHeloW ", min_size=1, max_size=3), min_size=1, max_size=3
        ),
        ratio=st.floats(min_value=0.05, max_value=1.0, allow_nan=False),
    )
    @example(text="hello world", answers=["hello"], ratio=0.05)
    def test_a_negative_ratio_searches_the_end(self, text, answers, ratio):
        expected = any(a.lower() in naive_span_text(text, ratio=-ratio).lower() for a in answers)
        assert check_span(text, {"x": answers}, ratio=-ratio) == {"x": expected}


class TestSplitOnSymbols:
    @PROFILE
    @given(text=st.text(alphabet=string.printable, max_size=40))
    def test_ascii_text_is_split_into_its_letter_runs(self, text):
        assert split_on_symbols(text) == naive_letter_runs(text)


class TestEntropy:
    @PROFILE
    @given(weights=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), max_size=8))
    def test_calculate_entropy_equals_shannon_entropy_of_the_normalised_weights(self, weights):
        decisions = [{"positive_probability": weight} for weight in weights]
        entropy = ModelBasedAnswerJudge.calculate_entropy(None, decisions)  # type: ignore[arg-type]
        assert entropy == pytest.approx(naive_entropy(weights), abs=1e-9)
        assert -1e-9 <= entropy <= math.log2(max(1, len(weights))) + 1e-9

    @PROFILE
    @given(size=st.integers(1, 12), weight=st.floats(min_value=0.001, max_value=1.0))
    def test_equal_weights_have_the_maximum_entropy(self, size, weight):
        decisions = [{"positive_probability": weight}] * size
        entropy = ModelBasedAnswerJudge.calculate_entropy(None, decisions)  # type: ignore[arg-type]
        assert entropy == pytest.approx(math.log2(size), abs=1e-9)
