"""Tests of ``rupsycho.scoring``: from judged answers to item scores and scale scores.

How the tests are organised
---------------------------
* A **hand-computed oracle**: a small Big-Five-like questionnaire with reverse-keyed items, three
  respondents and every number written down by hand (item scores, scale means and sums,
  ``n_items`` / ``n_missing``, the overall total).
* **Parametrised rules**: every rule of ``docs/tutorials/scoring.md`` is a test case (weights that
  are not 1..n, reverse keying with ignored extremes, the options of an item vs. the questionnaire
  defaults, duplicate texts, case and white space, sentinels and missing values,
  ``valid == False``, dtypes, empty frames, dimensions, ordering, aggregation, errors).
* **Properties** (``hypothesis``): reverse keying is the same as scoring the flipped weights, the
  vectorised scorer agrees with a plain row-by-row oracle that is written down here, and scale
  scores obey their invariants (counts add up, the order of the rows does not matter).
* **Integration**: a fake-LLM experiment is run with ``CSVCallback``, post-processed with the real
  ``PostprocessingPipeline`` and scored; so is the example that ships with the package.
* **Performance** (200 000 rows) and the **documentation**: the examples in the docstrings and the
  code of the tutorial are executed and the output they print is compared with the output shown.

Everything runs offline; the only model is a ``FakeListLLM``.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import io
import json
import math
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
import rupsycho.scoring as scoring
from rupsycho.callbacks import CSVCallback
from rupsycho.models.questionnaire import Questionnaire
from rupsycho.parsers.cleaners import BasicCleaner
from rupsycho.parsers.judges import MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import PostprocessingPipeline
from rupsycho.scoring import (
    DEFAULT_GROUPS,
    JUDGE_SENTINELS,
    TOTAL_SCALE,
    scale_scores,
    score_answers,
    score_experiment,
)

# Unexpected warnings (pandas' FutureWarnings too, which the project's pytest settings ignore) are
# errors here. The tests that run experiments silence the libraries they call (``quietly``).
pytestmark = pytest.mark.filterwarnings(
    "error", "ignore:Models of type .* cannot be seeded:UserWarning"
)

REPO = Path(__file__).resolve().parents[1]
TUTORIAL = REPO / "docs" / "tutorials" / "scoring.md"
API_PAGE = REPO / "docs" / "api" / "scoring.md"
NAN = math.nan

PROPERTIES = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

LIKERT_TEXTS = [
    "1. Disagree strongly",
    "2. Disagree a little",
    "3. Neither agree nor disagree",
    "4. Agree a little",
    "5. Agree strongly",
]
LIKERT = {str(i): {"text": text, "weight": i} for i, text in enumerate(LIKERT_TEXTS, start=1)}


def option(text: str, weight: int, ignored: bool = False) -> dict[str, Any]:
    """One answer option in its configuration form."""
    return {"text": text, "weight": weight, "ignored_for_scale": ignored}


def options(*entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Answer options in the configuration form ``{id: option}``."""
    return {str(index): entry for index, entry in enumerate(entries, start=1)}


def weighted(weights: list[int], ignored: tuple[int, ...] = ()) -> dict[str, dict[str, Any]]:
    """The options ``opt0``, ``opt1``, ... with the given weights (``ignored`` are positions)."""
    return options(*(option(f"opt{i}", w, i in ignored) for i, w in enumerate(weights)))


def item(
    question: str = "Question",
    *,
    reversed: bool = False,
    answer_options: dict[str, Any] | None = None,
    dimension: Any = None,
) -> dict[str, Any]:
    """One instruction item in its configuration form."""
    spec: dict[str, Any] = {"question": question, "reversed": reversed}
    if answer_options is not None:
        spec["answer_options"] = answer_options
    if dimension is not None:
        spec["attributes"] = {"dimension": dimension}
    return spec


def questionnaire(
    items: list[dict[str, Any]],
    *,
    defaults: dict[str, Any] | None = None,
    dimensions: Any = None,
) -> Questionnaire:
    """A questionnaire; ``dimensions`` is what is stored in ``attributes["dimension"]``."""
    data: dict[str, Any] = {
        "name": "Test questionnaire",
        "general_instruction": "Rate the statement.",
        "instruction_items": items,
    }
    if defaults is not None:
        data["default_answer_options"] = defaults
    if dimensions is not None:
        data["attributes"] = {"dimension": dimensions}
    return Questionnaire.model_validate(data)


def single_item(
    weights: list[int], *, reversed: bool = False, ignored: tuple[int, ...] = ()
) -> Questionnaire:
    """A questionnaire of one item with the options ``opt0``, ``opt1``, ..."""
    return questionnaire([item(reversed=reversed, answer_options=weighted(weights, ignored))])


def answers(decisions: Any, item_ids: Any = 0, **columns: Any) -> pd.DataFrame:
    """A post-processed frame with one row per decision (one item id, or one per row)."""
    count = len(decisions)
    ids = [item_ids] * count if isinstance(item_ids, int) else item_ids
    return pd.DataFrame({"instruction_item_id": ids, "decision": decisions, **columns})


def assert_scores(result: pd.DataFrame, expected: list[float]) -> None:
    """The ``score`` column equals ``expected`` exactly (``NaN`` equals ``NaN``)."""
    np.testing.assert_array_equal(result["score"].to_numpy(), np.asarray(expected, dtype=float))


def dimension_list(result: pd.DataFrame) -> list[str | None]:
    """The ``dimension`` column with every kind of missing value as ``None``."""
    return [None if pd.isna(value) else value for value in result["dimension"]]


@contextlib.contextmanager
def quietly():
    """Silence the warnings of the libraries that build and run experiments (not of scoring)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def build_experiment(**data: Any) -> rup.ExperimentDocument:
    """An experiment from keyword arguments; no model is configured unless ``models`` says so."""
    with quietly():
        return rup.ExperimentDocument(**{"models": {}, **data})


def experiment_from_config(config: dict) -> rup.ExperimentDocument:
    """An experiment from a configuration dictionary (which is left unchanged)."""
    with quietly():
        return rup.experiment_from_dict(copy.deepcopy(config))


@pytest.fixture
def likert_item() -> Questionnaire:
    """One item that uses the default options 1..5."""
    return questionnaire([item()], defaults=LIKERT)


# ===========================================================================
# the public surface
# ===========================================================================


def test_the_module_exports_its_public_names():
    assert sorted(scoring.__all__) == [
        "DEFAULT_GROUPS",
        "JUDGE_SENTINELS",
        "TOTAL_SCALE",
        "scale_scores",
        "score_answers",
        "score_experiment",
    ]
    assert DEFAULT_GROUPS == ("model_id", "profile_id", "random_seed")
    assert JUDGE_SENTINELS == ("not present", "inconclusive")
    assert TOTAL_SCALE == "total"


@pytest.mark.parametrize("name", ["score_answers", "scale_scores", "score_experiment"])
def test_public_functions_are_documented_with_the_google_sections(name):
    doc = inspect.getdoc(getattr(scoring, name))

    for section in ("Args:", "Returns:", "Raises:", "Example:"):
        assert section in doc, f"{name} lacks {section}"


def test_scoring_is_a_sub_package_of_the_package():
    assert rup.scoring is scoring
    assert "scoring" in dir(rup)


def test_importing_the_module_does_not_load_the_model_stack():
    code = (
        "import sys, rupsycho.scoring\n"
        "heavy = [m for m in ('langchain_core', 'pydantic', 'torch', 'transformers') "
        "if m in sys.modules]\n"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# ===========================================================================
# the oracle: a mini Big Five inventory, computed by hand
# ===========================================================================

MINI_ITEMS = [
    item("Is talkative", dimension="1"),
    item("Is reserved", reversed=True, dimension="1"),
    item("Is helpful to others", dimension="2"),
    item("Starts quarrels", reversed=True, dimension="2"),
    item("Is relaxed", reversed=True, dimension=3),  # an integer id on purpose
    item("Gets nervous easily", dimension="3"),
]
MINI_DIMENSIONS = {"1": "Extraversion", "2": "Agreeableness", "3": "Neuroticism"}

# (model, profile, seed) -> (decision, valid) for each of the six items
RESPONDENTS: dict[tuple[str, str, str], list[tuple[str, bool]]] = {
    ("m1", "p1", "1"): [
        ("5. Agree strongly", True),
        ("1. Disagree strongly", True),
        ("4. Agree a little", True),
        ("2. Disagree a little", True),
        ("3. Neither agree nor disagree", True),
        ("not present", True),
    ],
    ("m1", "p2", "1"): [
        ("1. Disagree strongly", True),
        ("5. Agree strongly", True),
        ("2. Disagree a little", True),
        ("4. Agree a little", False),  # a refusal the validator flagged
        ("5. Agree strongly", True),
        ("5. Agree strongly", True),
    ],
    ("m2", "p1", "1"): [
        ("inconclusive", True),
        ("inconclusive", True),
        ("inconclusive", True),
        ("inconclusive", True),
        ("inconclusive", True),
        (" 4. agree a little ", True),  # white space and case do not matter
    ],
}
# Item scores by hand. Items 1, 3 and 4 are reverse-keyed on a 1-5 scale: w becomes 6 - w.
#   m1/p1: 5, 6-1=5, 4, 6-2=4, 6-3=3, - (not present)
#   m1/p2: 1, 6-5=1, 2, - (the answer is invalid), 6-5=1, 5
#   m2/p1: -, -, -, -, -, 4
EXPECTED_ITEM_SCORES = {
    ("m1", "p1", "1"): [5, 5, 4, 4, 3, NAN],
    ("m1", "p2", "1"): [1, 1, 2, NAN, 1, 5],
    ("m2", "p1", "1"): [NAN, NAN, NAN, NAN, NAN, 4],
}


@pytest.fixture
def mini_bfi() -> Questionnaire:
    return questionnaire(MINI_ITEMS, defaults=LIKERT, dimensions=MINI_DIMENSIONS)


@pytest.fixture
def mini_processed() -> pd.DataFrame:
    """The answers of the three respondents in the order of a run (items, then respondents)."""
    rows = []
    for position in range(len(MINI_ITEMS)):
        for (model, profile, seed), decisions in RESPONDENTS.items():
            decision, valid = decisions[position]
            rows.append((position, model, profile, seed, decision, valid))
    columns = ["instruction_item_id", "model_id", "profile_id", "random_seed", "decision", "valid"]
    return pd.DataFrame(rows, columns=columns)


def scale_frame(rows: list[tuple[Any, ...]]) -> pd.DataFrame:
    """The expected output of ``scale_scores`` for the three respondents."""
    columns = [
        "model_id",
        "profile_id",
        "random_seed",
        "dimension",
        "score",
        "n_items",
        "n_missing",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.astype({"score": "float64", "n_items": "int64", "n_missing": "int64"})


def test_oracle_item_scores(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)

    for (model, profile, seed), expected in EXPECTED_ITEM_SCORES.items():
        respondent = scored[
            (scored["model_id"] == model)
            & (scored["profile_id"] == profile)
            & (scored["random_seed"] == seed)
        ]
        assert respondent["instruction_item_id"].tolist() == list(range(6))
        assert_scores(respondent, expected)


def test_oracle_dimension_of_every_item(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)

    by_item = dict(zip(scored["instruction_item_id"], scored["dimension"]))
    assert by_item == {
        0: "Extraversion",
        1: "Extraversion",
        2: "Agreeableness",
        3: "Agreeableness",
        4: "Neuroticism",  # the item says 3, an integer; the mapping has the key "3"
        5: "Neuroticism",
    }


def test_oracle_scale_scores_mean(mini_bfi, mini_processed):
    scales = scale_scores(score_answers(mini_processed, mini_bfi))

    expected = scale_frame(
        [
            ("m1", "p1", "1", "Extraversion", 5.0, 2, 0),
            ("m1", "p1", "1", "Agreeableness", 4.0, 2, 0),
            ("m1", "p1", "1", "Neuroticism", 3.0, 1, 1),
            ("m1", "p2", "1", "Extraversion", 1.0, 2, 0),
            ("m1", "p2", "1", "Agreeableness", 2.0, 1, 1),
            ("m1", "p2", "1", "Neuroticism", 3.0, 2, 0),
            ("m2", "p1", "1", "Extraversion", NAN, 0, 2),
            ("m2", "p1", "1", "Agreeableness", NAN, 0, 2),
            ("m2", "p1", "1", "Neuroticism", 4.0, 1, 1),
        ]
    )
    pd.testing.assert_frame_equal(scales, expected)


def test_oracle_scale_scores_sum_with_the_overall_total(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)

    scales = scale_scores(scored, agg="sum", include_total=True)

    expected = scale_frame(
        [
            ("m1", "p1", "1", "Extraversion", 10.0, 2, 0),
            ("m1", "p1", "1", "Agreeableness", 8.0, 2, 0),
            ("m1", "p1", "1", "Neuroticism", 3.0, 1, 1),
            ("m1", "p1", "1", "total", 21.0, 5, 1),
            ("m1", "p2", "1", "Extraversion", 2.0, 2, 0),
            ("m1", "p2", "1", "Agreeableness", 2.0, 1, 1),
            ("m1", "p2", "1", "Neuroticism", 6.0, 2, 0),
            ("m1", "p2", "1", "total", 10.0, 5, 1),
            ("m2", "p1", "1", "Extraversion", NAN, 0, 2),  # the sum of nothing is not 0
            ("m2", "p1", "1", "Agreeableness", NAN, 0, 2),
            ("m2", "p1", "1", "Neuroticism", 4.0, 1, 1),
            ("m2", "p1", "1", "total", 4.0, 1, 5),
        ]
    )
    pd.testing.assert_frame_equal(scales, expected)


def test_oracle_total_is_the_mean_of_all_items_not_the_mean_of_the_scales(mini_bfi, mini_processed):
    scales = scale_scores(score_answers(mini_processed, mini_bfi), include_total=True)

    totals = scales[scales["dimension"] == TOTAL_SCALE].set_index("profile_id")["score"]
    # m1/p1: (5 + 5 + 4 + 4 + 3) / 5 = 4.2, whereas the mean of its scale means is 4.0
    assert totals.loc["p1"].tolist() == pytest.approx([4.2, 4.0])
    assert totals.loc["p2"] == pytest.approx(2.0)


def test_an_experiment_and_its_questionnaire_score_alike(mini_bfi, mini_processed):
    experiment = build_experiment(name="Mini", questionnaire=mini_bfi)

    from_experiment = score_answers(mini_processed, experiment)

    pd.testing.assert_frame_equal(from_experiment, score_answers(mini_processed, mini_bfi))
    pd.testing.assert_frame_equal(
        score_experiment(experiment, mini_processed), score_experiment(mini_bfi, mini_processed)
    )


# ===========================================================================
# weights, reverse keying and ignored options
# ===========================================================================

#          weights               ignored  reversed  score of opt0, opt1, ...
RULES = [
    pytest.param([1, 2, 3, 4, 5], (), False, [1, 2, 3, 4, 5], id="1..5"),
    pytest.param([1, 2, 3, 4, 5], (), True, [5, 4, 3, 2, 1], id="1..5-reversed"),
    pytest.param([0, 1, 2, 3], (), False, [0, 1, 2, 3], id="0..3"),
    pytest.param([0, 1, 2, 3], (), True, [3, 2, 1, 0], id="0..3-reversed"),
    pytest.param([-2, -1, 0, 1, 2], (), False, [-2, -1, 0, 1, 2], id="-2..2"),
    pytest.param([-2, -1, 0, 1, 2], (), True, [2, 1, 0, -1, -2], id="-2..2-reversed"),
    pytest.param([3, 2, 1, 0], (), True, [0, 1, 2, 3], id="options-in-descending-order-reversed"),
    pytest.param([0, 1, 3], (), True, [3, 2, 0], id="gaps-reversed"),
    pytest.param([-5, -3, -1], (), True, [-1, -3, -5], id="negative-reversed"),
    pytest.param([4], (), True, [4], id="single-option-reversed"),
    pytest.param([1, 1], (), True, [1, 1], id="equal-weights-reversed"),
    pytest.param([1, 2, 3, 4, 5], (2,), False, [1, 2, NAN, 4, 5], id="ignored-middle"),
    pytest.param([1, 2, 3, 4, 5], (2,), True, [5, 4, NAN, 2, 1], id="ignored-middle-reversed"),
    # An ignored option at an extreme does not widen the range that is flipped
    pytest.param([0, 1, 2, 3, 4, 5], (0,), True, [NAN, 5, 4, 3, 2, 1], id="ignored-low-reversed"),
    pytest.param([1, 2, 3, 4, 5, 99], (5,), True, [5, 4, 3, 2, 1, NAN], id="ignored-high-reversed"),
    pytest.param([0, 1, 2, 3], (3,), True, [2, 1, 0, NAN], id="ignored-top-of-0..3-reversed"),
    # If both extremes are ignored the flipped range is the one that remains: 2..4
    pytest.param([1, 2, 3, 4, 5], (0, 4), True, [NAN, 4, 3, 2, NAN], id="ignored-both-ends"),
    pytest.param([1, 2, 3], (0, 1, 2), False, [NAN, NAN, NAN], id="all-ignored"),
    pytest.param([1, 2, 3], (0, 1, 2), True, [NAN, NAN, NAN], id="all-ignored-reversed"),
]


@pytest.mark.parametrize(("weights", "ignored", "reverse", "expected"), RULES)
def test_weights_reverse_keying_and_ignored_options(weights, ignored, reverse, expected):
    question = single_item(weights, reversed=reverse, ignored=ignored)
    processed = answers([f"opt{i}" for i in range(len(weights))])

    assert_scores(score_answers(processed, question), expected)


def test_the_flag_reversed_belongs_to_the_item_and_not_to_the_options():
    # both items share the default options
    question = questionnaire([item(reversed=True), item(reversed=False)], defaults=LIKERT)

    scored = score_answers(answers(["2. Disagree a little"] * 2, [0, 1]), question)

    assert_scores(scored, [4, 2])


def test_a_reversed_item_uses_the_range_of_its_own_options():
    question = questionnaire(
        [
            item(reversed=True, answer_options=weighted([0, 1, 2])),
            item(reversed=True, answer_options=weighted([1, 2, 3, 4, 5, 6, 7])),
        ]
    )

    scored = score_answers(answers(["opt0", "opt0"], [0, 1]), question)

    assert_scores(scored, [2, 7])  # 0 + 2 - 0 and 1 + 7 - 1


# ===========================================================================
# the options of an item or the questionnaire defaults
# ===========================================================================


def test_an_item_uses_its_own_options_and_the_other_items_the_defaults():
    question = questionnaire(
        [item(answer_options=options(option("yes", 10), option("no", 0))), item()],
        defaults=LIKERT,
    )
    processed = answers(
        ["yes", "no", "4. Agree a little", "yes", "4. Agree a little"], [0, 0, 1, 1, 0]
    )

    # the own options replace the defaults (they are not added to them) and vice versa
    assert_scores(score_answers(processed, question), [10, 0, 4, NAN, NAN])


def test_an_item_with_an_empty_set_of_options_uses_the_defaults():
    question = questionnaire([item(answer_options={"options": {}})], defaults=LIKERT)

    assert_scores(score_answers(answers(["5. Agree strongly"]), question), [5])


def test_an_item_without_any_options_gets_no_score():
    question = questionnaire([item()])  # neither own options nor defaults

    with pytest.warns(UserWarning, match="None of the 2 decisions"):
        scored = score_answers(answers(["anything", "4"]), question)

    assert_scores(scored, [NAN, NAN])


def test_each_item_is_scored_with_the_options_it_was_asked_with():
    question = questionnaire(
        [
            item(answer_options=weighted([1, 2, 3])),
            item(answer_options=weighted([10, 20])),
            item(reversed=True),
        ],
        defaults=LIKERT,
    )
    processed = answers(["opt1", "opt1", "1. Disagree strongly"], [0, 1, 2])

    assert_scores(score_answers(processed, question), [2, 20, 5])


# ===========================================================================
# matching the decision with an option text
# ===========================================================================

MATCHING = [
    pytest.param("4. Agree a little", 4, id="exact"),
    pytest.param("  4. Agree a little\n", 4, id="surrounding-white-space"),
    pytest.param("4. agree a little", 4, id="lower-case"),
    pytest.param("4. AGREE A LITTLE", 4, id="upper-case"),
    pytest.param("4.   Agree\ta   little", 4, id="inner-white-space"),
    pytest.param("\xa04. Agree a little ", 4, id="unicode-white-space"),
    pytest.param("4. Agree a litte", NAN, id="typo"),
    pytest.param("Agree a little", NAN, id="part-of-an-option"),
    pytest.param("4", NAN, id="the-number-only"),
    pytest.param("4. Agree a little, I think", NAN, id="option-inside-a-sentence"),
    pytest.param("", NAN, id="empty"),
    pytest.param("   ", NAN, id="blank"),
    pytest.param("not present", NAN, id="sentinel-not-present"),
    pytest.param("inconclusive", NAN, id="sentinel-inconclusive"),
]


@pytest.mark.parametrize(("decision", "expected"), MATCHING)
def test_a_decision_matches_an_option_text_ignoring_case_and_white_space(
    likert_item, decision, expected
):
    # the second row keeps the frame from having no matching decision at all (that would warn)
    processed = answers([decision, "1. Disagree strongly"])

    assert_scores(score_answers(processed, likert_item), [expected, 1])


def test_an_exact_match_wins_over_a_case_insensitive_one():
    question = questionnaire([item(answer_options=options(option("Agree", 1), option("agree", 2)))])
    processed = answers(["Agree", "agree", "AGREE", " agree"])

    # "AGREE" and " agree" match both options once case and white space are ignored: ambiguous
    assert_scores(score_answers(processed, question), [1, 2, NAN, NAN])


def test_texts_that_differ_in_case_but_score_alike_are_not_ambiguous():
    question = questionnaire([item(answer_options=options(option("Agree", 3), option("agree", 3)))])

    assert_scores(score_answers(answers(["AGREE", "Agree"]), question), [3, 3])


def test_a_judge_sentinel_is_not_mistaken_for_an_option_of_the_same_name():
    question = questionnaire(
        [item(answer_options=options(option("Not present", 0), option("Present", 1)))]
    )
    processed = answers(["Not present", "not present", "Present", "present", "NOT PRESENT"])

    # the judge's "not present" never matches by case alone; the other spellings still do
    assert_scores(score_answers(processed, question), [0, NAN, 1, 1, 0])


def test_numbers_are_compared_as_text():
    question = questionnaire([item(answer_options=options(option("1", 1), option("2", 2)))])
    decisions = [
        [1, 2],
        [1.0, 2.0],
        np.array([1, 2], dtype=np.int32),
        np.array([1.0, 2.0], dtype=np.float32),
        pd.array([1, 2], dtype="Int64"),
        ["1", "2"],
    ]

    for column in decisions:
        assert_scores(score_answers(answers(column), question), [1, 2])


@pytest.mark.parametrize(
    "decision", [1.5, 3, float("inf"), True, np.bool_(False), (1,), ("a", "b"), b"1"], ids=repr
)
def test_values_that_match_no_option_get_no_score(decision):
    question = questionnaire([item(answer_options=options(option("1", 1), option("2", 2)))])
    processed = pd.DataFrame(
        {"instruction_item_id": [0, 0], "decision": pd.Series([decision, "1"], dtype=object)}
    )

    assert_scores(score_answers(processed, question), [NAN, 1])


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, float("nan"), pd.NaT], ids=repr)
def test_missing_decisions_get_no_score(likert_item, missing):
    processed = pd.DataFrame(
        {"instruction_item_id": [0, 0], "decision": pd.Series(["4. Agree a little", missing])}
    )

    assert_scores(score_answers(processed, likert_item), [4, NAN])


@pytest.mark.parametrize(
    "decisions",
    [
        pd.Series([np.nan, np.nan]),  # a float column
        pd.Series([None, None], dtype=object),
        pd.Series([pd.NA, pd.NA], dtype="string"),
        pd.Series([None, None], dtype="category"),
    ],
    ids=["float", "object", "string", "category"],
)
def test_a_column_without_any_decision_gives_no_scores(likert_item, decisions):
    processed = pd.DataFrame({"instruction_item_id": [0, 0], "decision": decisions})

    assert_scores(score_answers(processed, likert_item), [NAN, NAN])


def test_options_without_text_can_never_be_chosen():
    question = questionnaire(
        [item(answer_options=options(option("", 1), option("  ", 2), option("A", 3)))]
    )

    # the two blank options do not count as duplicates either (no warning, see pytestmark)
    assert_scores(score_answers(answers(["", "  ", "A"]), question), [NAN, NAN, 3])


# ===========================================================================
# duplicate option texts
# ===========================================================================


def test_options_with_the_same_text_and_the_same_score_are_fine():
    question = questionnaire(
        [item(answer_options=options(option("Yes", 1), option("Yes", 1), option("No", 0)))]
    )

    assert_scores(score_answers(answers(["Yes", "No"]), question), [1, 0])


def test_options_with_the_same_text_but_different_scores_cannot_be_scored_and_warn():
    question = questionnaire(
        [item(answer_options=options(option("Yes", 1), option("Yes", 2), option("No", 0)))]
    )

    with pytest.warns(UserWarning, match=r"item 0: 'Yes'"):
        scored = score_answers(answers(["Yes", "yes", "No"]), question)

    assert_scores(scored, [NAN, NAN, 0])


def test_the_duplicate_warning_names_every_affected_item():
    clash = options(option("A", 1), option("A", 2), option("B", 3), option("B", 4))
    question = questionnaire(
        [
            item(answer_options=clash),
            item(answer_options=weighted([1, 2])),
            item(answer_options=clash),
        ]
    )

    with pytest.warns(UserWarning) as caught:
        score_answers(answers(["A", "A", "A"], [0, 1, 2]), question)

    (warning,) = caught
    assert "item 0: 'A', 'B'" in str(warning.message)
    assert "item 2: 'A', 'B'" in str(warning.message)
    assert "item 1" not in str(warning.message)


def test_a_duplicate_of_an_ignored_option_is_ambiguous_too():
    question = questionnaire(
        [item(answer_options=options(option("Skip", 1, ignored=True), option("Skip", 1)))]
    )

    with pytest.warns(UserWarning, match="'Skip'"):
        scored = score_answers(answers(["Skip"]), question)

    assert_scores(scored, [NAN])


def test_duplicates_that_are_all_ignored_are_not_a_conflict():
    question = questionnaire(
        [
            item(
                answer_options=options(
                    option("Skip", 1, ignored=True), option("Skip", 2, ignored=True), option("A", 1)
                )
            )
        ]
    )

    assert_scores(score_answers(answers(["Skip", "A"]), question), [NAN, 1])


def test_conflicting_duplicates_only_matter_for_items_that_are_in_the_frame():
    question = questionnaire(
        [
            item(answer_options=options(option("A", 1), option("A", 2))),
            item(answer_options=weighted([1])),
        ]
    )

    assert_scores(score_answers(answers(["opt0"], [1]), question), [1])  # no warning


# ===========================================================================
# decisions that are no options at all
# ===========================================================================


def test_a_warning_says_when_not_a_single_decision_is_the_text_of_an_option(likert_item):
    raw = ["I would say 4.", "Probably a 5", "I think so", "not present"]

    with pytest.warns(UserWarning) as caught:
        scored = score_answers(answers(raw), likert_item)

    (warning,) = caught
    message = str(warning.message)
    assert "None of the 3 decisions in column 'decision'" in message  # "not present" is no answer
    assert "for instance 'I would say 4.'" in message
    assert "PostprocessingPipeline" in message
    assert_scores(scored, [NAN] * 4)


@pytest.mark.parametrize(
    "decisions",
    [
        ["4. Agree a little", "free text"],  # one option text is enough
        ["not present", "inconclusive", None, " "],  # nothing that was meant to be an option
        [],
    ],
    ids=["one-match", "only-sentinels-and-missing", "no-rows"],
)
def test_no_warning_while_a_decision_matches_or_none_is_meant_to(likert_item, decisions):
    score_answers(answers(decisions), likert_item)  # a warning would be an error here


def test_a_decision_that_matches_an_ignored_option_counts_as_a_match():
    question = questionnaire(
        [item(answer_options=options(option("Don't know", 0, ignored=True), option("Yes", 1)))]
    )

    scored = score_answers(answers(["Don't know", "garbage"]), question)  # no warning

    assert_scores(scored, [NAN, NAN])


def test_ambiguous_decisions_count_as_matches_too():
    question = questionnaire([item(answer_options=options(option("A", 1), option("A", 2)))])

    with pytest.warns(UserWarning, match="same text") as caught:
        score_answers(answers(["A"]), question)

    assert len(caught) == 1  # the duplicate warning only


# ===========================================================================
# invalid answers
# ===========================================================================


def test_invalid_answers_get_no_score_by_default(likert_item):
    processed = answers(["4. Agree a little"] * 3, valid=[True, False, True])

    assert_scores(score_answers(processed, likert_item), [4, NAN, 4])


def test_only_valid_false_scores_invalid_answers_too(likert_item):
    processed = answers(["4. Agree a little"] * 3, valid=[True, False, True])

    assert_scores(score_answers(processed, likert_item, only_valid=False), [4, 4, 4])


def test_without_a_valid_column_every_answer_counts(likert_item):
    processed = answers(["4. Agree a little"] * 2)

    assert_scores(score_answers(processed, likert_item, only_valid=True), [4, 4])


def test_a_missing_verdict_does_not_exclude_an_answer(likert_item):
    verdicts = pd.Series([True, None, np.nan, pd.NA, False], dtype=object)
    processed = answers(["4. Agree a little"] * 5, valid=verdicts)

    assert_scores(score_answers(processed, likert_item), [4, 4, 4, 4, NAN])


@pytest.mark.parametrize(
    ("verdict", "excluded"),
    [
        (False, True),
        (True, False),
        (np.bool_(False), True),
        (0, True),
        (1, False),
        (0.0, True),
        (2, False),
        ("False", True),
        ("false", True),
        (" FALSE ", True),
        ("f", True),
        ("no", True),
        ("n", True),
        ("0", True),
        ("True", False),
        ("yes", False),
        ("1", False),
        ("maybe", False),
        ((), False),
    ],
    ids=repr,
)
def test_verdicts_saved_as_text_or_numbers(likert_item, verdict, excluded):
    processed = answers(["4. Agree a little"], valid=pd.Series([verdict], dtype=object))

    assert_scores(score_answers(processed, likert_item), [NAN if excluded else 4])


def test_the_verdict_may_have_a_nullable_or_categorical_type(likert_item):
    texts = ["4. Agree a little"] * 3

    nullable = answers(texts, valid=pd.array([True, False, None], dtype="boolean"))
    categorical = answers(texts, valid=pd.Categorical([True, False, None]))

    assert_scores(score_answers(nullable, likert_item), [4, NAN, 4])
    assert_scores(score_answers(categorical, likert_item), [4, NAN, 4])


def test_the_column_with_the_verdicts_can_be_named(likert_item):
    processed = answers(["4. Agree a little"] * 2, ok=[False, True], valid=[True, True])

    assert_scores(score_answers(processed, likert_item, valid_column="ok"), [NAN, 4])


# ===========================================================================
# dtypes
# ===========================================================================


@pytest.mark.parametrize(
    "ids",
    [
        pd.Series([0, 1, 0], dtype="int64"),
        pd.Series([0, 1, 0], dtype="int8"),
        pd.Series([0, 1, 0], dtype="uint16"),
        pd.Series([0.0, 1.0, 0.0]),
        pd.Series([0, 1, 0], dtype="Int64"),
        pd.Series([0, 1, 0], dtype=object),
        pd.Series(["0", "1", " 0 "]),
        pd.Series(["0", "1", "0"], dtype="string"),
        pd.Series(pd.Categorical([0, 1, 0])),
        pd.Series(pd.Categorical(["0", "1", "0"])),
    ],
    ids=lambda ids: str(ids.dtype),
)
def test_item_ids_of_any_dtype(ids):
    question = questionnaire(
        [item(answer_options=weighted([1, 2])), item(answer_options=weighted([10, 20]))]
    )
    processed = pd.DataFrame({"instruction_item_id": ids, "decision": ["opt1", "opt1", "opt0"]})

    assert_scores(score_answers(processed, question), [2, 20, 1])


@pytest.mark.parametrize(
    "decisions",
    [
        pd.Series(["4. Agree a little", None, "5. Agree strongly"], dtype=object),
        pd.Series(["4. Agree a little", None, "5. Agree strongly"], dtype="string"),
        pd.Series(["4. Agree a little", None, "5. Agree strongly"], dtype="category"),
        pd.Series(
            pd.Categorical(
                ["4. Agree a little", None, "5. Agree strongly"],
                categories=["5. Agree strongly", "unused", "4. Agree a little"],
                ordered=True,
            )
        ),
    ],
    ids=["object", "string", "category", "ordered-category"],
)
def test_decisions_of_any_dtype(likert_item, decisions):
    processed = pd.DataFrame({"instruction_item_id": 0, "decision": decisions})

    assert_scores(score_answers(processed, likert_item), [4, NAN, 5])


# ===========================================================================
# the frame that comes back
# ===========================================================================


def test_the_input_is_not_changed_and_the_result_is_a_copy(likert_item):
    processed = answers(["4. Agree a little"] * 2, valid=[True, True])
    before = processed.copy(deep=True)

    result = score_answers(processed, likert_item)
    result.loc[:, "decision"] = "changed"

    pd.testing.assert_frame_equal(processed, before)
    assert processed.attrs == {}
    assert list(result.columns) == [
        "instruction_item_id",
        "decision",
        "valid",
        "score",
        "dimension",
    ]


def test_index_and_extra_columns_are_kept(likert_item):
    processed = answers(
        ["4. Agree a little", "5. Agree strongly", "not present"],
        time=[0.5, 0.25, 0.125],
        note=["a", "b", "c"],
    ).set_axis(["x", "y", "x"])  # the index is not unique

    result = score_answers(processed, likert_item)

    assert list(result.index) == ["x", "y", "x"]
    assert list(result.columns) == [
        "instruction_item_id",
        "decision",
        "time",
        "note",
        "score",
        "dimension",
    ]
    assert result["time"].tolist() == [0.5, 0.25, 0.125]
    assert_scores(result, [4, 5, NAN])


def test_columns_named_score_and_dimension_are_replaced(likert_item):
    processed = answers(["4. Agree a little"], score=["old"], dimension=["old"])

    result = score_answers(processed, likert_item)

    assert list(result.columns) == ["instruction_item_id", "decision", "score", "dimension"]
    assert_scores(result, [4])
    assert dimension_list(result) == [None]


def test_scoring_a_scored_frame_again_changes_nothing(mini_bfi, mini_processed):
    once = score_answers(mini_processed, mini_bfi)

    twice = score_answers(once, mini_bfi)

    pd.testing.assert_frame_equal(once, twice)


def test_the_score_column_is_a_float_column_and_the_dimension_an_object_column(likert_item):
    result = score_answers(answers(["4. Agree a little"]), likert_item)

    assert result["score"].dtype == np.dtype("float64")
    assert result["dimension"].dtype == np.dtype("object")
    assert result["dimension"].tolist() == [None]  # an item without dimension has None


# ===========================================================================
# empty frames
# ===========================================================================


def test_an_empty_frame_gets_the_columns(likert_item):
    processed = answers(["4. Agree a little"], valid=[True]).iloc[0:0]

    result = score_answers(processed, likert_item)

    assert list(result.columns) == [
        "instruction_item_id",
        "decision",
        "valid",
        "score",
        "dimension",
    ]
    assert len(result) == 0
    assert result["score"].dtype == np.dtype("float64")


def test_a_frame_of_untyped_empty_columns_is_scored(likert_item):
    processed = pd.DataFrame(columns=["instruction_item_id", "decision", "valid"])

    result = score_answers(processed, likert_item)

    assert list(result.columns) == [
        "instruction_item_id",
        "decision",
        "valid",
        "score",
        "dimension",
    ]
    assert len(result) == 0


def test_an_empty_frame_can_be_scored_with_a_questionnaire_without_items():
    empty_questionnaire = Questionnaire(name="Empty", general_instruction="-")

    result = score_answers(answers([]), empty_questionnaire)

    assert len(result) == 0
    assert list(result.columns) == ["instruction_item_id", "decision", "score", "dimension"]


# ===========================================================================
# errors
# ===========================================================================


def test_a_missing_item_column_is_reported_with_the_available_columns(likert_item):
    with pytest.raises(
        ValueError, match=r"item id column 'instruction_item_id' is missing.*decision"
    ):
        score_answers(pd.DataFrame({"decision": []}), likert_item)


def test_a_missing_decision_column_is_reported_with_the_available_columns(likert_item):
    with pytest.raises(
        ValueError, match=r"decision column 'decision' is missing.*instruction_item_id"
    ):
        score_answers(pd.DataFrame({"instruction_item_id": []}), likert_item)


def test_the_columns_can_be_named(likert_item):
    processed = pd.DataFrame({"item": [0, 0], "choice": ["4. Agree a little", "5. Agree strongly"]})

    result = score_answers(processed, likert_item, item_column="item", decision_column="choice")

    assert_scores(result, [4, 5])
    with pytest.raises(ValueError, match="item id column 'wrong' is missing"):
        score_answers(processed, likert_item, item_column="wrong", decision_column="choice")
    with pytest.raises(ValueError, match="decision column 'wrong' is missing"):
        score_answers(processed, likert_item, item_column="item", decision_column="wrong")


def test_a_column_that_occurs_twice_is_ambiguous(likert_item):
    processed = pd.concat([answers(["4. Agree a little"]), answers(["x"])[["decision"]]], axis=1)

    with pytest.raises(ValueError, match="decision column 'decision' occurs more than once"):
        score_answers(processed, likert_item)


@pytest.mark.parametrize("bad", [-1, 2, 99, 1.5, "abc", None, np.nan, float("inf")], ids=repr)
def test_an_invalid_item_id_is_an_error(bad):
    question = questionnaire(
        [item(answer_options=weighted([1])), item(answer_options=weighted([1]))]
    )
    processed = pd.DataFrame(
        {"instruction_item_id": pd.Series([0, bad], dtype=object), "decision": ["opt0", "opt0"]}
    )

    with pytest.raises(ValueError, match=r"1 invalid item id\(s\).*ids are integers from 0 to 1"):
        score_answers(processed, question)


def test_the_error_for_item_ids_shows_examples_and_hints_at_the_cause():
    question = questionnaire([item()], defaults=LIKERT)
    processed = answers(["4. Agree a little"] * 4, [0, 7, 7, 8])

    with pytest.raises(ValueError) as error:
        score_answers(processed, question)

    message = str(error.value)
    assert "'instruction_item_id'" in message
    assert "3 invalid item id(s), e.g. 7, 8" in message
    assert "the questionnaire has one item, so the only valid id is 0" in message
    assert "Is this the experiment the results belong to?" in message


def test_a_questionnaire_without_items_has_no_valid_item_id():
    question = Questionnaire(name="Empty", general_instruction="-")

    with pytest.raises(ValueError, match="the questionnaire has no instruction items"):
        score_answers(answers(["x"]), question)


def test_an_experiment_without_a_questionnaire_cannot_score():
    experiment = build_experiment(name="No questionnaire")

    with pytest.raises(ValueError, match="no questionnaire"):
        score_answers(answers(["x"]), experiment)
    with pytest.raises(ValueError, match="no questionnaire"):
        score_experiment(experiment)


@pytest.mark.parametrize("wrong", [None, "config.json", {"questionnaire": {}}, 3, [1]], ids=repr)
def test_something_that_is_not_an_experiment_is_a_type_error(wrong):
    with pytest.raises(TypeError, match="ExperimentDocument or a Questionnaire"):
        score_answers(answers(["x"]), wrong)


@pytest.mark.parametrize("collection", [["A"], {"validation_status": "valid"}, {1, 2}], ids=repr)
def test_columns_with_lists_or_dictionaries_are_a_type_error(likert_item, collection):
    # for instance the column validation_status of the pipeline, which holds dictionaries
    cells = pd.Series([collection, collection], dtype=object)
    texts = ["4. Agree a little"] * 2

    with pytest.raises(TypeError, match=r"decision column 'decision' must hold single values"):
        score_answers(answers(cells), likert_item)
    with pytest.raises(
        TypeError, match=r"valid column 'verdict' must hold single values.*not coll"
    ):
        score_answers(answers(texts, verdict=cells), likert_item, valid_column="verdict")
    with pytest.raises(TypeError, match=r"dimension column 'dimension' must hold single values"):
        scale_scores(pd.DataFrame({"dimension": cells, "score": [1.0, 2.0]}), by=[])


@pytest.mark.parametrize("wrong", [[1, 2], {"a": [1]}, "frame.csv", None], ids=repr)
def test_processed_must_be_a_data_frame(likert_item, wrong):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        score_answers(wrong, likert_item)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        scale_scores(wrong)


# ===========================================================================
# dimensions
# ===========================================================================


def dimensions_of(item_dimension: Any, declared: Any) -> str | None:
    """The scale name of a one-item questionnaire."""
    question = questionnaire([item(dimension=item_dimension)], defaults=LIKERT, dimensions=declared)
    return dimension_list(score_answers(answers(["4. Agree a little"]), question))[0]


@pytest.mark.parametrize(
    ("item_dimension", "declared", "expected"),
    [
        pytest.param("1", {"1": "Extraversion"}, "Extraversion", id="name-of-the-id"),
        pytest.param(1, {"1": "Extraversion"}, "Extraversion", id="integer-id-string-key"),
        pytest.param(1.0, {"1": "Extraversion"}, "Extraversion", id="whole-float-id"),
        pytest.param(" 1 ", {"1": "Extraversion"}, "Extraversion", id="padded-id"),
        pytest.param("1", {1: "Extraversion"}, "Extraversion", id="string-id-integer-key"),
        pytest.param("1", {" 1": "Extraversion"}, "Extraversion", id="padded-key"),
        pytest.param("9", {"1": "Extraversion"}, "9", id="id-missing-from-the-mapping"),
        pytest.param(9, {"1": "Extraversion"}, "9", id="integer-id-missing-from-the-mapping"),
        pytest.param(
            "1", {"": "Nothing", " ": "Blank", "1": "One"}, "One", id="blank-ids-are-ignored"
        ),
        pytest.param("1", {"1": ""}, "1", id="blank-name-falls-back-to-the-id"),
        pytest.param("1", {"1": None}, "1", id="missing-name-falls-back-to-the-id"),
        pytest.param("1", {"1": 7}, "7", id="the-name-is-text"),
        pytest.param("Warmth", None, "Warmth", id="no-mapping-uses-the-value"),
        pytest.param("Warmth", {}, "Warmth", id="empty-mapping-uses-the-value"),
        pytest.param("Warmth", ["Warmth", "Other"], "Warmth", id="a-mapping-is-needed"),
        pytest.param("Warmth", "Warmth", "Warmth", id="a-text-is-no-mapping"),
        pytest.param(None, {"1": "Extraversion"}, None, id="item-without-dimension"),
        pytest.param("", {"1": "Extraversion"}, None, id="blank-dimension"),
        pytest.param("  ", {"1": "Extraversion"}, None, id="whitespace-dimension"),
    ],
)
def test_the_dimension_of_an_item(item_dimension, declared, expected):
    assert dimensions_of(item_dimension, declared) == expected


def test_an_item_with_a_null_dimension_has_none():
    question = questionnaire(
        [{"question": "Q", "attributes": {"dimension": None}}],
        defaults=LIKERT,
        dimensions={"1": "Extraversion"},
    )

    assert dimension_list(score_answers(answers(["4. Agree a little"]), question)) == [None]


def test_questionnaire_attributes_without_a_dimension_key_leave_the_values_as_they_are():
    data = {
        "name": "Q",
        "general_instruction": "-",
        "attributes": {"language": "en"},
        "default_answer_options": LIKERT,
        "instruction_items": [item(dimension="1")],
    }

    result = score_answers(answers(["4. Agree a little"]), Questionnaire.model_validate(data))

    assert dimension_list(result) == ["1"]


def test_two_ids_may_name_the_same_scale():
    question = questionnaire(
        [item(dimension="1"), item(dimension="2")],
        defaults=LIKERT,
        dimensions={"1": "Warmth", "2": "Warmth"},
    )
    scored = score_answers(answers(["4. Agree a little", "2. Disagree a little"], [0, 1]), question)

    scales = scale_scores(scored, by=[])

    assert scales["dimension"].tolist() == ["Warmth"]
    assert scales["score"].tolist() == [3.0]


# ===========================================================================
# scale scores
# ===========================================================================


def scored_frame(rows: list[tuple[Any, ...]], columns: list[str] | None = None) -> pd.DataFrame:
    """A frame of item scores: ``model_id``, ``dimension``, ``score`` unless named otherwise."""
    return pd.DataFrame(rows, columns=columns or ["model_id", "dimension", "score"])


def test_a_scale_is_the_mean_of_the_scores_that_exist():
    scored = scored_frame(
        [("m", "A", 1.0), ("m", "A", 2.0), ("m", "A", NAN), ("m", "B", 4.0), ("n", "A", 5.0)]
    )

    scales = scale_scores(scored, by="model_id")

    expected = pd.DataFrame(
        {
            "model_id": ["m", "m", "n"],
            "dimension": ["A", "B", "A"],
            "score": [1.5, 4.0, 5.0],
            "n_items": [2, 1, 1],
            "n_missing": [1, 0, 0],
        }
    )
    pd.testing.assert_frame_equal(scales, expected)


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("mean", 2.0),
        ("sum", 6.0),
        ("median", 2.0),
        ("min", 1.0),
        ("max", 3.0),
        ("std", 1.0),
        ("count", 3),
        (lambda scores: float(scores.max() - scores.min()), 2.0),
    ],
    ids=["mean", "sum", "median", "min", "max", "std", "count", "lambda"],
)
def test_aggregations(agg, expected):
    scored = scored_frame([("m", "A", 1.0), ("m", "A", 2.0), ("m", "A", NAN), ("m", "A", 3.0)])

    scales = scale_scores(scored, by="model_id", agg=agg)

    assert scales["score"].tolist() == [expected]
    assert scales["n_items"].tolist() == [3]
    assert scales["n_missing"].tolist() == [1]


@pytest.mark.parametrize("agg", ["mean", "sum", "median", "max", "std", lambda s: s.sum()])
def test_a_scale_without_a_single_score_has_no_score_whatever_the_aggregation(agg):
    scored = scored_frame([("m", "A", NAN), ("m", "A", NAN), ("m", "B", 2.0)])

    scales = scale_scores(scored, by="model_id", agg=agg)

    empty = scales[scales["dimension"] == "A"].iloc[0]
    assert math.isnan(empty["score"])
    assert (empty["n_items"], empty["n_missing"]) == (0, 2)


@pytest.mark.parametrize("agg", ["average", "nope", "_private"])
def test_an_unknown_aggregation_is_a_value_error(agg):
    with pytest.raises(ValueError, match=f"Cannot aggregate the scores with agg='{agg}'"):
        scale_scores(scored_frame([("m", "A", 1.0)]), by="model_id", agg=agg)


def test_an_aggregation_that_fails_is_a_value_error_with_the_cause():
    def broken(scores):
        raise TypeError("cannot do that")

    with pytest.raises(ValueError, match="cannot do that") as error:
        scale_scores(scored_frame([("m", "A", 1.0)]), by="model_id", agg=broken)

    assert isinstance(error.value.__cause__, TypeError)


def test_group_columns_that_the_frame_lacks_are_skipped_silently():
    # no profile_id and no random_seed (see pytestmark: not even a warning)
    scored = scored_frame([("m", "A", 1.0), ("m", "A", 3.0), ("n", "A", 5.0)])

    scales = scale_scores(scored)

    assert list(scales.columns) == ["model_id", "dimension", "score", "n_items", "n_missing"]
    assert scales["score"].tolist() == [2.0, 5.0]


def test_without_any_group_column_there_is_one_row_per_scale():
    scored = pd.DataFrame({"dimension": ["B", "A", "B", None], "score": [1.0, 2.0, 3.0, 4.0]})

    scales = scale_scores(scored)

    assert list(scales.columns) == ["dimension", "score", "n_items", "n_missing"]
    assert scales["dimension"].tolist() == ["B", "A", "total"]  # order of appearance
    assert scales["score"].tolist() == [2.0, 2.0, 4.0]


def test_an_unknown_group_column_is_skipped_with_a_warning():
    scored = scored_frame([("m", "A", 1.0), ("n", "A", 3.0)])

    with pytest.warns(UserWarning, match=r"unknown group column\(s\) \['modle'\]"):
        scales = scale_scores(scored, by=("model_id", "modle"))

    assert scales["model_id"].tolist() == ["m", "n"]


def test_the_standard_group_columns_are_never_reported_when_missing():
    scored = scored_frame([("m", "A", 1.0)])

    # no warning, or the module-wide "error" filter would fail the test
    scales = scale_scores(scored, by=("model_id", "profile_id", "random_seed"))

    assert list(scales.columns)[0] == "model_id"


@pytest.mark.parametrize("by", ["model_id", ["model_id"], ("model_id",), ("model_id", "model_id")])
def test_by_may_be_a_name_or_a_sequence_of_names(by):
    scales = scale_scores(scored_frame([("m", "A", 1.0), ("n", "A", 3.0)]), by=by)

    assert scales["model_id"].tolist() == ["m", "n"]


def test_by_must_not_contain_the_dimension_or_the_score_column():
    scored = scored_frame([("m", "A", 1.0)])

    for column in ("dimension", "score"):
        with pytest.raises(ValueError, match="by must not contain"):
            scale_scores(scored, by=["model_id", column])


def test_the_result_is_sorted_by_the_group_columns_and_has_a_fresh_index():
    scored = pd.DataFrame(
        {
            "model_id": ["b", "a", "b", "a", "a"],
            "profile_id": ["y", "y", "x", "x", "y"],
            "random_seed": [2, 1, 1, 1, 1],
            "dimension": ["A"] * 5,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[40, 30, 20, 10, 0],
    )

    scales = scale_scores(scored)

    assert scales[["model_id", "profile_id", "random_seed"]].values.tolist() == [
        ["a", "x", 1],
        ["a", "y", 1],
        ["b", "x", 1],
        ["b", "y", 2],
    ]
    assert scales["score"].tolist() == [4.0, 3.5, 3.0, 1.0]
    assert list(scales.index) == [0, 1, 2, 3]


def test_missing_values_in_a_group_column_form_a_group_that_comes_last():
    scored = scored_frame([("b", "A", 1.0), (None, "A", 2.0), ("a", "A", 3.0), (np.nan, "A", 4.0)])

    scales = scale_scores(scored, by="model_id")

    assert scales["model_id"].tolist()[:2] == ["a", "b"]
    assert scales["model_id"].isna().tolist() == [False, False, True]
    assert scales["score"].tolist() == [3.0, 1.0, 3.0]
    assert scales["n_items"].tolist() == [1, 1, 2]


def test_categorical_group_columns_do_not_produce_empty_groups():
    scored = scored_frame([("m", "A", 1.0), ("n", "A", 2.0)])
    scored["model_id"] = pd.Categorical(
        scored["model_id"], categories=["z", "n", "m", "unused"], ordered=True
    )

    scales = scale_scores(scored, by="model_id")

    assert scales["model_id"].astype(str).tolist() == ["n", "m"]  # the order of the categories
    assert scales["score"].tolist() == [2.0, 1.0]


@pytest.mark.parametrize(
    "dimension",
    [
        pd.Series(["B", "A", "B"], dtype="string"),
        pd.Series(pd.Categorical(["B", "A", "B"], categories=["C", "A", "B"])),
        pd.Series(["B", "A", "B"], dtype=object),
    ],
    ids=["string", "category", "object"],
)
def test_the_dimension_column_may_have_any_text_type(dimension):
    scored = pd.DataFrame({"dimension": dimension, "score": [1.0, 2.0, 3.0]})

    scales = scale_scores(scored, by=[])

    assert scales["dimension"].tolist() == ["B", "A"]
    assert scales["score"].tolist() == [2.0, 2.0]


def test_dimension_ids_that_are_numbers_become_text():
    scored = pd.DataFrame({"dimension": [2, 1, 2, np.nan], "score": [1.0, 2.0, 3.0, 4.0]})

    scales = scale_scores(scored, by=[])

    assert scales["dimension"].tolist() == ["2", "1", "total"]


def test_blank_dimensions_belong_to_the_total_scale():
    scored = pd.DataFrame({"dimension": ["A", "", "  ", None], "score": [1.0, 2.0, 3.0, 4.0]})

    scales = scale_scores(scored, by=[])

    assert scales["dimension"].tolist() == ["A", "total"]
    assert scales["n_items"].tolist() == [1, 3]


def test_the_columns_can_be_named_and_the_result_uses_the_same_names():
    scored = pd.DataFrame({"who": ["m", "m"], "scale": ["A", "A"], "points": [1.0, 2.0]})

    scales = scale_scores(scored, by="who", dimension_column="scale", score_column="points")

    assert list(scales.columns) == ["who", "scale", "points", "n_items", "n_missing"]
    assert scales["points"].tolist() == [1.5]


def test_scores_may_be_stored_as_objects_or_integers():
    objects = pd.DataFrame({"dimension": ["A"] * 3, "score": pd.Series([1, None, 3], dtype=object)})
    integers = pd.DataFrame({"dimension": ["A"] * 3, "score": pd.Series([1, 2, 3], dtype="Int64")})
    categories = pd.DataFrame({"dimension": ["A"] * 3, "score": pd.Categorical([1.0, 2.0, 3.0])})

    assert scale_scores(objects, by=[])["score"].tolist() == [2.0]
    assert scale_scores(objects, by=[])["n_missing"].tolist() == [1]
    assert scale_scores(integers, by=[])["score"].tolist() == [2.0]
    assert scale_scores(categories, by=[])["score"].tolist() == [2.0]


def test_scores_that_are_not_numbers_are_a_value_error():
    scored = pd.DataFrame({"dimension": ["A", "A"], "score": ["1", "high"]})

    with pytest.raises(ValueError, match="score column 'score' must contain numbers"):
        scale_scores(scored, by=[])


@pytest.mark.parametrize("missing", ["dimension", "score"])
def test_missing_dimension_or_score_columns_are_reported(missing):
    scored = scored_frame([("m", "A", 1.0)]).drop(columns=missing)

    with pytest.raises(ValueError, match=f"{missing} column '{missing}' is missing"):
        scale_scores(scored, by="model_id")


def test_the_empty_result_has_all_columns_and_types():
    scored = pd.DataFrame({"model_id": [], "dimension": [], "score": []}, dtype=object).astype(
        {"score": "float64"}
    )

    scales = scale_scores(scored, include_total=True)

    assert list(scales.columns) == ["model_id", "dimension", "score", "n_items", "n_missing"]
    assert len(scales) == 0
    assert scales["score"].dtype == np.dtype("float64")
    assert scales["n_items"].dtype == np.dtype("int64")
    assert scales["n_missing"].dtype == np.dtype("int64")


def test_scores_of_an_empty_frame_have_the_columns_of_the_result(likert_item):
    processed = answers(["4. Agree a little"], model_id=["m"]).iloc[0:0]

    scales = scale_scores(score_answers(processed, likert_item))

    assert list(scales.columns) == ["model_id", "dimension", "score", "n_items", "n_missing"]
    assert len(scales) == 0


# ---- the order of the scales ----------------------------------------------


def order_of(scales: pd.DataFrame) -> list[str]:
    return scales["dimension"].drop_duplicates().tolist()


def test_the_scales_follow_the_order_of_the_questionnaire():
    # the questionnaire lists Zeta before Alpha although the first item belongs to Alpha
    question = questionnaire(
        [item(dimension="a"), item(dimension="z"), item(dimension="m")] * 2,
        defaults=LIKERT,
        dimensions={"z": "Zeta", "a": "Alpha", "m": "Mu"},
    )
    processed = answers(["4. Agree a little"] * 6, [0, 1, 2, 3, 4, 5], model_id="m1")

    scales = scale_scores(score_answers(processed, question), include_total=True)

    assert order_of(scales) == ["Zeta", "Alpha", "Mu", "total"]


def test_scales_of_dimensions_that_are_not_declared_follow_the_declared_ones():
    question = questionnaire(
        [item(dimension="7"), item(dimension="1"), item(dimension="3")],
        defaults=LIKERT,
        dimensions={"1": "One", "2": "Two"},
    )
    processed = answers(["4. Agree a little"] * 3, [0, 1, 2], model_id="m")

    scales = scale_scores(score_answers(processed, question))

    assert order_of(scales) == ["One", "7", "3"]  # declared, then in order of appearance


def test_total_is_always_the_last_scale():
    question = questionnaire(
        [item(), item(dimension="1"), item(dimension="2")],
        defaults=LIKERT,
        dimensions={"1": "One", "2": "Two"},
    )
    processed = answers(["4. Agree a little"] * 3, [0, 1, 2])

    assert order_of(scale_scores(score_answers(processed, question), by=[])) == [
        "One",
        "Two",
        "total",
    ]


def test_the_order_survives_operations_on_the_scored_frame(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)
    shuffled = scored.sample(frac=1, random_state=3).reset_index(drop=True)
    filtered = shuffled[shuffled["model_id"] == "m1"]

    for frame in (shuffled, filtered, shuffled.sort_values("score"), shuffled.copy()):
        assert order_of(scale_scores(frame)) == ["Extraversion", "Agreeableness", "Neuroticism"]


def test_without_a_recorded_order_the_scales_come_in_order_of_appearance(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)
    scored = scored.sort_values("instruction_item_id", ascending=False)
    scored.attrs.clear()

    assert order_of(scale_scores(scored)) == ["Neuroticism", "Agreeableness", "Extraversion"]


def test_an_explicit_order_wins(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)

    scales = scale_scores(scored, dimensions=["Neuroticism", "Extraversion"], include_total=True)

    # unlisted scales follow the listed ones; total is last whatever the list says
    assert order_of(scales) == ["Neuroticism", "Extraversion", "Agreeableness", "total"]
    again = scale_scores(scored, dimensions=("total", "Agreeableness"), include_total=True)
    assert order_of(again) == ["Agreeableness", "Extraversion", "Neuroticism", "total"]


def test_an_order_given_as_numbers_is_taken_as_text():
    scored = pd.DataFrame({"dimension": ["1", "2", "3"], "score": [1.0, 2.0, 3.0]})

    assert order_of(scale_scores(scored, by=[], dimensions=[3, 2])) == ["3", "2", "1"]


def test_the_scored_frame_remembers_the_order_in_its_attributes(mini_bfi, mini_processed):
    scored = score_answers(mini_processed, mini_bfi)

    assert list(scored.attrs.values()) == [["Extraversion", "Agreeableness", "Neuroticism"]]


# ---- the overall total ------------------------------------------------------


def test_include_total_adds_the_overall_scale_for_questionnaires_with_dimensions():
    scored = scored_frame([("m", "A", 1.0), ("m", "A", 2.0), ("m", "B", 6.0), ("m", "B", NAN)])

    scales = scale_scores(scored, by="model_id", include_total=True)

    assert scales["dimension"].tolist() == ["A", "B", "total"]
    assert scales["score"].tolist() == [1.5, 6.0, 3.0]  # the mean of all three scores
    assert scales["n_items"].tolist() == [2, 1, 3]
    assert scales["n_missing"].tolist() == [0, 1, 1]


def test_without_include_total_there_is_no_total_for_questionnaires_with_dimensions():
    scored = scored_frame([("m", "A", 1.0), ("m", "B", 6.0)])

    assert scale_scores(scored, by="model_id")["dimension"].tolist() == ["A", "B"]


def test_a_questionnaire_without_dimensions_has_just_the_total():
    scored = scored_frame([("m", None, 1.0), ("m", None, 3.0), ("n", None, 5.0)])

    for include_total in (False, True):
        scales = scale_scores(scored, by="model_id", include_total=include_total)

        assert scales["dimension"].tolist() == ["total", "total"]
        assert scales["score"].tolist() == [2.0, 5.0]


def test_items_without_a_dimension_next_to_items_with_one():
    scored = scored_frame([("m", "A", 1.0), ("m", None, 5.0), ("m", None, 3.0)])

    plain = scale_scores(scored, by="model_id")
    with_total = scale_scores(scored, by="model_id", include_total=True)

    # without include_total, "total" is the scale of the items that have no dimension ...
    assert plain["dimension"].tolist() == ["A", "total"]
    assert plain["score"].tolist() == [1.0, 4.0]
    # ... with it, "total" covers every item, the unassigned ones included
    assert with_total["dimension"].tolist() == ["A", "total"]
    assert with_total["score"].tolist() == [1.0, 3.0]
    assert with_total["n_items"].tolist() == [1, 3]


def test_include_total_is_not_repeated_for_groups_and_keeps_the_sorting():
    scored = scored_frame([("b", "A", 1.0), ("a", "B", 2.0), ("a", "A", 3.0), ("b", "B", 4.0)])

    scales = scale_scores(scored, by="model_id", include_total=True, dimensions=["A", "B"])

    assert scales["model_id"].tolist() == ["a"] * 3 + ["b"] * 3
    assert scales["dimension"].tolist() == ["A", "B", "total"] * 2
    assert scales["score"].tolist() == [3.0, 2.0, 2.5, 1.0, 4.0, 2.5]


def test_a_dimension_that_is_called_total_merges_with_the_total_scale():
    scored = scored_frame([("m", "total", 1.0), ("m", None, 3.0), ("m", "A", 5.0)])

    scales = scale_scores(scored, by="model_id")

    assert scales["dimension"].tolist() == ["A", "total"]
    assert scales["score"].tolist() == [5.0, 2.0]


# ===========================================================================
# score_experiment
# ===========================================================================


PERSONAS = {
    "Optimistic Persona": {
        "attributes": {"name": "Muller", "age": 18},
        "template": "{name} is {age} years old and optimistic.",
    },
    "Conservative Persona": {
        "attributes": {"name": "Grueber", "age": 65},
        "template": "{name} is {age} years old and conservative.",
    },
}


def mini_config(config_dict: dict) -> dict:
    """The six items of the oracle, two personas and one seed (prompt of the test config)."""
    config = copy.deepcopy(config_dict)
    config["parameters"] = {"seeds": ["7"]}
    config["demographic_profiles"] = copy.deepcopy(PERSONAS)
    config["questionnaire"] = {
        "name": "Mini Big Five",
        "general_instruction": "Indicate how much you agree or disagree with each statement.",
        "attributes": {"dimension": dict(MINI_DIMENSIONS)},
        "default_answer_options": copy.deepcopy(LIKERT),
        "instruction_items": copy.deepcopy(MINI_ITEMS),
    }
    return config


def run_fake(config: dict, responses: list[str], tmp_path: Path):
    """Run an experiment with a scripted model; returns the experiment and the files written.

    The model answers the calls in the order of the run: item by item, both personas per item.
    """
    experiment = experiment_from_config(config)
    results = tmp_path / "results.csv"
    with quietly():
        experiment.add_model(FakeListLLM(responses=list(responses)), identifier="fake")
        experiment.run(callbacks=[CSVCallback(str(results))], show_progress=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({**config, "models": {}}), encoding="utf-8")
    return experiment, results, config_file


def postprocess(config_file: Path, results: Path, tmp_path: Path, judge_options: list[str]):
    """Clean, validate and judge the results the way the documentation does."""
    pipeline = PostprocessingPipeline(
        config_file_path=str(config_file),
        results_file_patterns=[str(results)],
        cleaner=BasicCleaner(),
        validator=ValidatorParser(),
        judge=MultipleChoiceJudge(judge_options),
        output_path=str(tmp_path / "processed.csv"),
        show_progress=False,
    )
    with quietly():
        return pipeline.run()


# the model answers with exactly the text of an option (Optimistic, Conservative) per item
EXACT_ANSWERS = [
    "5. Agree strongly",
    "3. Neither agree nor disagree",
    "1. Disagree strongly",
    "4. Agree a little",
    "4. Agree a little",
    "I would rather not say",
    "2. Disagree a little",
    "5. Agree strongly",
    "5. Agree strongly",
    " 3. neither agree nor disagree ",
    "3. Neither agree nor disagree",
    "5. Agree strongly",
]


def test_score_experiment_uses_the_answers_of_the_experiment(config_dict, tmp_path):
    experiment, _, _ = run_fake(mini_config(config_dict), EXACT_ANSWERS, tmp_path)

    scales = score_experiment(experiment)

    # Optimistic:   5, 6-1=5, 4, 6-2=4, 6-5=1, 3  ->  E 5.0, A 4.0, N 2.0
    # Conservative: 3, 6-4=2, - (no option), 6-5=1, 6-3=3 (matched ignoring case), 5
    #               ->  E 2.5, A 1.0 (one item missing), N 4.0
    assert (
        scales["profile_id"].tolist() == ["Conservative Persona"] * 3 + ["Optimistic Persona"] * 3
    )
    assert scales["dimension"].tolist() == ["Extraversion", "Agreeableness", "Neuroticism"] * 2
    assert scales["score"].tolist() == [2.5, 1.0, 4.0, 5.0, 4.0, 2.0]
    assert scales["n_items"].tolist() == [2, 1, 2, 2, 2, 2]
    assert scales["n_missing"].tolist() == [0, 1, 0, 0, 0, 0]
    assert scales["model_id"].unique().tolist() == ["fake"]
    assert scales["random_seed"].unique().tolist() == ["7"]


def test_score_experiment_accepts_the_options_of_the_scoring_functions(config_dict, tmp_path):
    experiment, _, _ = run_fake(mini_config(config_dict), EXACT_ANSWERS, tmp_path)

    scales = score_experiment(
        experiment, by="profile_id", agg="sum", include_total=True, only_valid=False
    )

    assert list(scales.columns) == ["profile_id", "dimension", "score", "n_items", "n_missing"]
    totals = scales[scales["dimension"] == "total"]
    assert totals["profile_id"].tolist() == ["Conservative Persona", "Optimistic Persona"]
    assert totals["score"].tolist() == [3 + 2 + 1 + 3 + 5, 5 + 5 + 4 + 4 + 1 + 3]


def test_raw_free_text_answers_are_not_scored_without_the_pipeline(fake_experiment, config_dict):
    # the model answers "3", which is not the text of an option: the pipeline would judge it
    with quietly():
        fake_experiment.run(show_progress=False)
    n_answers = len(config_dict["questionnaire"]["instruction_items"]) * len(
        config_dict["demographic_profiles"]
    )

    with pytest.warns(UserWarning, match=rf"None of the {n_answers} decisions.*for instance '3'"):
        scales = score_experiment(fake_experiment)

    assert scales["score"].isna().all()
    assert (scales["n_items"] == 0).all()
    assert scales["n_missing"].sum() == n_answers


def test_score_experiment_takes_the_processed_frame_when_it_is_given(mini_bfi, mini_processed):
    scales = score_experiment(mini_bfi, mini_processed, by=["model_id"], include_total=True)

    expected = score_experiment(
        mini_bfi, mini_processed.assign(extra=1), by="model_id", include_total=True
    )
    pd.testing.assert_frame_equal(scales, expected)
    assert scales["model_id"].tolist() == ["m1"] * 4 + ["m2"] * 4


def test_score_experiment_passes_the_column_names_on_to_the_scoring(mini_bfi, mini_processed):
    renamed = mini_processed.rename(
        columns={"instruction_item_id": "item", "decision": "choice", "valid": "ok"}
    )
    renamed.loc[renamed["ok"], "ok"] = True

    scales = score_experiment(
        mini_bfi, renamed, item_column="item", decision_column="choice", valid_column="ok"
    )

    pd.testing.assert_frame_equal(scales, score_experiment(mini_bfi, mini_processed))


def test_score_experiment_honours_only_valid(mini_bfi, mini_processed):
    lenient = score_experiment(mini_bfi, mini_processed, only_valid=False)

    refused = lenient[(lenient["profile_id"] == "p2") & (lenient["dimension"] == "Agreeableness")]
    assert refused["score"].tolist() == [2.0]  # (2 + (6 - 4)) / 2: the refusal is counted
    assert refused["n_items"].tolist() == [2]
    assert refused["n_missing"].tolist() == [0]


def test_a_questionnaire_has_no_answers_to_score(mini_bfi):
    with pytest.raises(ValueError, match="holds no answers.*`processed`"):
        score_experiment(mini_bfi)


def test_an_experiment_that_has_not_run_has_no_scores(config_dict):
    experiment = experiment_from_config(mini_config(config_dict))

    scales = score_experiment(experiment)

    assert list(scales.columns) == [
        "model_id",
        "profile_id",
        "random_seed",
        "dimension",
        "score",
        "n_items",
        "n_missing",
    ]
    assert len(scales) == 0


# ===========================================================================
# properties
# ===========================================================================

TEXTS = ["agree", "Agree", "AGREE", "disagree", "never", "a b", "A  B", "x"]
DECISIONS = [
    *TEXTS,
    " agree ",
    "AGREE\t",
    "A   B",
    "a  b",
    "not present",
    "inconclusive",
    "junk",
    "",
]

option_lists = st.lists(
    st.tuples(st.sampled_from(TEXTS), st.integers(-3, 6), st.booleans()), max_size=6
)
item_specs = st.fixed_dictionaries(
    {"options": st.one_of(st.none(), option_lists), "reversed": st.booleans()}
)
decision_values = st.one_of(st.sampled_from(DECISIONS), st.none(), st.just(NAN))
answer_rows = st.tuples(st.integers(0, 3), decision_values, st.sampled_from([True, False, None]))


def naive_score(
    own: list[tuple[str, int, bool]] | None,
    defaults: list[tuple[str, int, bool]],
    reverse: bool,
    decision: Any,
) -> float:
    """The documented rules for one answer, applied with plain loops."""
    effective = own if own else defaults
    if not isinstance(decision, str) or not decision.strip():
        return NAN
    kept = [weight for _, weight, ignored in effective if not ignored]

    def outcome(weight: int, ignored: bool) -> float | None:
        if ignored:
            return None
        return float(max(kept) + min(kept) - weight) if reverse else float(weight)

    def resolve(matches: list[tuple[str, int, bool]]) -> float:
        outcomes = {outcome(weight, ignored) for _, weight, ignored in matches}
        if len(outcomes) != 1:
            return NAN  # two options with this text score differently
        (only,) = outcomes
        return NAN if only is None else only

    exact = [entry for entry in effective if entry[0] == decision]
    if exact:
        return resolve(exact)
    if decision in JUDGE_SENTINELS:
        return NAN

    def fold(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    similar = [entry for entry in effective if fold(entry[0]) == fold(decision)]
    return resolve(similar) if similar else NAN


@PROPERTIES
@given(
    defaults=option_lists,
    items=st.lists(item_specs, min_size=1, max_size=4),
    rows=st.lists(answer_rows, max_size=30),
    only_valid=st.booleans(),
)
def test_property_the_vectorised_scoring_agrees_with_a_row_by_row_oracle(
    defaults, items, rows, only_valid
):
    def to_options(entries):
        return options(*(option(text, weight, ignored) for text, weight, ignored in entries))

    question = questionnaire(
        [
            item(reversed=spec["reversed"], answer_options=to_options(spec["options"] or []))
            if spec["options"]
            else item(reversed=spec["reversed"])
            for spec in items
        ],
        defaults=to_options(defaults) if defaults else None,
    )
    ids = [position % len(items) for position, _, _ in rows]
    processed = pd.DataFrame(
        {
            "instruction_item_id": pd.Series(ids, dtype="int64"),
            "decision": pd.Series([decision for _, decision, _ in rows], dtype=object),
            "valid": pd.Series([valid for _, _, valid in rows], dtype=object),
        }
    )
    expected = []
    for (_, decision, valid), position in zip(rows, ids):
        spec = items[position]
        score = naive_score(spec["options"], defaults, spec["reversed"], decision)
        expected.append(NAN if only_valid and valid is False else score)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # texts of two options may clash
        scored = score_answers(processed, question, only_valid=only_valid)

    np.testing.assert_array_equal(scored["score"].to_numpy(), np.asarray(expected, dtype=float))
    assert len(scored) == len(rows)


@PROPERTIES
@given(specs=st.lists(st.tuples(st.integers(-6, 9), st.booleans()), min_size=1, max_size=8))
def test_property_reverse_keying_is_scoring_the_flipped_weights(specs):
    weights = [weight for weight, _ in specs]
    ignored = tuple(index for index, (_, skip) in enumerate(specs) if skip)
    kept = [weight for weight, skip in specs if not skip]
    lowest_plus_highest = (min(kept) + max(kept)) if kept else 0
    flipped = [w if skip else lowest_plus_highest - w for w, skip in specs]
    processed = answers([f"opt{i}" for i in range(len(specs))])

    reversed_scores = score_answers(processed, single_item(weights, reversed=True, ignored=ignored))
    flipped_scores = score_answers(processed, single_item(flipped, reversed=False, ignored=ignored))
    plain_scores = score_answers(processed, single_item(weights, reversed=False, ignored=ignored))

    np.testing.assert_array_equal(
        reversed_scores["score"].to_numpy(), flipped_scores["score"].to_numpy()
    )
    for index, (_, skip) in enumerate(specs):
        if skip:
            assert math.isnan(reversed_scores["score"].iloc[index])
        else:  # the two keyings of an option add up to the same number for every option
            total = reversed_scores["score"].iloc[index] + plain_scores["score"].iloc[index]
            assert total == lowest_plus_highest
            assert min(kept) <= reversed_scores["score"].iloc[index] <= max(kept)


score_rows = st.lists(
    st.tuples(
        st.sampled_from(["m1", "m2"]),
        st.sampled_from(["A", "B", None]),
        st.one_of(st.integers(-3, 6).map(float), st.just(NAN)),
    ),
    max_size=40,
)


def naive_scales(
    rows, agg: str, include_total: bool
) -> dict[tuple[str, str], tuple[float, int, int]]:
    """Scale scores with plain loops: (model, scale) -> (score, n_items, n_missing)."""
    groups: dict[tuple[str, str], list[float]] = {}
    for model, dimension, score in rows:
        scale = TOTAL_SCALE if dimension is None else dimension
        groups.setdefault((model, scale), []).append(score)
    if include_total and any(dimension is not None for _, dimension, _ in rows):
        groups = {key: values for key, values in groups.items() if key[1] != TOTAL_SCALE}
        for model, _, score in rows:
            groups.setdefault((model, TOTAL_SCALE), []).append(score)
    result = {}
    for key, values in groups.items():
        present = [value for value in values if not math.isnan(value)]
        if not present:
            score = NAN
        else:
            score = sum(present) / len(present) if agg == "mean" else float(sum(present))
        result[key] = (score, len(present), len(values) - len(present))
    return result


@PROPERTIES
@given(rows=score_rows, agg=st.sampled_from(["mean", "sum"]), include_total=st.booleans())
def test_property_scale_scores_agree_with_a_loop_and_their_counts_add_up(rows, agg, include_total):
    scored = pd.DataFrame(rows, columns=["model_id", "dimension", "score"])

    scales = scale_scores(scored, by="model_id", agg=agg, include_total=include_total)

    expected = naive_scales(rows, agg, include_total)
    actual = {
        (row.model_id, row.dimension): (row.score, row.n_items, row.n_missing)
        for row in scales.itertuples()
    }
    assert actual.keys() == expected.keys()
    for key, (score, n_items, n_missing) in expected.items():
        assert actual[key][1:] == (n_items, n_missing)
        assert actual[key][0] == pytest.approx(score, nan_ok=True)
    assert len(scales) == len(expected)  # one row per model and scale, none twice


@PROPERTIES
@given(rows=score_rows, seed=st.integers(0, 1000))
def test_property_the_order_of_the_rows_does_not_matter(rows, seed):
    scored = pd.DataFrame(rows, columns=["model_id", "dimension", "score"])
    shuffled = scored.sample(frac=1, random_state=seed)

    # without an order, the scales would come in order of appearance, so it is fixed here
    one = scale_scores(scored, by="model_id", include_total=True, dimensions=["B", "A"])
    other = scale_scores(shuffled, by="model_id", include_total=True, dimensions=["B", "A"])

    pd.testing.assert_frame_equal(one, other)


# ===========================================================================
# the configurations that ship with the repository
# ===========================================================================

SHIPPED_CONFIGS = [
    path
    for path in sorted((REPO / "examples" / "data").glob("*.json"))
    if "questionnaire" in json.loads(path.read_text(encoding="utf-8"))
]


def option_tuples(spec: dict | None) -> list[tuple[str, int, bool]] | None:
    """The answer options of a configuration as ``(text, weight, ignored)`` tuples."""
    if not spec:
        return None
    entries = spec["options"] if "options" in spec else spec
    return [
        (
            entry.get("text", "Choose an option"),
            entry.get("weight", 0),
            entry.get("ignored_for_scale", False),
        )
        for entry in entries.values()
    ]


def test_there_are_configurations_to_check():
    assert len(SHIPPED_CONFIGS) >= 8


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_shipped_configuration_is_scored_as_its_weights_say(path):
    spec = json.loads(path.read_text(encoding="utf-8"))["questionnaire"]
    defaults = option_tuples(spec.get("default_answer_options")) or []
    scales = spec.get("attributes", {}).get("dimension", {})
    rows, expected, expected_dimensions = [], [], []
    for position, entry in enumerate(spec["instruction_items"]):
        own = option_tuples(entry.get("answer_options"))
        raw = entry.get("attributes", {}).get("dimension")
        for text, _, _ in own or defaults:  # the model chooses every option once
            rows.append((position, text))
            expected.append(naive_score(own, defaults, entry.get("reversed", False), text))
            expected_dimensions.append(None if raw is None else scales.get(str(raw), str(raw)))
    processed = pd.DataFrame(rows, columns=["instruction_item_id", "decision"])

    scored = score_answers(processed, Questionnaire.model_validate(spec))

    assert len(scored) > 0
    assert_scores(scored, expected)
    assert dimension_list(scored) == expected_dimensions


def test_the_reverse_keyed_items_of_the_regulatory_focus_questionnaire():
    # examples/data/rfq_*.json: items with options of their own ("2." is a complete option text),
    # seven of eleven reverse-keyed, dimensions written as integers next to a mapping of strings
    (path, *_) = [path for path in SHIPPED_CONFIGS if path.name.startswith("rfq_")]
    spec = json.loads(path.read_text(encoding="utf-8"))
    question = Questionnaire.model_validate(spec["questionnaire"])
    items = question.instruction_items
    flipped = next(position for position, entry in enumerate(items) if entry.reversed)
    ordinary = next(position for position, entry in enumerate(items) if not entry.reversed)

    def texts(position: int) -> list[str]:
        return [entry.text for entry in items[position].answer_options.options.values()]

    chosen = [texts(flipped)[index] for index in (0, 1, -1)]  # the first, second and last option
    chosen += [texts(ordinary)[index] for index in (0, 1, -1)]
    processed = answers(chosen, [flipped] * 3 + [ordinary] * 3)

    scored = score_answers(processed, question)

    assert sum(entry.reversed for entry in items) == 7
    assert scored["score"].tolist() == [5.0, 4.0, 1.0, 1.0, 2.0, 5.0]
    assert set(scored["dimension"]) <= {"Promotion", "Prevention"}


def test_the_tutorial_describes_the_shipped_questionnaires_correctly():
    def load(name: str) -> dict:
        return json.loads((REPO / "examples" / "data" / name).read_text(encoding="utf-8"))[
            "questionnaire"
        ]

    bfi = load("bfi_small_and_mid.json")
    assert len(bfi["instruction_items"]) == 44
    assert sum(entry["reversed"] for entry in bfi["instruction_items"]) == 16
    assert len(bfi["attributes"]["dimension"]) == 5

    for name in ("rfq_json_format_small.json", "rfq_model_friendly_small.json"):
        rfq = load(name)
        assert len(rfq["instruction_items"]) == 11
        assert all(entry["answer_options"] for entry in rfq["instruction_items"])
        assert sum(entry["reversed"] for entry in rfq["instruction_items"]) == 7

    bdi = option_tuples(load("bdi_qwen72.json")["instruction_items"][0]["answer_options"])
    assert sorted(weight for _, weight, _ in bdi) == [0, 1, 2, 3]
    gsdb = load("gsdb_new_qwen.json")
    assert len(gsdb["default_answer_options"]) == 7
    assert len(gsdb["attributes"]["dimension"]) == 5


# ===========================================================================
# integration with the experiment and the post-processing pipeline
# ===========================================================================

# Free-text answers of the two personas per item: (Optimistic, Conservative)
FREE_TEXT = [
    ("5", "I choose 3"),  # Is talkative
    ("Answer: 1", "4"),  # Is reserved (reversed)
    ('{"answer": "4. Agree a little"}', "Sorry, I think 4"),  # Is helpful: the second is a refusal
    ("2", "1 or 2"),  # Starts quarrels (reversed): the second is inconclusive
    ("5", "As an AI language model, I do not have opinions."),  # Is relaxed (reversed)
    ("3", "Answer: 5"),  # Gets nervous easily
]


@pytest.fixture
def pipeline_result(config_dict, tmp_path):
    """Experiment, post-processed frame and the files for the free-text answers above."""
    responses = [answer for pair in FREE_TEXT for answer in pair]
    experiment, results, config_file = run_fake(mini_config(config_dict), responses, tmp_path)
    processed = postprocess(config_file, results, tmp_path, LIKERT_TEXTS)
    return experiment, processed, tmp_path


def test_the_pipeline_judges_what_the_scores_are_based_on(pipeline_result):
    _, processed, _ = pipeline_result

    optimistic = processed[processed["profile_id"] == "Optimistic Persona"]
    conservative = processed[processed["profile_id"] == "Conservative Persona"]
    assert optimistic["decision"].tolist() == [
        "5. Agree strongly",
        "1. Disagree strongly",
        "4. Agree a little",
        "2. Disagree a little",
        "5. Agree strongly",
        "3. Neither agree nor disagree",
    ]
    assert conservative["decision"].tolist() == [
        "3. Neither agree nor disagree",
        "4. Agree a little",
        "4. Agree a little",  # a refusal, judged all the same
        "inconclusive",
        "not present",
        "5. Agree strongly",
    ]
    assert conservative["valid"].tolist() == [True, True, False, True, False, True]


def test_run_postprocess_score(pipeline_result):
    experiment, processed, _ = pipeline_result

    scored = score_answers(processed, experiment)
    scales = scale_scores(scored, by=["profile_id"], include_total=True)

    by_profile = {
        profile: scored[scored["profile_id"] == profile]["score"].tolist()
        for profile in ("Optimistic Persona", "Conservative Persona")
    }
    # reversed items 1, 3, 4: 6 - w; the refusal (item 2) and the unjudged answers are missing
    assert by_profile["Optimistic Persona"] == [5, 5, 4, 4, 1, 3]
    np.testing.assert_array_equal(by_profile["Conservative Persona"], [3, 2, NAN, NAN, NAN, 5])
    expected = pd.DataFrame(
        {
            "profile_id": ["Conservative Persona"] * 4 + ["Optimistic Persona"] * 4,
            "dimension": ["Extraversion", "Agreeableness", "Neuroticism", "total"] * 2,
            "score": [2.5, NAN, 5.0, 10 / 3, 5.0, 4.0, 2.0, 22 / 6],
            "n_items": [2, 0, 1, 3, 2, 2, 2, 6],
            "n_missing": [0, 2, 1, 3, 0, 0, 0, 0],
        }
    )
    pd.testing.assert_frame_equal(scales, expected)


def test_the_refusal_is_scored_when_invalid_answers_are_allowed(pipeline_result):
    experiment, processed, _ = pipeline_result

    scales = score_experiment(experiment, processed, by="profile_id", only_valid=False)

    agreeableness = scales[scales["dimension"] == "Agreeableness"].set_index("profile_id")
    assert agreeableness["score"].to_dict() == {
        "Conservative Persona": 4.0,
        "Optimistic Persona": 4.0,
    }
    assert agreeableness["n_missing"].to_dict() == {
        "Conservative Persona": 1,
        "Optimistic Persona": 0,
    }


def test_scores_do_not_change_when_the_processed_file_is_read_back(pipeline_result):
    experiment, processed, tmp_path = pipeline_result
    from_file = pd.read_csv(tmp_path / "processed.csv")

    pd.testing.assert_frame_equal(
        score_answers(processed, experiment)[["score", "dimension"]],
        score_answers(from_file, experiment)[["score", "dimension"]],
    )
    pd.testing.assert_frame_equal(
        score_experiment(experiment, processed), score_experiment(experiment, from_file)
    )


def test_a_questionnaire_alone_is_enough_to_score_a_processed_file(pipeline_result):
    experiment, processed, _ = pipeline_result

    pd.testing.assert_frame_equal(
        score_experiment(experiment.questionnaire, processed),
        score_experiment(experiment, processed),
    )


def test_scoring_with_the_default_test_fixture(fake_experiment, config_dict, tmp_path):
    # The model of the fixture always answers "3". On a 1-5 scale that is worth 3 points whether
    # the item is reverse-keyed or not, so every scale must come out at exactly 3.0.
    results = tmp_path / "results.csv"
    with quietly():
        fake_experiment.run(callbacks=[CSVCallback(str(results))], show_progress=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_dict), encoding="utf-8")
    options = fake_experiment.questionnaire.default_answer_options.get_options_as_list()
    processed = postprocess(config_file, results, tmp_path, options)
    spec = config_dict["questionnaire"]
    names = {str(key): name for key, name in spec["attributes"]["dimension"].items()}
    items_per_scale: dict[str, int] = {}
    for entry in spec["instruction_items"]:
        scale = names[str(entry["attributes"]["dimension"])]
        items_per_scale[scale] = items_per_scale.get(scale, 0) + 1
    profiles = sorted(config_dict["demographic_profiles"])

    scales = score_experiment(fake_experiment, processed)

    assert processed["decision"].unique().tolist() == [options[2]]  # "3. Neither agree nor ..."
    assert scales["profile_id"].tolist() == [p for p in profiles for _ in items_per_scale]
    assert scales["dimension"].tolist() == list(items_per_scale) * len(profiles)
    assert scales["score"].tolist() == [3.0] * len(scales)
    assert scales["n_items"].tolist() == list(items_per_scale.values()) * len(profiles)
    assert (scales["n_missing"] == 0).all()


def example_script(config: dict) -> tuple[list[str], pd.DataFrame]:
    """What the personas of the bundled example answer, and the scores that must come out of it.

    The persona ``p`` chooses the option ``(item + 2 * p)`` (modulo the number of options) for each
    item. The expected scale scores are computed from the JSON with plain loops, so the test keeps
    working when the example grows or changes its scales.
    """
    spec = config["questionnaire"]
    items = spec["instruction_items"]
    assert not any("answer_options" in entry for entry in items), "the example uses the defaults"
    defaults = [
        (entry["text"], entry["weight"], entry.get("ignored_for_scale", False))
        for entry in spec["default_answer_options"].values()
    ]
    names = {str(key): name for key, name in spec["attributes"]["dimension"].items()}
    profiles = list(config["demographic_profiles"])
    chosen = {
        (position, persona): defaults[(position + 2 * persona) % len(defaults)][0]
        for position in range(len(items))
        for persona in range(len(profiles))
    }
    responses = [
        chosen[position, persona]
        for position in range(len(items))
        for persona in range(len(profiles))
    ]

    rows = []
    for persona in sorted(range(len(profiles)), key=lambda index: profiles[index]):
        by_scale: dict[str, list[float]] = {}
        for position, entry in enumerate(items):
            scale = names[str(entry["attributes"]["dimension"])]
            score = naive_score(
                None, defaults, entry.get("reversed", False), chosen[position, persona]
            )
            by_scale.setdefault(scale, []).append(score)
        every_score = [score for scores in by_scale.values() for score in scores]
        for scale, scores in [*by_scale.items(), (TOTAL_SCALE, every_score)]:
            rows.append((profiles[persona], scale, sum(scores) / len(scores), len(scores), 0))
    columns = ["profile_id", "dimension", "score", "n_items", "n_missing"]
    return responses, pd.DataFrame(rows, columns=columns)


def test_the_bundled_bfi_example_scores_end_to_end(tmp_path):
    config = rup.load_example_config("bfi")
    config["models"] = {}
    responses, expected = example_script(config)
    experiment, results, config_file = run_fake(config, responses, tmp_path)
    judge_options = experiment.questionnaire.default_answer_options.get_options_as_list()

    processed = postprocess(config_file, results, tmp_path, judge_options)
    scales = score_experiment(experiment, processed, by="profile_id", include_total=True)

    assert processed["valid"].all()
    assert processed["decision"].tolist() == responses  # the judge found every chosen option
    pd.testing.assert_frame_equal(scales, expected, check_dtype=False)
    assert (scales["n_missing"] == 0).all()


def test_the_bundled_example_scores_from_answers_that_are_exact_option_texts():
    with quietly():
        experiment = rup.load_example_experiment("bfi", models={})
        responses, expected = example_script(rup.load_example_config("bfi"))
        experiment.add_model(FakeListLLM(responses=responses), identifier="fake")
        experiment.run(show_progress=False)

    scales = score_experiment(experiment, by="profile_id", include_total=True)

    pd.testing.assert_frame_equal(scales, expected, check_dtype=False)


# ===========================================================================
# performance
# ===========================================================================


def big_questionnaire(n_items: int) -> Questionnaire:
    """A Big-Five-sized questionnaire: five scales, every third item reverse-keyed."""
    items = [
        item(f"Question {i}", reversed=i % 3 == 0, dimension=str(i % 5 + 1)) for i in range(n_items)
    ]
    scales = {str(i + 1): name for i, name in enumerate("EACNO")}
    return questionnaire(items, defaults=LIKERT, dimensions=scales)


def big_frame(n_rows: int, n_items: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = np.array(
        [*LIKERT_TEXTS, " 3. neither agree nor disagree ", "not present", "inconclusive"],
        dtype=object,
    )
    return pd.DataFrame(
        {
            "instruction_item_id": rng.integers(0, n_items, n_rows),
            "model_id": rng.choice(["model-a", "model-b", "model-c", "model-d"], n_rows),
            "profile_id": rng.choice([f"persona {i}" for i in range(100)], n_rows),
            "random_seed": rng.integers(0, 10, n_rows),
            "decision": pool[rng.integers(0, len(pool), n_rows)],
            "valid": rng.random(n_rows) > 0.1,
        }
    )


def test_200_000_answers_are_scored_in_well_under_two_seconds():
    question = big_questionnaire(44)
    processed = big_frame(200_000, 44)

    start = time.perf_counter()
    scored = score_answers(processed, question)
    middle = time.perf_counter()
    scales = scale_scores(scored, include_total=True)
    end = time.perf_counter()

    # generous: it takes well below a second on a laptop, a Python loop over the rows ~10 times more
    assert middle - start < 2.0, f"score_answers took {middle - start:.2f} s"
    assert end - middle < 2.0, f"scale_scores took {end - middle:.2f} s"
    assert len(scored) == 200_000
    assert scored["score"].notna().mean() == pytest.approx(0.9 * 6 / 8, abs=0.01)
    # 4 models, 100 personas, 10 seeds, 5 scales and the total; with about 50 answers for each of
    # the 4000 respondents a respondent that has none for a scale is rare but possible
    assert 23_900 <= len(scales) <= 4 * 100 * 10 * 6
    assert scales["n_items"].sum() + scales["n_missing"].sum() == 2 * 200_000


def test_the_big_frame_is_scored_correctly_too():
    question = big_questionnaire(44)
    processed = big_frame(3_000, 44, seed=1)

    scored = score_answers(processed, question)

    defaults = [(text, weight, False) for weight, text in enumerate(LIKERT_TEXTS, start=1)]
    reversed_items = {i for i in range(44) if i % 3 == 0}
    expected = [
        NAN if not valid else naive_score(None, defaults, item_id in reversed_items, decision)
        for item_id, decision, valid in zip(
            processed["instruction_item_id"], processed["decision"], processed["valid"]
        )
    ]
    np.testing.assert_array_equal(scored["score"].to_numpy(), np.asarray(expected))


def test_many_distinct_free_text_decisions_are_still_fast():
    # worst case for the look-up: nearly every row has a decision of its own (raw model answers)
    question = big_questionnaire(44)
    processed = big_frame(50_000, 44)
    processed["decision"] = [f"answer number {i}" for i in range(len(processed))]

    start = time.perf_counter()
    with pytest.warns(UserWarning, match="None of the 50000 decisions"):
        scored = score_answers(processed, question)

    assert time.perf_counter() - start < 2.0
    assert scored["score"].isna().all()


# ===========================================================================
# the documentation
# ===========================================================================

FENCE = re.compile(r"^[ ]*```(\w*)[^\n]*\n(.*?)^[ ]*```", re.MULTILINE | re.DOTALL)


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """``(language, code)`` of every fenced code block of a markdown text or docstring."""
    return [(language, textwrap.dedent(code)) for language, code in FENCE.findall(text)]


def run_docstring_example(code: str, namespace: dict[str, Any]) -> int:
    """Run an example statement by statement and check the output the comments announce.

    Full-line comments directly below a ``print(...)`` statement are the output it prints.
    Returns the number of outputs that were checked.
    """
    lines = code.splitlines()
    checked = 0
    for node in ast.parse(code).body:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), quietly():
            exec(compile(ast.Module([node], []), "<example>", "exec"), namespace)
        is_print = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "print"
        )
        if not is_print:
            continue
        expected = []
        for line in lines[node.end_lineno :]:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            expected.append(stripped[2:] if stripped.startswith("# ") else stripped[1:])
        if expected:
            assert [line.rstrip() for line in printed.getvalue().splitlines()] == expected, code
            checked += 1
    return checked


def test_the_examples_in_the_docstrings_run_and_print_what_they_show():
    namespace: dict[str, Any] = {"__name__": "docstring_example"}
    checked = 0
    for owner in (scoring, scoring.score_answers, scoring.scale_scores, scoring.score_experiment):
        for language, code in fenced_blocks(inspect.getdoc(owner) or ""):
            if language == "python":
                checked += run_docstring_example(code, namespace)

    assert checked == 6  # two in the module, one for score_answers, two and one


def test_the_api_page_renders_the_module():
    text = API_PAGE.read_text(encoding="utf-8")

    assert text.startswith("# Scoring")
    assert "::: rupsycho.scoring" in text


def test_the_tutorial_code_runs_and_prints_what_the_page_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the tutorial writes to a temp folder
    blocks = fenced_blocks(TUTORIAL.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "tutorial"}
    shown_outputs = 0

    for position, (language, code) in enumerate(blocks):
        if language != "python":
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), quietly():
            exec(compile(code, f"scoring.md[{position}]", "exec"), namespace)
        following = blocks[position + 1] if position + 1 < len(blocks) else ("", "")
        if following[0] == "text":
            actual = [line.rstrip() for line in printed.getvalue().rstrip().splitlines()]
            shown = [line.rstrip() for line in following[1].rstrip().splitlines()]
            assert actual == shown, f"block {position} prints something else than the page shows"
            shown_outputs += 1

    assert shown_outputs >= 5


def tutorial_table(heading: str) -> list[list[str]]:
    """The body rows of the first table below ``heading`` in the tutorial, as lists of cells."""
    section = TUTORIAL.read_text(encoding="utf-8").split(heading, 1)[1]
    rows = []
    for line in section.splitlines()[1:]:
        if line.startswith("#"):
            break  # the next heading
        if line.startswith("|") and not set(line) <= set("|-: "):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]  # without the header row


def test_the_weight_table_of_the_tutorial_is_what_the_code_computes():
    rows = tutorial_table("### The score of an option")
    reversed_flags = {"no": [False], "yes": [True], "yes or no": [False, True]}

    assert len(rows) >= 9
    for weights_cell, reversed_cell, scores_cell in rows:
        tokens = [token.strip() for token in weights_cell.split(",")]
        weights = [int(token.strip("()")) for token in tokens]
        ignored = tuple(index for index, token in enumerate(tokens) if token.startswith("("))
        listed = scores_cell.split(":")[0].split(",")  # the text after a colon is a comment
        expected = [NAN if token.strip() == "none" else float(token) for token in listed]
        for reverse in reversed_flags[reversed_cell]:
            question = single_item(weights, reversed=reverse, ignored=ignored)
            processed = answers([f"opt{index}" for index in range(len(weights))])
            assert_scores(score_answers(processed, question), expected)


def test_the_questionnaire_in_the_json_snippet_of_the_tutorial_is_scored_as_described():
    (snippet,) = [
        code
        for language, code in fenced_blocks(TUTORIAL.read_text(encoding="utf-8"))
        if language == "json"
    ]
    question = Questionnaire.model_validate(json.loads(snippet)["questionnaire"])
    processed = answers(["2. Agree", "2. Agree", "1. Disagree"], [0, 1, 2])

    scored = score_answers(processed, question)

    assert_scores(scored, [2, 1, 1])  # the second item is reverse-keyed
    assert dimension_list(scored) == ["Extraversion", "Extraversion", "Agreeableness"]
