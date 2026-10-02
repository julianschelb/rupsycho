# scoring.py
"""Scores from judged answers: item scores, scale scores and the rules behind them.

A questionnaire carries everything that is needed to score it: every answer option has a
``weight`` and can be ``ignored_for_scale``, an item can be ``reversed`` (reverse-keyed) and
belongs to a ``dimension`` (a scale such as "Extraversion"). This module applies that metadata
to the output of the
[`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline], whose ``decision``
column holds the text of the answer option the judge chose for every free-text answer.

* [`score_answers`][rupsycho.scoring.score_answers] gives every answer an item **score** and the
  **dimension** (scale) of its item.
* [`scale_scores`][rupsycho.scoring.scale_scores] aggregates the item scores to one score per
  respondent (model, persona and seed) and scale.
* [`score_experiment`][rupsycho.scoring.score_experiment] does both in one call.

The rules in short (the [scoring tutorial](../tutorials/scoring.md) explains them with tables):

* The options of an item are its own ``answer_options`` or, if it has none, the questionnaire's
  ``default_answer_options``. A decision is matched with an option text exactly or, failing
  that, ignoring case and white space.
* The score of an option is its ``weight``. On a reversed item it is ``min + max - weight``,
  where ``min`` and ``max`` are taken over the options that are not ignored.
* There is no score (``NaN``) for ignored options, for decisions that match no option (the
  judges' ``"not present"`` and ``"inconclusive"``, missing values, text that is not an option)
  and, by default, for answers that the validator marked as invalid.
* A scale score aggregates the item scores that exist, by default their mean. The number of
  items without score is reported next to it, because a scale built from few answers is weak.

Example:
    ```python
    import pandas as pd

    from rupsycho.models.questionnaire import Questionnaire
    from rupsycho.scoring import scale_scores, score_answers

    questionnaire = Questionnaire(
        name="Mini inventory",
        general_instruction="Rate the statement.",
        attributes={"dimension": {"1": "Extraversion"}},
        default_answer_options={
            "1": {"text": "1. Disagree", "weight": 1},
            "2": {"text": "2. Neutral", "weight": 2},
            "3": {"text": "3. Agree", "weight": 3},
        },
        instruction_items=[
            {"question": "Is talkative", "attributes": {"dimension": "1"}},
            {"question": "Is reserved", "reversed": True, "attributes": {"dimension": "1"}},
        ],
    )
    processed = pd.DataFrame(
        {
            "instruction_item_id": [0, 1],
            "model_id": ["model", "model"],
            "decision": ["3. Agree", "1. Disagree"],
        }
    )
    scored = score_answers(processed, questionnaire)
    print(scored["score"].tolist())
    # [3.0, 3.0]
    print(scale_scores(scored).to_string(index=False))
    # model_id    dimension  score  n_items  n_missing
    #    model Extraversion    3.0        2          0
    ```

    Both answers are worth 3 points: "Agree" on the first item, and "Disagree" on the
    reverse-keyed second item.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rupsycho.experiment import ExperimentDocument
    from rupsycho.models.questionnaire import (
        AnswerOption,
        AnswerOptions,
        InstructionItem,
        Questionnaire,
    )

__all__ = [
    "DEFAULT_GROUPS",
    "JUDGE_SENTINELS",
    "TOTAL_SCALE",
    "scale_scores",
    "score_answers",
    "score_experiment",
]

TOTAL_SCALE = "total"
"""Name of the scale that covers the items without a dimension (or, on request, all items)."""

DEFAULT_GROUPS: tuple[str, ...] = ("model_id", "profile_id", "random_seed")
"""Columns that identify one simulated respondent: the model, the persona and the seed."""

JUDGE_SENTINELS: tuple[str, ...] = ("not present", "inconclusive")
"""Decisions of the judges that mean "no answer option was chosen"; they never get a score."""

_DIMENSIONS_ATTR = "rupsycho_dimensions"
"""Key in ``DataFrame.attrs`` under which ``score_answers`` records the order of the scales."""

_NO_DECISION, _NO_MATCH, _MATCH = 0, 1, 2
"""What a decision is: missing, blank or a sentinel / text that is no option / an option's text."""

_FALSE_WORDS = frozenset({"false", "f", "no", "n", "0"})
"""Spellings of "not valid" in a ``valid`` column that was saved as text."""

_RAW_COLUMNS = {
    "Instruction ID": "instruction_item_id",
    "Instruction Question": "instruction_item",
    "Model ID": "model_id",
    "Persona ID": "profile_id",
    "Run Seed": "random_seed",
    "Answer": "answer",
}
"""``get_answers_as_dataframe`` columns and the post-processing columns they correspond to."""


# ===========================================================================
# Small helpers
# ===========================================================================


def _fold(text: str) -> str:
    """Normalise ``text`` for the lenient comparison: collapsed white space, case-folded."""
    return " ".join(text.split()).casefold()


def _identifier(value: object) -> str:
    """A dimension id as text (``1`` and ``1.0`` are ``"1"``); ``""`` for a missing one."""
    if value is None:
        return ""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _as_text(value: object) -> str | None:
    """The text of a decision, or ``None`` if the value cannot name an answer option.

    Numbers are compared as text (``4`` and ``4.0`` are ``"4"``), because a CSV round trip turns
    options such as ``"4"`` into numbers.
    """
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        return str(int(number)) if number.is_integer() else str(number)
    return None


def _is_false(value: object) -> bool:
    """Whether a ``valid`` cell says "not valid": ``False``, ``0`` or a text such as ``"False"``."""
    if isinstance(value, str):
        return value.strip().lower() in _FALSE_WORDS
    if isinstance(value, bool | int | float | np.bool_ | np.integer | np.floating):
        return bool(value == 0)
    return False


def _per_code(values: Sequence[Any], codes: np.ndarray, missing: Any) -> np.ndarray:
    """Look up one value per code of ``pd.factorize``; the code ``-1`` (missing) gets ``missing``.

    The missing value is appended to the table, where the index ``-1`` finds it.
    """
    return np.array([*values, missing])[codes]


def _factorize(column: pd.Series, role: str) -> tuple[np.ndarray, list[Any]]:
    """The distinct values of ``column`` and a code per row (``-1`` for a missing value)."""
    try:
        codes, uniques = pd.factorize(column.astype(object))
    except TypeError as error:  # lists and dictionaries cannot be told apart cheaply
        raise TypeError(
            f"The {role} column {column.name!r} must hold single values such as text, numbers "
            f"or booleans, not collections ({error})"
        ) from error
    return codes, list(uniques)


def _questionnaire_of(experiment: object) -> Questionnaire:
    """The questionnaire of an experiment (a questionnaire is returned as it is)."""
    from rupsycho.models.questionnaire import Questionnaire

    if isinstance(experiment, Questionnaire):
        return experiment
    if not hasattr(experiment, "questionnaire"):
        raise TypeError(
            "experiment must be an ExperimentDocument or a Questionnaire, "
            f"got {type(experiment).__name__}"
        )
    questionnaire = experiment.questionnaire
    if questionnaire is None:
        raise ValueError("The experiment has no questionnaire to score the answers with")
    return questionnaire  # type: ignore[no-any-return]


def _column(frame: pd.DataFrame, name: str, role: str) -> pd.Series:
    """The column ``name`` of ``frame``; a missing or ambiguous column is a ``ValueError``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(frame).__name__}")
    if name not in frame.columns:
        raise ValueError(
            f"The {role} column {name!r} is missing; the frame has the columns "
            f"{[str(column) for column in frame.columns]}"
        )
    column = frame[name]
    if isinstance(column, pd.DataFrame):
        raise ValueError(f"The {role} column {name!r} occurs more than once")
    return column


# ===========================================================================
# From answer options to scores (one look-up table per item)
# ===========================================================================


@dataclass(frozen=True)
class _ItemTable:
    """What a decision is worth on one item.

    Attributes:
        exact: Score by option text.
        folded: Score by option text with case and white space ignored (the fallback).
        conflicts: Texts that several options of the item share although they score differently.
    """

    exact: dict[str, float]
    folded: dict[str, float]
    conflicts: tuple[str, ...]

    def lookup(self, decision: object) -> tuple[float, int]:
        """The score of ``decision`` (``NaN`` if there is none) and what the decision is."""
        text = _as_text(decision)
        if text is None:
            return math.nan, _NO_DECISION
        if text in self.exact:
            return self.exact[text], _MATCH
        if text in JUDGE_SENTINELS:
            # a judge's "not present" must not be mistaken for an option called "Not present"
            return math.nan, _NO_DECISION
        folded = _fold(text)
        if folded in self.folded:
            return self.folded[folded], _MATCH
        return math.nan, _NO_MATCH


def _options_of(item: InstructionItem, defaults: AnswerOptions | None) -> list[AnswerOption]:
    """The answer options of an item: its own, or the questionnaire's defaults."""
    own = item.answer_options
    if own is not None and own.options:
        return list(own.options.values())
    if defaults is not None:
        return list(defaults.options.values())
    return []


def _resolve(outcomes: set[float | None]) -> float:
    """The score of a text whose options score ``outcomes`` (``NaN`` if ignored or ambiguous)."""
    if len(outcomes) == 1:
        (outcome,) = outcomes
        if outcome is not None:
            return outcome
    return math.nan


def _item_table(item: InstructionItem, defaults: AnswerOptions | None) -> _ItemTable:
    """Build the look-up table of one item from its answer options."""
    options = _options_of(item, defaults)
    kept = [float(option.weight) for option in options if not option.ignored_for_scale]
    flip = min(kept) + max(kept) if item.reversed and kept else None

    outcomes: dict[str, set[float | None]] = {}
    for option in options:
        if not option.text.strip():
            continue  # a blank text can never be a decision
        outcome: float | None
        if option.ignored_for_scale:
            outcome = None
        elif flip is None:
            outcome = float(option.weight)
        else:
            outcome = flip - option.weight
        outcomes.setdefault(option.text, set()).add(outcome)

    folded: dict[str, set[float | None]] = {}
    for text, scores in outcomes.items():
        folded.setdefault(_fold(text), set()).update(scores)
    return _ItemTable(
        exact={text: _resolve(scores) for text, scores in outcomes.items()},
        folded={text: _resolve(scores) for text, scores in folded.items()},
        conflicts=tuple(text for text, scores in outcomes.items() if len(scores) > 1),
    )


def _item_positions(column: pd.Series, n_items: int, name: str) -> np.ndarray:
    """The item ids of ``column`` as positions in the questionnaire (``ValueError`` if invalid)."""
    numbers = pd.to_numeric(column, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    valid = np.isfinite(numbers) & (numbers == np.floor(numbers))
    valid &= (numbers >= 0) & (numbers < n_items)
    if not valid.all():
        bad = column.iloc[np.flatnonzero(~valid)]
        examples = ", ".join(repr(value) for value in bad.drop_duplicates().head(5))
        if n_items == 0:
            reason = "the questionnaire has no instruction items"
        elif n_items == 1:
            reason = "the questionnaire has one item, so the only valid id is 0"
        else:
            reason = (
                f"the questionnaire has {n_items} items, so ids are integers from 0 to "
                f"{n_items - 1}"
            )
        raise ValueError(
            f"Column {name!r} has {len(bad)} invalid item id(s), e.g. {examples}; {reason}. "
            "Is this the experiment the results belong to?"
        )
    return numbers.astype(np.int64)


def _lookup_scores(
    positions: np.ndarray, decisions: pd.Series, tables: Mapping[int, _ItemTable]
) -> tuple[np.ndarray, int, int, object]:
    """Score every row without a Python loop over the rows.

    Only the distinct (item, decision) pairs are looked up; there are few of them (the options of
    the items plus a handful of sentinels), and the scores are spread back with array indexing.

    Returns:
        The scores, the number of rows whose decision is the text of an option, the number whose
        decision is a text that matches no option, and the first such text (``None`` if none).
    """
    codes, texts = _factorize(decisions, "decision")
    width = len(texts) + 1  # code 0 stands for a missing decision, code k for texts[k - 1]
    pairs = positions * width + (codes + 1)
    distinct, inverse, counts = np.unique(pairs, return_inverse=True, return_counts=True)
    scores = np.full(len(distinct), np.nan)
    rows = [0, 0, 0]  # the number of rows per kind of decision
    example = None
    for index, pair in enumerate(distinct.tolist()):
        position, code = divmod(pair, width)
        kind = _NO_DECISION
        if code:
            scores[index], kind = tables[position].lookup(texts[code - 1])
        if kind == _NO_MATCH and example is None:
            example = texts[code - 1]
        rows[kind] += int(counts[index])
    return scores[inverse], rows[_MATCH], rows[_NO_MATCH], example


def _invalid_rows(valid: pd.Series) -> np.ndarray:
    """Rows whose ``valid`` cell says "not valid"; a missing cell does not exclude the row."""
    codes, verdicts = _factorize(valid, "valid")
    return _per_code([_is_false(value) for value in verdicts], codes, missing=False)


def _scale_labels(questionnaire: Questionnaire) -> tuple[list[str | None], list[str]]:
    """The scale of every item (``None`` for an item without dimension) and the scale order."""
    declared = questionnaire.attributes.get("dimension")
    names: dict[str, str] = {}
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            if identifier := _identifier(key):
                names[identifier] = _identifier(value) or identifier

    labels: list[str | None] = []
    for item in questionnaire.instruction_items or []:
        identifier = _identifier(item.attributes.get("dimension"))
        labels.append(names.get(identifier, identifier) if identifier else None)
    order = list(dict.fromkeys([*names.values(), *(label for label in labels if label)]))
    return labels, order


# ===========================================================================
# Public API
# ===========================================================================


def score_answers(
    processed: pd.DataFrame,
    experiment: ExperimentDocument | Questionnaire,
    *,
    decision_column: str = "decision",
    item_column: str = "instruction_item_id",
    valid_column: str = "valid",
    only_valid: bool = True,
) -> pd.DataFrame:
    """Give every answer an item score and the scale (dimension) of its item.

    The item of a row is found by its position in the questionnaire (``item_column``, the
    ``instruction_item_id`` that ``CSVCallback`` writes). The ``decision`` is matched with the
    texts of the item's answer options, and the weight of the matching option is the score.

    * **Options.** The options of an item are its own ``answer_options``; an item without
      options of its own (or with an empty set of them) uses the questionnaire's
      ``default_answer_options``.
    * **Matching.** A decision matches an option text exactly, or, if there is no exact match,
      when both are equal after collapsing white space and ignoring case. The judge's
      sentinels ``"not present"`` and ``"inconclusive"`` are only ever matched exactly, so that
      they cannot be mistaken for an option called "Not present". Numbers are compared as text
      (``4`` and ``4.0`` match the option ``"4"``). Options with the same text that score
      differently cannot be told apart: such a text is not scored and a ``UserWarning`` names
      the items.
    * **Reverse keying.** The score of a reversed item is ``min + max - weight``, with ``min``
      and ``max`` taken over the options that are not ignored (a 1-5 scale turns 2 into 4; a
      0-3 scale turns 0 into 3).
    * **No score.** The score is ``NaN`` for an ignored option (``ignored_for_scale``), for a
      decision that matches no option (sentinels, missing values, text that is not an option),
      for an item with no scored option, and, if ``only_valid`` is true and the frame has the
      ``valid_column``, for rows with ``valid == False``. A missing ``valid`` cell does not
      exclude a row.
    * **Dimension.** The ``"dimension"`` attribute of an item is looked up in the
      questionnaire's ``attributes["dimension"]`` mapping (``{"1": "Extraversion"}``) to get
      the name of the scale; an id that is not listed is used as it is (as text, so ``1`` and
      ``"1"`` are the same id), and an item without dimension gets ``None``.

    The scoring is vectorised: only the distinct (item, decision) pairs are looked up, so a
    frame with hundreds of thousands of rows is scored in a fraction of a second.

    Args:
        processed: The post-processed answers, one row per answer, e.g. the result of
            [`PostprocessingPipeline.run`][rupsycho.postprocessing.PostprocessingPipeline.run]
            (or the CSV it wrote, read with ``pandas.read_csv``).
        experiment: The experiment the answers belong to, or just its questionnaire.
        decision_column: Column with the chosen answer option (its text).
        item_column: Column with the position of the item in the questionnaire (0-based).
        valid_column: Column with the validator's verdict (``True`` for a usable answer).
        only_valid: Give no score to rows with ``valid == False``. Has no effect if the frame
            has no ``valid_column``.

    Returns:
        A copy of ``processed`` with two more columns, ``"score"`` (float, ``NaN`` where the
        row cannot be scored) and ``"dimension"`` (the scale name, ``None`` for items without
        dimension). Columns of these names are replaced. The order of the scales is recorded in
        ``result.attrs`` so that [`scale_scores`][rupsycho.scoring.scale_scores] lists the
        scales in the order of the questionnaire.

    Raises:
        TypeError: If ``processed`` is not a data frame, ``experiment`` is neither an
            experiment nor a questionnaire, or the decision or validity column holds lists or
            dictionaries (for instance the ``validation_status`` column instead of ``valid``).
        ValueError: If the item or decision column is missing, if an item id is not an
            integer position of the questionnaire (a hint that the experiment does not match
            the results), or if the experiment has no questionnaire.

    Warns:
        UserWarning: If options of an item share a text but not their score, or if no decision
            at all is the text of an option (a hint that the raw answers or the wrong
            experiment were passed).

    Example:
        ```python
        import pandas as pd

        from rupsycho.models.questionnaire import Questionnaire
        from rupsycho.scoring import score_answers

        questionnaire = Questionnaire(
            name="Mini inventory",
            general_instruction="Rate the statement.",
            attributes={"dimension": {"1": "Extraversion"}},
            default_answer_options={
                "1": {"text": "1. Disagree", "weight": 1},
                "2": {"text": "2. Neutral", "weight": 2},
                "3": {"text": "3. Agree", "weight": 3},
                "4": {"text": "4. Don't know", "weight": 0, "ignored_for_scale": True},
            },
            instruction_items=[
                {"question": "Is talkative", "attributes": {"dimension": "1"}},
                {"question": "Is reserved", "reversed": True, "attributes": {"dimension": "1"}},
            ],
        )
        processed = pd.DataFrame(
            {
                "instruction_item_id": [0, 0, 1, 1, 1],
                "decision": [
                    "3. Agree",
                    " 2. neutral ",
                    "1. Disagree",
                    "4. Don't know",
                    "not present",
                ],
                "valid": [True, True, True, True, False],
            }
        )
        scored = score_answers(processed, questionnaire)
        print(scored[["decision", "score", "dimension"]].to_string())
        #         decision  score     dimension
        # 0       3. Agree    3.0  Extraversion
        # 1    2. neutral     2.0  Extraversion
        # 2    1. Disagree    3.0  Extraversion
        # 3  4. Don't know    NaN  Extraversion
        # 4    not present    NaN  Extraversion
        ```

        Row 1 matches although its case and white space differ, row 2 is reverse-keyed (1 becomes
        3), and rows 3 and 4 get no score: an ignored option and a judge's sentinel.
    """
    questionnaire = _questionnaire_of(experiment)
    items = questionnaire.instruction_items or []
    item_ids = _column(processed, item_column, "item id")
    decisions = _column(processed, decision_column, "decision")

    positions = _item_positions(item_ids, len(items), item_column)
    tables = {
        position: _item_table(items[position], questionnaire.default_answer_options)
        for position in np.unique(positions).tolist()
    }
    scores, n_matched, n_unmatched, example = _lookup_scores(positions, decisions, tables)
    if n_unmatched and not n_matched:
        warnings.warn(
            f"None of the {n_unmatched} decisions in column {decision_column!r} is the text of an "
            f"answer option of its item (for instance {example!r}), so no answer gets a score. "
            "Pass the chosen answer options, such as the output of the PostprocessingPipeline, "
            "and the experiment the answers belong to.",
            UserWarning,
            stacklevel=2,
        )
    if only_valid and valid_column in processed.columns:
        scores[_invalid_rows(_column(processed, valid_column, "valid"))] = np.nan

    conflicts = {position: table.conflicts for position, table in tables.items() if table.conflicts}
    if conflicts:
        detail = "; ".join(
            f"item {position}: {', '.join(repr(text) for text in texts)}"
            for position, texts in conflicts.items()
        )
        warnings.warn(
            "Answer options with the same text but different scores cannot be told apart, so "
            f"decisions with these texts get no score ({detail})",
            UserWarning,
            stacklevel=2,
        )

    labels, order = _scale_labels(questionnaire)
    result = processed.copy()
    result["score"] = scores
    result["dimension"] = np.array(labels, dtype=object)[positions]
    result.attrs[_DIMENSIONS_ATTR] = order
    return result


def scale_scores(
    scored: pd.DataFrame,
    *,
    by: str | Sequence[str] = DEFAULT_GROUPS,
    dimension_column: str = "dimension",
    score_column: str = "score",
    agg: str | Callable[[pd.Series], Any] = "mean",
    include_total: bool = False,
    dimensions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate item scores to scale scores, one row per respondent and scale.

    A respondent is one combination of the ``by`` columns (by default model, persona and
    seed), a scale is a dimension. The score of a scale is the aggregate (``agg``, by default
    the mean) of the item scores that exist; items without score (see
    [`score_answers`][rupsycho.scoring.score_answers]) are not counted as zero but reported in
    ``n_missing``. A scale without any item score gets the score ``NaN``, whatever ``agg`` is.

    * Items without dimension form the scale ``"total"``. If ``include_total`` is true and
      there are items with a dimension, ``"total"`` is instead the aggregate over *all* item
      scores (not the mean of the scale scores).
    * Columns of ``by`` that the frame does not have are skipped, so results without a
      ``random_seed`` column can be aggregated as they are. The three standard columns are
      skipped silently, any other missing name triggers a ``UserWarning`` (a typo would
      otherwise merge respondents).
    * Rows with a missing value in a ``by`` column form a group of their own.
    * The result is sorted by the ``by`` columns, then by scale: in the order of the
      questionnaire (recorded by ``score_answers``, or the order given in ``dimensions``),
      scales that are not listed in the order of appearance, and ``"total"`` last.

    Items of one scale should use the same weights; the mean of items scored 1-5 and items
    scored 0-3 is hard to interpret.

    Args:
        scored: Item scores, usually the result of ``score_answers``.
        by: Column(s) that identify a respondent. A single name is accepted.
        dimension_column: Column with the scale of each row; missing values and blank text mean
            "no dimension". The result uses the same column name.
        score_column: Column with the item scores. The result uses the same column name.
        agg: How to aggregate the item scores: ``"mean"``, ``"sum"``, any other name of a pandas
            aggregation (``"median"``, ``"std"``, ``"max"``, ...) or a function from a series
            to a number.
        include_total: Also report the overall ``"total"`` of questionnaires with dimensions.
        dimensions: Order of the scales, if it should differ from the questionnaire's.

    Returns:
        A data frame with the ``by`` columns that exist, ``dimension_column`` (scale name),
        ``score_column`` (the aggregate), ``n_items`` (how many item scores went into it) and
        ``n_missing`` (how many rows had no score). It has a fresh index and no rows if
        ``scored`` has none.

    Raises:
        TypeError: If ``scored`` is not a data frame or its dimension column holds lists or
            dictionaries.
        ValueError: If the dimension or score column is missing, if the scores are not
            numbers, if ``by`` contains the dimension or score column, or if ``agg`` is not a
            valid aggregation.

    Warns:
        UserWarning: If ``by`` names a column other than the three standard ones that the frame
            does not have.

    Example:
        ```python
        import pandas as pd

        from rupsycho.scoring import scale_scores

        scored = pd.DataFrame(
            {
                "model_id": ["a", "a", "a", "a", "b", "b"],
                "dimension": ["Extraversion", "Extraversion", "Neuroticism", "Neuroticism"]
                + ["Extraversion", "Neuroticism"],
                "score": [5.0, 3.0, 2.0, float("nan"), 4.0, 1.0],
            }
        )
        print(scale_scores(scored, by="model_id").to_string(index=False))
        # model_id    dimension  score  n_items  n_missing
        #        a Extraversion    4.0        2          0
        #        a  Neuroticism    2.0        1          1
        #        b Extraversion    4.0        1          0
        #        b  Neuroticism    1.0        1          0
        totals = scale_scores(scored, by="model_id", agg="sum", include_total=True)
        print(totals.to_string(index=False))
        # model_id    dimension  score  n_items  n_missing
        #        a Extraversion    8.0        2          0
        #        a  Neuroticism    2.0        1          1
        #        a        total   10.0        3          1
        #        b Extraversion    4.0        1          0
        #        b  Neuroticism    1.0        1          0
        #        b        total    5.0        2          0
        ```
    """
    dimension = _column(scored, dimension_column, "dimension")
    scores = _column(scored, score_column, "score")
    try:
        points = scores.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(f"The score column {score_column!r} must contain numbers") from error

    wanted = list(dict.fromkeys([by] if isinstance(by, str) else by))
    if {dimension_column, score_column} & set(wanted):
        raise ValueError("by must not contain the dimension column or the score column")
    group_columns = [name for name in wanted if name in scored.columns]
    unknown = [name for name in wanted if name not in scored.columns and name not in DEFAULT_GROUPS]
    if unknown:
        warnings.warn(
            f"scale_scores ignores the unknown group column(s) {unknown}",
            UserWarning,
            stacklevel=2,
        )

    # The scale of every row becomes an integer rank, which sorts in the order of the scales
    order = scored.attrs.get(_DIMENSIONS_ATTR, ()) if dimensions is None else dimensions
    listed = [name for name in dict.fromkeys(map(str, order)) if name != TOTAL_SCALE]
    rank = {name: index for index, name in enumerate(listed)}
    codes, found = _factorize(dimension, "dimension")
    names = [_identifier(value) or TOTAL_SCALE for value in found]
    for name in names:
        if name != TOTAL_SCALE:
            rank.setdefault(name, len(rank))
    rank[TOTAL_SCALE] = total_rank = len(rank)
    row_ranks = _per_code([rank[name] for name in names], codes, missing=total_rank)

    key_names = [f"key{index}" for index in range(len(group_columns))]
    work = pd.DataFrame(
        {
            **{
                key: _column(scored, name, "group").array
                for key, name in zip(key_names, group_columns)
            },
            "rank": row_ranks,
            "score": points,
        }
    )
    if include_total and (row_ranks != total_rank).any():
        work = pd.concat(
            [work[work["rank"] != total_rank], work.assign(rank=total_rank)], ignore_index=True
        )

    grouped = work.groupby([*key_names, "rank"], sort=True, dropna=False, observed=True)
    try:
        table = grouped["score"].agg(score=agg, n_items="count", n_rows="size").reset_index()
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot aggregate the scores with agg={agg!r}: {error}") from error

    result = table[key_names].copy()
    result.columns = pd.Index(group_columns)
    label = {index: name for name, index in rank.items()}
    result[dimension_column] = [label[value] for value in table["rank"]]
    result[score_column] = table["score"].where(table["n_items"] > 0)
    result["n_items"] = table["n_items"]
    result["n_missing"] = table["n_rows"] - table["n_items"]
    return result.reset_index(drop=True)


def score_experiment(
    experiment: ExperimentDocument | Questionnaire,
    processed: pd.DataFrame | None = None,
    *,
    decision_column: str = "decision",
    item_column: str = "instruction_item_id",
    valid_column: str = "valid",
    only_valid: bool = True,
    by: str | Sequence[str] = DEFAULT_GROUPS,
    agg: str | Callable[[pd.Series], Any] = "mean",
    include_total: bool = False,
) -> pd.DataFrame:
    """Score an experiment: item scores from ``score_answers``, then ``scale_scores``.

    Pass the output of the
    [`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline] as
    ``processed``. Without it, the answers stored in the experiment
    (``experiment.get_answers_as_dataframe()``) are used, and the raw answer text is taken as the
    decision. That only works if the model answered with exactly the text of an answer option
    (and was not asked for anything else, such as a JSON object); real free-text answers need to
    go through the pipeline first (``score_answers`` warns if not a single answer is an option).
    Calls that failed have no row in the stored answers, so they are not counted in
    ``n_missing``.

    Args:
        experiment: The experiment (or its questionnaire) with the scoring metadata. Without
            ``processed`` it must be an experiment that holds the answers.
        processed: The post-processed answers. Default: the answers of ``experiment`` (see above).
        decision_column: See [`score_answers`][rupsycho.scoring.score_answers]; refers to
            ``processed`` (the frame built from the experiment uses the standard names).
        item_column: See ``score_answers``; refers to ``processed``.
        valid_column: See ``score_answers``; refers to ``processed``.
        only_valid: See ``score_answers``.
        by: See [`scale_scores`][rupsycho.scoring.scale_scores].
        agg: See ``scale_scores``.
        include_total: See ``scale_scores``.

    Returns:
        The scale scores, see ``scale_scores``.

    Raises:
        ValueError: If ``processed`` is omitted and ``experiment`` holds no answers (it is a
            questionnaire), plus the errors of ``score_answers`` and ``scale_scores``.

    Warns:
        UserWarning: The warnings of ``score_answers`` and ``scale_scores``.

    Example:
        ```python
        import rupsycho as rup
        from langchain_core.language_models.fake import FakeListLLM

        from rupsycho.scoring import score_experiment

        experiment = rup.ExperimentDocument(
            name="Mini inventory",
            models={},
            demographic_profiles={
                "Anna": {"attributes": {"name": "Anna", "age": 30}},
                "Ben": {"attributes": {"name": "Ben", "age": 45}},
            },
            questionnaire={
                "name": "Mini inventory",
                "general_instruction": "Rate the statement.",
                "attributes": {"dimension": {"1": "Extraversion"}},
                "default_answer_options": {
                    "1": {"text": "1. Disagree", "weight": 1},
                    "2": {"text": "2. Neutral", "weight": 2},
                    "3": {"text": "3. Agree", "weight": 3},
                },
                "instruction_items": [
                    {"question": "Is talkative", "attributes": {"dimension": "1"}},
                    {"question": "Is reserved", "reversed": True, "attributes": {"dimension": "1"}},
                ],
            },
        )
        experiment.add_model(FakeListLLM(responses=["3. Agree"]), identifier="agreeable-model")
        experiment.run(show_progress=False)
        print(score_experiment(experiment, by="profile_id").to_string(index=False))
        # profile_id    dimension  score  n_items  n_missing
        #       Anna Extraversion    2.0        2          0
        #        Ben Extraversion    2.0        2          0
        ```

        A model that agrees with everything lands in the middle of the scale, because the
        second item is reverse-keyed.
    """
    _questionnaire_of(experiment)  # fail early, before any answers are collected
    if processed is None:
        get_answers = getattr(experiment, "get_answers_as_dataframe", None)
        if get_answers is None:
            raise ValueError(
                "A questionnaire holds no answers: pass the post-processed answers as "
                "`processed`, or an experiment that has been run"
            )
        processed = get_answers().rename(columns=_RAW_COLUMNS)
        processed["decision"] = processed["answer"]
        decision_column, item_column, valid_column = "decision", "instruction_item_id", "valid"

    scored = score_answers(
        processed,
        experiment,
        decision_column=decision_column,
        item_column=item_column,
        valid_column=valid_column,
        only_valid=only_valid,
    )
    return scale_scores(scored, by=by, agg=agg, include_total=include_total)
