# postprocessing.py
"""Post-processing of experiment results: clean, validate and judge free-text answers.

Language models answer in free text ("I would say 4, because ..."). The
[`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline] reads the CSV files
written by ``CSVCallback`` and adds, for every answer,

* ``cleaned_answer`` - the answer after the *cleaner*,
* ``validation_status`` - the verdict of the *validator* (a dictionary with the key
  ``"validation_status"`` that is ``"valid"`` or ``"invalid"``), and ``valid`` - a boolean
  shortcut for it,
* ``decision`` - the answer option chosen by the *judge*.

All three stages are LangChain output parsers from [`rupsycho.parsers`][rupsycho.parsers].
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from rupsycho.experiment import ExperimentDocument
from rupsycho.reader import experiment_from_dict

__all__ = ["REQUIRED_COLUMNS", "PostprocessingPipeline"]

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("instruction_item_id", "answer")
"""Columns that every results file must contain (``CSVCallback`` writes them)."""


def _runner(parser: Any) -> Callable[[Any], Any]:
    """Return the callable that applies a LangChain parser, chain or plain parser object."""
    invoke = getattr(parser, "invoke", None)
    return invoke if callable(invoke) else parser.parse  # type: ignore[no-any-return]


class PostprocessingPipeline:
    """Clean, validate and judge the answers stored in result CSV files.

    The judge chooses among the answer options of the question; items that define no options
    of their own use the questionnaire's ``default_answer_options``.

    Args:
        config_file_path: The experiment the results belong to: the path of its JSON
            configuration, a configuration dictionary or an ``ExperimentDocument`` (only its
            questionnaire is used; no model is ever loaded).
        results_file_patterns: A path / glob pattern or a list of them locating the result
            CSV files (as written by ``CSVCallback``).
        cleaner: Output parser (or LangChain chain of parsers, such as
            ``BasicCleaner() | RegexExtractorCleaner(...)``) applied to the raw answer.
        validator: Output parser returning a dictionary with ``"validation_status"``, e.g.
            ``ValidatorParser()``.
        judge: Output parser whose ``parse(text, possible_answers)`` returns the chosen
            answer option, e.g. ``MultipleChoiceJudge(...)``.
        output_path: Where ``run`` writes the processed results.
        errors: ``"raise"`` (default) stops at the first row a parser cannot handle,
            ``"coerce"`` logs it and leaves the cell empty.
        show_progress: Show progress bars.

    Example:
        ```python
        from rupsycho.parsers.cleaners import BasicCleaner
        from rupsycho.parsers.judges import MultipleChoiceJudge
        from rupsycho.parsers.validators import ValidatorParser
        from rupsycho.postprocessing import PostprocessingPipeline

        pipeline = PostprocessingPipeline(
            "config.json",
            "results.csv",
            cleaner=BasicCleaner(),
            validator=ValidatorParser(),
            judge=MultipleChoiceJudge(["1. Disagree", "2. Agree"]),
            output_path="processed.csv",
        )
        processed = pipeline.run()
        ```
    """

    def __init__(
        self,
        config_file_path: str | Path | Mapping[str, Any] | ExperimentDocument,
        results_file_patterns: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cleaner: Any,
        validator: Any,
        judge: Any,
        output_path: str | Path = "processed_results.csv",
        *,
        errors: Literal["raise", "coerce"] = "raise",
        show_progress: bool = True,
    ) -> None:
        if errors not in ("raise", "coerce"):
            raise ValueError(f"errors must be 'raise' or 'coerce', got {errors!r}")
        self.config_file_path = config_file_path
        self.results_file_patterns = [
            os.fspath(pattern)
            for pattern in (
                [results_file_patterns]
                if isinstance(results_file_patterns, (str, os.PathLike))
                else results_file_patterns
            )
        ]
        self.output_path = output_path
        self.cleaner = cleaner
        self.validator = validator
        self.judge = judge
        self.errors = errors
        self.show_progress = show_progress

        # Only the questionnaire is needed: never load (or download) the models of the config
        self.experiment: ExperimentDocument = self._load_experiment(config_file_path)

    @staticmethod
    def _load_experiment(
        config: str | Path | Mapping[str, Any] | ExperimentDocument,
    ) -> ExperimentDocument:
        """Build the experiment that provides the questionnaire, without any model."""
        if isinstance(config, ExperimentDocument):
            return config
        if isinstance(config, Mapping):
            data = dict(config)
        else:
            path = Path(config)
            if not path.is_file():
                raise FileNotFoundError(f"Experiment configuration not found: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
        data["models"] = {}
        data["parameters"] = {**(data.get("parameters") or {}), "lazy_load_models": True}
        return experiment_from_dict(data)

    # ------------------------------------------------------------------ loading

    def _load_csv_files(self) -> pd.DataFrame:
        """Load and concatenate all result files.

        Returns:
            The combined results.

        Raises:
            FileNotFoundError: If no file matches the patterns.
            ValueError: If a file lacks one of ``REQUIRED_COLUMNS``.
        """
        paths: list[str] = []
        for pattern in self.results_file_patterns:
            # an existing file is taken literally (its name may contain glob characters)
            matches = (
                [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern, recursive=True))
            )
            paths.extend(path for path in matches if path not in paths)  # no duplicate rows
        if not paths:
            raise FileNotFoundError(f"No result files match {self.results_file_patterns}")

        frames = []
        for path in paths:
            # Only a blank cell is a missing answer: a model may literally answer "None" or "N/A"
            frame = pd.read_csv(
                path, dtype={"answer": "string"}, keep_default_na=False, na_values=[""]
            )
            missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
            if missing:
                raise ValueError(
                    f"{path} lacks the column(s) {missing}; expected CSVCallback output"
                )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ processing

    def _answer_options(self, item_id: int) -> list[str]:
        """Answer options of an item, falling back to the questionnaire defaults."""
        questionnaire = self.experiment.questionnaire
        if questionnaire is None or not questionnaire.instruction_items:
            raise ValueError("The configuration contains no questionnaire items")
        options = questionnaire.instruction_items[item_id].get_answer_options_as_list()
        if not options and questionnaire.default_answer_options:
            options = questionnaire.default_answer_options.get_options_as_list()
        return options

    def _apply(self, series: pd.Series, function: Callable[[Any], Any], label: str) -> pd.Series:
        """Apply ``function`` to every non-missing value, honouring the error policy."""

        def guarded(value: Any) -> Any:
            if pd.isna(value):
                return None
            try:
                return function(value)
            except Exception as exc:
                if self.errors == "raise":
                    raise
                logger.warning("%s failed for %r: %s", label, value, exc)
                return None

        if self.show_progress:
            from tqdm import tqdm

            tqdm.pandas(desc=label)
            return series.progress_apply(guarded)  # type: ignore[attr-defined,no-any-return]
        return series.apply(guarded)

    def _decide(self, text: Any, possible_answers: list[str]) -> Any:
        """Let the judge choose among ``possible_answers`` (``None`` for missing text)."""
        if pd.isna(text):
            return None
        try:
            return self.judge.parse(text=text, possible_answers=possible_answers)
        except Exception as exc:
            if self.errors == "raise":
                raise
            logger.warning("Judging failed for %r: %s", text, exc)
            return None

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the columns ``cleaned_answer``, ``validation_status``, ``valid`` and ``decision``.

        Args:
            df: The results as loaded by ``_load_csv_files``.

        Returns:
            The same frame with the new columns.
        """
        df["cleaned_answer"] = self._apply(df["answer"], _runner(self.cleaner), "Cleaning")
        df["validation_status"] = self._apply(
            df["cleaned_answer"], _runner(self.validator), "Validating"
        )
        df["valid"] = df["validation_status"].map(
            lambda verdict: (
                isinstance(verdict, dict) and verdict.get("validation_status") == "valid"
            )
        )

        # The options only depend on the item, so look them up once per item
        options = {
            int(item_id): self._answer_options(int(item_id))
            for item_id in df["instruction_item_id"].unique()
        }
        df["decision"] = [
            self._decide(text, options[int(item_id)])
            for text, item_id in zip(df["cleaned_answer"], df["instruction_item_id"])
        ]
        return df

    # ------------------------------------------------------------------ entry point

    def run(self) -> pd.DataFrame:
        """Load the results, process them and write ``output_path``.

        Returns:
            The processed results (also saved as CSV at ``output_path``).

        Raises:
            FileNotFoundError: If no result file matches.
            ValueError: If a result file is not a ``CSVCallback`` output.
        """
        logger.info("Loading results from %s", self.results_file_patterns)
        data = self._process_data(self._load_csv_files())
        data.to_csv(self.output_path, index=False)
        logger.info("Processed results saved to %s", self.output_path)
        return data
