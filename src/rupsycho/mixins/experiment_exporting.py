# ===========================================================================
#                        ExperimentExportMixin Implementation
# ===========================================================================
# This mixin provides methods to export the experiment results to a file or
# return the answers in various formats. It supports exporting to JSON files
# and converting the experiment data into a pandas DataFrame.


from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

CONFIG_FIELDS = (
    "name",
    "description",
    "parameters",
    "models",
    "prompt_template",
    "demographic_profiles",
    "questionnaire",
    "metadata",
)
"""Fields that make up an experiment configuration (``metadata`` only when it is not empty)."""

ANSWER_COLUMNS = [
    "Instruction ID",
    "Instruction Question",
    "Model ID",
    "Persona ID",
    "Run Seed",
    "Answer",
]
"""Columns of the DataFrame returned by ``get_answers_as_dataframe``."""


class ExperimentExportMixin:
    """
    Mixin providing methods to export the experiment results to a file or return the answers.
    """

    if TYPE_CHECKING:
        # Provided by ExperimentDocument, which mixes this class in.
        questionnaire: Any

    def to_config(self, *, include_answers: bool = True) -> dict[str, Any]:
        """Return the experiment as a plain, JSON-serialisable configuration dictionary.

        Only the configuration is exported (name, description, parameters, models, prompt
        template, personas, questionnaire), never runtime objects. Secrets such as API keys
        are masked. The result can be passed to
        [`experiment_from_dict`][rupsycho.reader.experiment_from_dict], so an exported
        experiment can be shared, versioned and re-loaded.

        Args:
            include_answers: Keep the answers collected so far on the questionnaire items.
                Pass ``False`` to export only the experiment *definition*.

        Returns:
            The configuration dictionary.

        Example:
            ```python
            config = experiment.to_config(include_answers=False)
            ```
        """
        exclude = {"questionnaire": {"instruction_items": {"__all__": {"answers"}}}}
        data: dict[str, Any] = self.model_dump(  # type: ignore[attr-defined]
            mode="json",
            include=set(CONFIG_FIELDS),
            exclude=None if include_answers else exclude,
            exclude_none=True,
        )
        if not data.get("metadata"):
            data.pop("metadata", None)
        return data

    def export_to_file(
        self, filename: str | os.PathLike[str], *, include_answers: bool = True
    ) -> None:
        """Write the experiment configuration (and answers) to a JSON file.

        Args:
            filename: Target file; it is written as UTF-8 and overwritten if it exists.
            include_answers: Keep the answers collected so far; see ``to_config``.

        Raises:
            OSError: If the file cannot be written.
        """
        data = self.to_config(include_answers=include_answers)
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def get_answers(self) -> list[dict[str, Any]]:
        """
        Extracts and returns the answers from the experiment in a list holding the nested answer structure.

        :return: A list of dictionaries representing the answers for each instruction item.
        """
        answers = []
        for item in self.questionnaire.instruction_items:
            answers.append(item.model_dump().get("answers", {}))
        return answers

    def get_answers_as_dataframe(self) -> pd.DataFrame:
        """
        Returns a flat pandas DataFrame with the experiment's answers.

        Columns include:
        - "Instruction ID"
        - "Instruction Question"
        - "Model ID"
        - "Persona ID"
        - "Run Seed"
        - "Answer"

        :return: A pandas DataFrame containing the flattened answers.
        """
        data = []

        # Loop through each instruction item
        for instruction_id, item in enumerate(self.questionnaire.instruction_items):
            question = item.question
            answers = item.answers

            # Loop through models
            for model_id, model_answers in answers.items():
                # Loop through personas
                for persona_id, persona_answers in model_answers.items():
                    # Loop through seeds (runs)
                    for run_seed, answer in persona_answers.items():
                        # Append the row to data
                        data.append(
                            {
                                "Instruction ID": instruction_id,
                                "Instruction Question": question,
                                "Model ID": model_id,
                                "Persona ID": persona_id,
                                "Run Seed": run_seed,
                                "Answer": answer,
                            }
                        )

        # Create DataFrame from the collected data (keeping the columns even when empty)
        import pandas as pd

        return pd.DataFrame(data, columns=ANSWER_COLUMNS)
