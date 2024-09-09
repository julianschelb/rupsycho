# ===========================================================================
#                        ExperimentExportMixin Implementation
# ===========================================================================
# This mixin provides methods to export the experiment results to a file or
# return the answers in various formats. It supports exporting to JSON files
# and converting the experiment data into a pandas DataFrame.


from typing import Dict, Any, List, Optional, Set
import pandas as pd
import json


class ExperimentExportMixin:
    """
    Mixin providing methods to export the experiment results to a file or return the answers.
    """

    def model_dump(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        exclude_unset: bool = True,
        exclude_none: bool = True
    ) -> Dict[str, Any]:
        """
        Convert the ExperimentDocument instance to a dictionary, allowing for inclusion or exclusion of specific fields.

        :param include: A set of field names to include in the output.
        :param exclude: A set of field names to exclude from the output.
        :param exclude_unset: Exclude fields that were not explicitly set.
        :param exclude_none: Exclude fields that are set to None.
        """

        # Initialize the dictionary with explicitly included fields, handling missing attributes gracefully
        experiment_data = {}

        # Safely include "name"
        if hasattr(self, 'name') and (not include or 'name' in include) and (not exclude or 'name' not in exclude):
            experiment_data["name"] = self.name

        # Safely include "description"
        if hasattr(self, 'description') and (not include or 'description' in include) and (not exclude or 'description' not in exclude):
            experiment_data["description"] = self.description

        # Safely include "parameters"
        if hasattr(self, 'parameters') and self.parameters is not None and \
                (not include or 'parameters' in include) and (not exclude or 'parameters' not in exclude):
            experiment_data["parameters"] = (
                self.parameters.dict(
                    exclude_unset=exclude_unset, exclude_none=exclude_none)
                if hasattr(self.parameters, "dict") else self.parameters
            )

        # Safely include "prompt_template"
        if hasattr(self, 'prompt_template') and self.prompt_template is not None and \
                (not include or 'prompt_template' in include) and (not exclude or 'prompt_template' not in exclude):
            experiment_data["prompt_template"] = (
                self.prompt_template.dict(
                    exclude_unset=exclude_unset, exclude_none=exclude_none)
                if hasattr(self.prompt_template, "dict") else self.prompt_template
            )

        # Safely include "models"
        if hasattr(self, 'models') and self.models and \
                (not include or 'models' in include) and (not exclude or 'models' not in exclude):
            experiment_data["models"] = {
                key: model.dict(exclude_unset=exclude_unset,
                                exclude_none=exclude_none)
                if hasattr(model, "dict") else model
                for key, model in self.models.items()
            }

        # Safely include "demographic_profiles"
        if hasattr(self, 'demographic_profiles') and self.demographic_profiles and \
                (not include or 'demographic_profiles' in include) and (not exclude or 'demographic_profiles' not in exclude):
            experiment_data["demographic_profiles"] = {
                key: profile.dict(exclude_unset=exclude_unset,
                                  exclude_none=exclude_none)
                if hasattr(profile, "dict") else profile
                for key, profile in self.demographic_profiles.items()
            }

        # Safely include "questionnaire"
        if hasattr(self, 'questionnaire') and self.questionnaire is not None and \
                (not include or 'questionnaire' in include) and (not exclude or 'questionnaire' not in exclude):
            experiment_data["questionnaire"] = (
                self.questionnaire.dict(
                    exclude_unset=exclude_unset, exclude_none=exclude_none)
                if hasattr(self.questionnaire, "dict") else self.questionnaire
            )

        # Optionally exclude None values
        if exclude_none:
            experiment_data = {k: v for k,
                               v in experiment_data.items() if v is not None}

        return experiment_data

    def export_to_file(self, filename: str) -> None:
        """Exports the experiment data to a file in JSON format."""
        try:
            data = self.model_dump()
            with open(filename, "w") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving to file: {e}")

    def get_answers(self) -> List[Dict[str, Any]]:
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
                        data.append({
                            "Instruction ID": instruction_id,
                            "Instruction Question": question,
                            "Model ID": model_id,
                            "Persona ID": persona_id,
                            "Run Seed": run_seed,
                            "Answer": answer
                        })

        # Create DataFrame from the collected data
        return pd.DataFrame(data)
