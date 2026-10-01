# ===========================================================================
#                           Prompt Template Mixin
# ===========================================================================
# This module defines a mixin class for managing the prompt template within
# an experiment. It provides methods to set and load prompt templates, ensuring
# that they are correctly handled and converted into runnable forms.


from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.load import dumpd, load


class PromptTemplateMixin:
    """
    Mixin providing methods to manage the prompt template in the experiment.

    This includes setting the prompt template, loading it, and converting it
    into a runnable form.
    """

    if TYPE_CHECKING:
        # Provided by ExperimentDocument, which mixes this class in.
        prompt_template: Any
        runnable_prompt: Any
        runnable_parser: Any

    def load_prompt(self, prompt_template: str) -> Any | None:
        """
        Load the prompt template from its serialized definition.

        :param prompt_template: Serialized prompt template.
        :return: Loaded prompt, or None if an error occurs.
        """
        try:
            prompt = load(prompt_template)
            return prompt
        except Exception as e:
            warnings.warn(f"Failed to load prompt template: {e}", UserWarning, stacklevel=2)
            return None

    def set_prompt(self, prompt: Any) -> None:
        """
        Adds a prompt to the experiment and converts it into its runnable form.

        :param prompt: The prompt to be set.
        """
        self.prompt_template = dumpd(prompt)
        self.runnable_prompt = prompt

    def get_prompt(self) -> Any | None:
        """
        Retrieve the current runnable prompt template if available.

        :return: The current runnable prompt, or None if not set.
        """
        return getattr(self, "runnable_prompt", None)

    def get_prompt_config(self) -> str | None:
        """
        Retrieve the serialized prompt template configuration.

        :return: Serialized prompt template or None if not set.
        """
        return getattr(self, "prompt_template", None)

    def set_prompt_config(self, prompt_template: str) -> None:
        """
        Set the prompt template configuration by loading its serialized form.

        :param prompt_template: Serialized prompt template configuration.
        """
        self.prompt_template = prompt_template
        self.runnable_prompt = self.load_prompt(prompt_template)

    def reset_prompt(self) -> None:
        """
        Resets the current prompt template and runnable form to None.
        """
        self.prompt_template = None
        self.runnable_prompt = None

    def has_prompt(self) -> bool:
        """
        Check whether a runnable prompt is set.

        :return: True if a runnable prompt is set, False otherwise.
        """
        return self.runnable_prompt is not None
