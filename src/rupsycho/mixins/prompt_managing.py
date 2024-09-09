# ===========================================================================
#                           Prompt Template Mixin
# ===========================================================================
# This module defines a mixin class for managing the prompt template within
# an experiment. It provides methods to set and load prompt templates, ensuring
# that they are correctly handled and converted into runnable forms.


import warnings
from langchain_core.load import dumpd, load
from typing import Any, Optional


class PromptTemplateMixin:
    """
    Mixin providing methods to manage the prompt template in the experiment.

    This includes setting the prompt template, loading it, and converting it
    into a runnable form.
    """

    def load_prompt(self, prompt_template: str) -> Optional[Any]:
        """
        Load the prompt template from its serialized definition.

        :param prompt_template: Serialized prompt template.
        :return: Loaded prompt, or None if an error occurs.
        """
        try:
            prompt = load(prompt_template)
            return prompt
        except Exception as e:
            warnings.warn(f"Failed to load prompt template: {e}", UserWarning)
            return None

    def set_prompt(self, prompt: Any) -> None:
        """
        Adds a prompt to the experiment and converts it into its runnable form.

        :param prompt: The prompt to be set.
        """
        self.prompt_template = dumpd(prompt)
        self.runnable_prompt = prompt

    def get_prompt(self) -> Optional[Any]:
        """
        Retrieve the current runnable prompt template if available.

        :return: The current runnable prompt, or None if not set.
        """
        return getattr(self, "runnable_prompt", None)

    def get_prompt_config(self) -> Optional[str]:
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