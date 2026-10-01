# ===========================================================================
#                           Prompt Template Mixin
# ===========================================================================
# This module defines a mixin class for managing the prompt template within
# an experiment. It provides methods to set and load prompt templates, ensuring
# that they are correctly handled and converted into runnable forms.


from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.load import dumpd
from pydantic import BaseModel

from rupsycho._compat import load_serialized
from rupsycho.models.prompt import LangchainPromptTemplateConfig


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

        @staticmethod
        def _convert_prompt(prompt: dict[str, Any] | BaseModel) -> Any: ...

    def load_prompt(self, prompt_template: dict[str, Any]) -> Any | None:
        """
        Load the prompt template from its serialized definition.

        :param prompt_template: Serialized prompt template.
        :return: Loaded prompt, or None if an error occurs.
        """
        try:
            prompt = load_serialized(prompt_template)
            return prompt
        except Exception as e:
            warnings.warn(f"Failed to load prompt template: {e}", UserWarning, stacklevel=2)
            return None

    def set_prompt(self, prompt: Any) -> None:
        """Use a ready-made LangChain prompt for this experiment.

        The prompt is kept as a serialized ``langchain`` prompt configuration, so it survives
        ``export_to_file`` and can be loaded again.

        Args:
            prompt: A LangChain prompt template such as ``ChatPromptTemplate``. Its input
                variables may be ``general_instruction``, ``persona_description``,
                ``question`` and ``answer_options``.

        Example:
            ```python
            from langchain_core.prompts import ChatPromptTemplate

            experiment.set_prompt(ChatPromptTemplate.from_messages([
                ("system", "Answer as {persona_description}."),
                ("user", "{question}\\n{answer_options}"),
            ]))
            ```
        """
        self.prompt_template = LangchainPromptTemplateConfig(definition=dumpd(prompt))
        self.runnable_prompt = prompt

    def get_prompt(self) -> Any | None:
        """Return the current runnable prompt template.

        Returns:
            The LangChain prompt, or ``None`` if none is set.
        """
        return getattr(self, "runnable_prompt", None)

    def get_prompt_config(self) -> Any | None:
        """Return the prompt template configuration (``normal``, ``chat`` or ``langchain``).

        Returns:
            The configuration object, or ``None`` if none is set.
        """
        return getattr(self, "prompt_template", None)

    def set_prompt_config(self, prompt_template: dict[str, Any] | BaseModel) -> None:
        """Set the prompt from a configuration, as it would appear in a JSON file.

        Args:
            prompt_template: A configuration dictionary (``{"type": "chat", "messages": [...]}``,
                ``{"type": "normal", "template": "..."}``) or LangChain's own serialization of a
                prompt (as produced by ``langchain_core.load.dumpd``), or a config object.

        Raises:
            ValueError: If the configuration is not recognised.
        """
        config = self._convert_prompt(prompt_template)
        self.prompt_template = config
        self.runnable_prompt = config.load_prompt_template()

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
