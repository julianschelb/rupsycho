# model_managing.py
"""Add, inspect, replace and remove the models of an experiment."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.load import dumpd

from rupsycho._compat import load_serialized
from rupsycho.models.model import LangChainModelConfig

__all__ = ["ModelManagementMixin"]


class ModelManagementMixin:
    """Methods to manage the models of an experiment.

    An experiment keeps two dictionaries under the same identifiers: ``models`` holds the
    *configurations* (what gets exported), ``runnable_models`` holds what is run - either the
    configuration itself (loaded lazily when the run reaches it) or a ready LangChain model
    added with ``add_model``.
    """

    if TYPE_CHECKING:
        # Provided by ExperimentDocument, which mixes this class in.
        models: dict[str, Any]
        runnable_models: dict[str, Any]

    def load_model(self, model_definition: dict[str, Any]) -> Any | None:
        """Deserialize a model from its LangChain definition.

        Args:
            model_definition: Serialized model as produced by ``langchain_core.load.dumpd``.

        Returns:
            The model, or ``None`` (with a warning) if it cannot be deserialized.
        """
        try:
            return load_serialized(model_definition)
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}", UserWarning, stacklevel=2)
            return None

    def add_model(self, model: Any, identifier: str | None = None) -> None:
        """Add a ready LangChain model to the experiment.

        The model is run as it is. Its serialized definition is stored in ``models`` so that the
        experiment can be exported; models that cannot be serialized (for example local
        pipelines) are exported as a placeholder and have to be added again after loading.

        Args:
            model: Any LangChain runnable (chat model, LLM, ...).
            identifier: Name of the model in the results. Defaults to its object id.

        Example:
            ```python
            experiment.add_model(ChatOpenAI(model="gpt-4o-mini"), identifier="gpt-4o-mini")
            ```

        Note:
            Adding a model under an identifier that exists replaces it and emits a warning.
        """
        key = identifier if identifier else str(id(model))

        if key in self.models:
            warnings.warn(
                f"A model with the identifier '{key}' already exists and is replaced.",
                UserWarning,
                stacklevel=2,
            )
        self.models[key] = LangChainModelConfig(definition=dumpd(model))
        self.runnable_models[key] = model

    def set_runnable_models(self) -> None:
        """Reset ``runnable_models`` to the configured ``models``.

        Every model is then loaded from its configuration when the run reaches it. Models that
        were added as live objects without a loadable configuration are no longer available
        afterwards.
        """
        self.runnable_models = dict(self.models)

    def get_model(self, identifier: str) -> Any | None:
        """Return the runnable entry of a model.

        Args:
            identifier: Identifier of the model.

        Returns:
            The LangChain model, or - for models that are loaded lazily - its configuration;
            ``None`` if there is no such model.
        """
        return self.runnable_models.get(identifier, None)

    def remove_model(self, identifier: str) -> None:
        """Remove a model from the experiment.

        Args:
            identifier: Identifier of the model. Unknown identifiers only emit a warning.
        """
        if identifier in self.models:
            del self.models[identifier]
            self.runnable_models.pop(identifier, None)
        else:
            warnings.warn(
                f"No model found with the identifier '{identifier}'.", UserWarning, stacklevel=2
            )

    def list_models(self) -> list[str]:
        """Return the identifiers of all models of the experiment."""
        return list(self.models.keys())

    def has_model(self, identifier: str) -> bool:
        """Return whether a model with this identifier exists."""
        return identifier in self.models

    def replace_model(self, identifier: str, new_model: Any) -> None:
        """Replace an existing model by a ready LangChain model.

        Args:
            identifier: Identifier of the model to replace. Unknown identifiers only emit a
                warning.
            new_model: The new model.
        """
        if identifier in self.models:
            self.models[identifier] = LangChainModelConfig(definition=dumpd(new_model))
            self.runnable_models[identifier] = new_model
        else:
            warnings.warn(
                f"No model found with the identifier '{identifier}'.", UserWarning, stacklevel=2
            )

    def clear_models(self) -> None:
        """Remove all models from the experiment."""
        self.models.clear()
        self.runnable_models.clear()

    def count_models(self) -> int:
        """Return the number of models in the experiment."""
        return len(self.models)

    def get_all_runnable_models(self) -> dict[str, Any]:
        """Return the runnable entries of all models, keyed by identifier."""
        return self.runnable_models
