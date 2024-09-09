# ===========================================================================
#                           Model Management Mixin
# ===========================================================================
# This module defines a mixin class for managing models within an experiment.
# The mixin provides methods to add, remove, replace, and manage models,
# as well as convert model definitions into runnable models.


from langchain_core.load import dumpd, load
from rupsycho.models.model import LangChainModelConfig
from typing import Any, Dict, List, Optional
import warnings


class ModelManagementMixin:
    """
    Mixin providing methods to manage models in the experiment.

    This includes adding models, setting models, and handling the conversion
    of model definitions into runnable models.
    """

    def load_model(self, model_definition):
        """
        Load the model from its serialized definition.

        :param model_definition: Serialized model definition
        :return: Loaded model, or None if an error occurs.
        """
        try:
            model = load(model_definition)
            return model
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}", UserWarning)
            return None

    def add_model(self, model: Any, identifier: Optional[str] = None) -> None:
        """
        Adds a model to the experiment.

        :param model: The model to be added.
        :param identifier: Optional identifier for the model.
        """
        key = identifier if identifier else str(id(model))

        if key in self.models:
            warnings.warn(
                f"A model with the identifier '{key}' already exists.", UserWarning)
        else:
            self.models[key] = LangChainModelConfig(definition=dumpd(model))

        # Ensure the model is added to runnable_models
        # runnable_model = self.load_model(self.models[key].definition)
        # if runnable_model:
        self.runnable_models[key] = model

    def set_runnable_models(self) -> None:
        """
        Converts model definitions into runnable models, updating the runnable_models dictionary.
        """
        self.runnable_models = {}
        for key, model in self.models.items():
            runnable_model = self.load_model(model.definition)
            if runnable_model:
                self.runnable_models[key] = runnable_model

    def get_model(self, identifier: str) -> Optional[Any]:
        """
        Retrieves a model by its identifier.

        :param identifier: The identifier of the model.
        :return: The model if found, otherwise None.
        """
        return self.runnable_models.get(identifier, None)

    def remove_model(self, identifier: str) -> None:
        """
        Removes a model from the experiment by its identifier.

        :param identifier: The identifier of the model to be removed.
        """
        if identifier in self.models:
            del self.models[identifier]
            if identifier in self.runnable_models:
                del self.runnable_models[identifier]
        else:
            warnings.warn(
                f"No model found with the identifier '{identifier}'.", UserWarning)

    def list_models(self) -> List[str]:
        """
        Lists all model identifiers in the experiment.

        :return: A list of model identifiers.
        """
        return list(self.models.keys())

    def has_model(self, identifier: str) -> bool:
        """
        Checks if a model exists in the experiment by its identifier.

        :param identifier: The identifier of the model to check.
        :return: True if the model exists, False otherwise.
        """
        return identifier in self.models

    def replace_model(self, identifier: str, new_model: Any) -> None:
        """
        Replaces an existing model in the experiment with a new one.

        :param identifier: The identifier of the model to replace.
        :param new_model: The new model to replace the old one.
        """
        if identifier in self.models:
            self.models[identifier] = LangChainModelConfig(
                definition=dumpd(new_model))
            self.set_runnable_models()  # Refresh the runnable models dictionary
        else:
            warnings.warn(
                f"No model found with the identifier '{identifier}'.", UserWarning)

    # def get_model_config(self, identifier: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Retrieves the configuration dictionary of a model by its identifier.

    #     :param identifier: The identifier of the model.
    #     :return: The configuration dictionary of the model, or None if not found.
    #     """
    #     model = self.get_model(identifier)
    #     return model.definition if model else None

    def clear_models(self) -> None:
        """
        Removes all models from the experiment.
        """
        self.models.clear()
        self.runnable_models.clear()

    def count_models(self) -> int:
        """
        Returns the number of models in the experiment.

        :return: The number of models.
        """
        return len(self.models)

    def get_all_runnable_models(self) -> Dict[str, Any]:
        """
        Retrieves a dictionary of all runnable models.

        :return: A dictionary of runnable models with identifiers as keys.
        """
        return self.runnable_models
