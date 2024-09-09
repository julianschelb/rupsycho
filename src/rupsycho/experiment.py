# ===========================================================================
#                        ExperimentDocument Class Implementation
# ===========================================================================
#  This module defines the ExperimentDocument class for managing experimental
#  data, questionnaires, and metadata. It integrates multiple mixins and
#  extends BaseMedia and Pydantic's BaseModel.

from typing import Any, Literal, Optional, Dict, Type, Union
from langchain_core.documents.base import BaseMedia
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel

from .models.model import (
    DEFAULT_MODEL_CONFIG,
    LangChainModelConfig,
    LocalHuggingFaceModelConfig,
    OllamaModelConfig,
    OpenAIModelConfig,
    RemoteHuggingFaceModelConfig
)

from .models.prompt import (
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
    NormalPromptTemplateConfig
)

from .prompts import DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG
from .models.parameters import Parameters
from .models.questionnaire import DemographicProfile, Questionnaire
from .mixins.experiment_processing import ExperimentProcessingMixin
from .mixins.experiment_exporting import ExperimentExportMixin
from .mixins.model_managing import ModelManagementMixin
from .mixins.prompt_managing import PromptTemplateMixin
from .mixins.persona_managing import PersonaManagementMixin

import warnings


# ================================= Prompt Type Mapping ================================


PROMPT_CONFIG_CLASSES: Dict[str, Type[BaseModel]] = {
    "normal": NormalPromptTemplateConfig,
    "chat": ChatPromptTemplateConfig,
    "langchain": LangchainPromptTemplateConfig
}

# ================================= Model Type Mapping ================================


MODEL_CONFIG_CLASSES: Dict[str, Type[BaseModel]] = {
    "local_huggingface": LocalHuggingFaceModelConfig,
    "remote_huggingface": RemoteHuggingFaceModelConfig,
    "ollama": OllamaModelConfig,
    "openai": OpenAIModelConfig,
    "langchain": LangChainModelConfig
}

# ================================= Experiment Class ================================


class ExperimentDocument(
    BaseMedia, ExperimentProcessingMixin, ExperimentExportMixin,
    ModelManagementMixin, PromptTemplateMixin, PersonaManagementMixin
):
    """Class for storing an experiment and associated questionnaires along with metadata.

    Extends LangChain's BaseMedia and Pydantic's BaseModel, integrating data validation 
    and serialization capabilities with document handling features.

    Example:

        .. code-block:: python

            from my_module import ExperimentDocument

            experiment_doc = ExperimentDocument(
                name="Experiment 1",
                description="Test Experiment",
                demographic_profiles={...},
                models={...},
                questionnaire=...,
                metadata={"source": "Lab A"}
            )
    """

    # --------------------------------- Attributes --------------------------------

    name: Optional[str] = Field(
        None,
        description="The name of the experiment",
        example="Generative Models for Big Five Inventory",
    )
    description: Optional[str] = Field(
        None,
        description="The description of the experiment",
        example="Testing Generative Models for BFI questionnaire using Rupsycho.",
    )
    parameters: Parameters = Field(
        default_factory=Parameters,
        description="The parameters for the experiment and text generation"
    )
    prompt_template: Union[NormalPromptTemplateConfig, ChatPromptTemplateConfig, LangchainPromptTemplateConfig] = Field(
        None, description="The prompt template used by the model"
    )
    models: Dict[str, Union[
        LangChainModelConfig, LocalHuggingFaceModelConfig, RemoteHuggingFaceModelConfig,
        OllamaModelConfig, OpenAIModelConfig
    ]] = Field(
        default_factory=dict, description="The models in the experiment"
    )
    demographic_profiles: Dict[str, DemographicProfile] = Field(
        default_factory=dict, description="The demographic profiles in the experiment"
    )
    questionnaire: Optional[Questionnaire] = Field(
        None, description="The questionnaire in the experiment"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the experiment document."
    )
    type: Literal["ExperimentDocument"] = "ExperimentDocument"
    runnable_prompt: Any = Field(
        default_factory=dict, description="The initialized prompts in the experiment"
    )
    runnable_models: Dict[str, Any] = Field(
        default_factory=dict, description="The initialized models in the experiment"
    )
    runnable_parser: Any = Field(
        default_factory=dict, description="The initialized output parsers in the experiment"
    )

    # --------------------------------- Config --------------------------------

    class Config:
        arbitrary_types_allowed = True

    # --------------------------------- Initialization --------------------------------

    def __init__(self, **data: Any):
        """Initialize ExperimentDocument with optional conversions for nested data."""

        # Handle nested data conversions for parameters
        if "parameters" in data:
            data["parameters"] = self._convert_parameters(data["parameters"])

        # Convert demographic profiles to instances
        if "demographic_profiles" in data:
            data["demographic_profiles"] = self._convert_demographic_profiles(
                data["demographic_profiles"])

        # Convert prompt template to runnable form and load it
        if "prompt_template" in data:
            data["prompt_template"] = self._convert_prompt(
                data["prompt_template"])
        else:
            data["prompt_template"] = DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG

        data["runnable_prompt"] = self._load_runnable_prompt(
            data["prompt_template"])

        # Convert models to runnable form and load them
        if "models" in data:
            data["models"] = self._convert_models(data["models"])
        else:
            data["models"] = {"default_model": DEFAULT_MODEL_CONFIG}

        data["runnable_models"] = self._load_runnable_models(
            data["models"])

        # Convert questionnaire to an instance of Questionnaire
        if "questionnaire" in data:
            data["questionnaire"] = self._convert_questionnaire(
                data["questionnaire"])

        super().__init__(**data)

    # --------------------------------- Conversion Methods --------------------------------

    @staticmethod
    def _convert_parameters(parameters: Union[Parameters, dict]) -> Parameters:
        """Convert parameters to an instance of Parameters."""
        if isinstance(parameters, dict):
            return Parameters(**parameters)
        return parameters

    @staticmethod
    def _convert_demographic_profiles(
        profiles: Dict[str, Union[DemographicProfile, dict]]
    ) -> Dict[str, DemographicProfile]:
        """Convert demographic profiles to instances of DemographicProfile."""
        return {
            key: DemographicProfile(
                **profile) if isinstance(profile, dict) else profile
            for key, profile in profiles.items()
        }

    @staticmethod
    def _convert_prompt(prompt: Union[dict, BaseModel]) -> Union[BaseModel, dict]:
        """Convert a single prompt configuration to an instance of its respective Pydantic PromptTemplateConfig class."""
        if isinstance(prompt, dict):
            prompt_type = prompt.get("type")
            if prompt_type and prompt_type in PROMPT_CONFIG_CLASSES:
                return PROMPT_CONFIG_CLASSES[prompt_type](**prompt)
            else:
                raise ValueError(f"Unknown or missing prompt type.")
        return prompt

    @staticmethod
    def _convert_models(
        models: Dict[str, Union[dict, BaseModel]]
    ) -> Dict[str, Union[BaseModel, dict]]:
        """Convert model configurations to instances of their respective Pydantic ModelConfig classes."""

        def convert_model(key: str, model: Union[dict, BaseModel]) -> Union[BaseModel, dict]:

            if isinstance(model, dict):
                model_type = model.get("type")
                if model_type and model_type in MODEL_CONFIG_CLASSES:
                    return MODEL_CONFIG_CLASSES[model_type](**model)
                else:
                    raise ValueError(
                        f"Unknown or missing model type for key: {key}")
            return model

        return {key: convert_model(key, model) for key, model in models.items()}

    @staticmethod
    def _convert_questionnaire(questionnaire: Union[Questionnaire, dict]) -> Questionnaire:
        """Convert questionnaire to an instance of Questionnaire."""
        if isinstance(questionnaire, dict):
            return Questionnaire(**questionnaire)
        return questionnaire

    def _load_runnable_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Load models into runnable instances."""
        runnable_models = {}
        for key, model in models.items():

            try:
                runnable_models[key] = model.load_model()
            except Exception as e:
                print(f"Failed to load model: {e}")

        return runnable_models

    def _load_runnable_prompt(self, prompt_template: Union[str, BaseModel]) -> Any:
        """Load a prompt template into a runnable LangChain prompt."""
        if isinstance(prompt_template, str):
            return prompt_template  # Directly use if it's a string

        runnable_prompt = None
        try:
            runnable_prompt = prompt_template.load_prompt_template()
        except Exception as e:
            warnings.warn(
                f"Failed to load prompt template: {e}", UserWarning)

        return runnable_prompt

    # --------------------------------- String Representation --------------------------------

    def __str__(self) -> str:
        """Override __str__ to restrict it to experiment details and metadata."""
        return (
            f"name={self.name}, "
            f"description={self.description}, "
            # f"parameters={self.parameters}, "
            # f"prompt_template={self.prompt_template}, "
            # f"demographic_profiles={self.demographic_profiles}, "
            # f"models={self.models}, "
            # f"questionnaire={self.questionnaire}, "
            f"metadata={self.metadata}"
        )

    # --------------------------------- Getters and Setters --------------------------------

    def set_questionnaire(self, questionnaire: Questionnaire) -> None:
        """Sets the questionnaire for the experiment."""
        self.questionnaire = questionnaire

    def set_parser(self, parser: Any) -> None:
        """Adds a parser to the experiment."""
        self.runnable_parser = parser
