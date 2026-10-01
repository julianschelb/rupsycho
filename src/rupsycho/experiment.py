# ===========================================================================
#                        ExperimentDocument Class Implementation
# ===========================================================================
#  This module defines the ExperimentDocument class for managing experimental
#  data, questionnaires, and metadata. It integrates multiple mixins and
#  extends BaseMedia and Pydantic's BaseModel.

import warnings
from typing import Any, Literal

from langchain_core.documents.base import BaseMedia
from pydantic import BaseModel, ConfigDict, Field

from .mixins.experiment_exporting import ExperimentExportMixin
from .mixins.experiment_processing import ExperimentProcessingMixin
from .mixins.model_managing import ModelManagementMixin
from .mixins.persona_managing import PersonaManagementMixin
from .mixins.prompt_managing import PromptTemplateMixin
from .models.model import (
    DEFAULT_MODEL_CONFIG,
    DeepSeekModelConfig,
    GoogleModelConfig,
    LangChainModelConfig,
    LocalHuggingFaceModelConfig,
    OllamaModelConfig,
    OpenAIModelConfig,
    RemoteHuggingFaceModelConfig,
)
from .models.parameters import ExperimentParameters
from .models.prompt import (
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
    NormalPromptTemplateConfig,
)
from .models.questionnaire import DemographicProfile, Questionnaire
from .prompts import DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG

# ================================= Prompt Type Mapping ================================


PROMPT_CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    "normal": NormalPromptTemplateConfig,
    "chat": ChatPromptTemplateConfig,
    "langchain": LangchainPromptTemplateConfig,
}

# ================================= Model Type Mapping ================================


MODEL_CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    "local_huggingface": LocalHuggingFaceModelConfig,
    "remote_huggingface": RemoteHuggingFaceModelConfig,
    "ollama": OllamaModelConfig,
    "openai": OpenAIModelConfig,
    "langchain": LangChainModelConfig,
    "google": GoogleModelConfig,
    "deepseek": DeepSeekModelConfig,
}

# ================================= Experiment Class ================================


class ExperimentDocument(  # type: ignore[misc]
    BaseMedia,
    ExperimentProcessingMixin,
    ExperimentExportMixin,
    ModelManagementMixin,
    PromptTemplateMixin,
    PersonaManagementMixin,
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

    name: str | None = Field(
        None,
        description="The name of the experiment",
        examples=["Generative Models for Big Five Inventory"],
    )
    description: str | None = Field(
        None,
        description="The description of the experiment",
        examples=["Testing Generative Models for BFI questionnaire using Rupsycho."],
    )
    parameters: ExperimentParameters = Field(
        default_factory=ExperimentParameters,
        description="The parameters for the experiment and text generation",
    )
    prompt_template: (
        NormalPromptTemplateConfig | ChatPromptTemplateConfig | LangchainPromptTemplateConfig
    ) = Field(None, description="The prompt template used by the model")  # type: ignore[assignment]
    models: dict[
        str,
        LangChainModelConfig
        | LocalHuggingFaceModelConfig
        | RemoteHuggingFaceModelConfig
        | OllamaModelConfig
        | OpenAIModelConfig
        | GoogleModelConfig
        | DeepSeekModelConfig,
    ] = Field(default_factory=dict, description="The models in the experiment")
    demographic_profiles: dict[str, DemographicProfile] = Field(
        default_factory=dict, description="The demographic profiles in the experiment"
    )
    questionnaire: Questionnaire | None = Field(
        None, description="The questionnaire in the experiment"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the experiment document."
    )
    type: Literal["ExperimentDocument"] = "ExperimentDocument"
    runnable_prompt: Any = Field(
        default_factory=dict, description="The initialized prompts in the experiment"
    )
    runnable_models: dict[str, Any] = Field(
        default_factory=dict, description="The initialized models in the experiment"
    )
    runnable_parser: Any = Field(
        default_factory=dict, description="The initialized output parsers in the experiment"
    )

    # --------------------------------- Config --------------------------------

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --------------------------------- Initialization --------------------------------

    def __init__(self, **data: Any):
        """Initialize ExperimentDocument with optional conversions for nested data."""

        # Handle nested data conversions for parameters
        if "parameters" in data:
            data["parameters"] = self._convert_parameters(data["parameters"])

        # Convert demographic profiles to instances
        if "demographic_profiles" in data:
            data["demographic_profiles"] = self._convert_demographic_profiles(
                data["demographic_profiles"]
            )

        # Convert prompt template to runnable form and load it
        if "prompt_template" in data:
            data["prompt_template"] = self._convert_prompt(data["prompt_template"])
        else:
            data["prompt_template"] = DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG

        data["runnable_prompt"] = self._load_runnable_prompt(data["prompt_template"])

        # Convert models to runnable form but load them conditionally based on lazy_load_models
        if "models" in data:
            data["models"] = self._convert_models(data["models"])
        else:
            data["models"] = {"default_model": DEFAULT_MODEL_CONFIG}

        # Load models if lazy_load_models is False; otherwise, leave them for later loading
        if not data.get("parameters").lazy_load_models:  # type: ignore[union-attr]
            data["runnable_models"] = self._load_runnable_models(data["models"])
        else:
            # Models will be loaded later
            data["runnable_models"] = data["models"]

        # Convert questionnaire to an instance of Questionnaire
        if "questionnaire" in data:
            data["questionnaire"] = self._convert_questionnaire(data["questionnaire"])

        super().__init__(**data)

    # --------------------------------- Conversion Methods --------------------------------

    @staticmethod
    def _convert_parameters(parameters: ExperimentParameters | dict) -> ExperimentParameters:
        """Convert parameters to an instance of Parameters."""
        if isinstance(parameters, dict):
            return ExperimentParameters(**parameters)
        return parameters

    @staticmethod
    def _convert_demographic_profiles(
        profiles: dict[str, DemographicProfile | dict],
    ) -> dict[str, DemographicProfile]:
        """Convert demographic profiles to instances of DemographicProfile."""
        return {
            key: DemographicProfile(**profile) if isinstance(profile, dict) else profile
            for key, profile in profiles.items()
        }

    @staticmethod
    def _convert_prompt(prompt: dict | BaseModel) -> BaseModel | dict:
        """Convert a single prompt configuration to an instance of its respective Pydantic PromptTemplateConfig class."""
        if isinstance(prompt, dict):
            prompt_type = prompt.get("type")
            if prompt_type and prompt_type in PROMPT_CONFIG_CLASSES:
                return PROMPT_CONFIG_CLASSES[prompt_type](**prompt)
            else:
                raise ValueError("Unknown or missing prompt type.")
        return prompt

    @staticmethod
    def _convert_models(models: dict[str, dict | BaseModel]) -> dict[str, BaseModel | dict]:
        """Convert model configurations to instances of their respective Pydantic ModelConfig classes."""

        def convert_model(key: str, model: dict | BaseModel) -> BaseModel | dict:

            if isinstance(model, dict):
                model_type = model.get("type")
                if model_type and model_type in MODEL_CONFIG_CLASSES:
                    return MODEL_CONFIG_CLASSES[model_type](**model)
                else:
                    raise ValueError(f"Unknown or missing model type for key: {key}")
            return model

        return {key: convert_model(key, model) for key, model in models.items()}

    @staticmethod
    def _convert_questionnaire(questionnaire: Questionnaire | dict) -> Questionnaire:
        """Convert questionnaire to an instance of Questionnaire."""
        if isinstance(questionnaire, dict):
            return Questionnaire(**questionnaire)
        return questionnaire

    def _load_runnable_models(self, models: dict[str, Any]) -> dict[str, Any]:
        """Load models into runnable instances."""
        runnable_models = {}
        for key, model in models.items():
            try:
                runnable_models[key] = model.load_model()
            except Exception as e:
                print(f"Failed to load model: {e}")

        return runnable_models

    def _load_runnable_prompt(self, prompt_template: str | BaseModel) -> Any:
        """Load a prompt template into a runnable LangChain prompt."""
        if isinstance(prompt_template, str):
            return prompt_template  # Directly use if it's a string

        runnable_prompt = None
        try:
            runnable_prompt = prompt_template.load_prompt_template()  # type: ignore[attr-defined]
        except Exception as e:
            warnings.warn(f"Failed to load prompt template: {e}", UserWarning, stacklevel=2)

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
