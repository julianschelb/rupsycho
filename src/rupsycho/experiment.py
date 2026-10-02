# ===========================================================================
#                        ExperimentDocument Class Implementation
# ===========================================================================
#  This module defines the ExperimentDocument class for managing experimental
#  data, questionnaires, and metadata. It integrates multiple mixins and
#  extends BaseMedia and Pydantic's BaseModel.

from collections.abc import Mapping
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


class ExperimentDocument(
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

        # Handle nested data conversions for parameters (optional section)
        data["parameters"] = self._convert_parameters(data.get("parameters") or {})

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

        # Models are loaded when the run reaches them (lazy_load_models) or right after validation
        eager = not data["parameters"].lazy_load_models
        data["runnable_models"] = data["models"]

        # Convert questionnaire to an instance of Questionnaire
        if "questionnaire" in data:
            data["questionnaire"] = self._convert_questionnaire(data["questionnaire"])

        super().__init__(**data)

        if eager:  # only after everything else validated: loading may download large models
            self.runnable_models = self._load_runnable_models(self.models)

    # --------------------------------- Conversion Methods --------------------------------

    @staticmethod
    def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
        """Return ``value`` if it is a dictionary, else raise a ``ValueError`` naming the section."""
        if not isinstance(value, Mapping):
            raise ValueError(
                f"'{section}' must be an object (a dictionary), got {type(value).__name__}"
            )
        return value

    @classmethod
    def _convert_parameters(cls, parameters: ExperimentParameters | dict) -> ExperimentParameters:
        """Convert parameters to an instance of ExperimentParameters."""
        if isinstance(parameters, ExperimentParameters):
            return parameters
        return ExperimentParameters(**cls._require_mapping(parameters, "parameters"))

    @classmethod
    def _convert_demographic_profiles(
        cls, profiles: dict[str, DemographicProfile | dict]
    ) -> dict[str, DemographicProfile]:
        """Convert demographic profiles to instances of DemographicProfile."""
        return {
            key: DemographicProfile(**cls._require_mapping(profile, f"demographic_profiles.{key}"))
            if not isinstance(profile, DemographicProfile)
            else profile
            for key, profile in cls._require_mapping(profiles, "demographic_profiles").items()
        }

    @classmethod
    def _convert_prompt(cls, prompt: dict | BaseModel) -> BaseModel:
        """Convert a prompt configuration to its Pydantic PromptTemplateConfig class."""
        if isinstance(prompt, BaseModel):
            return prompt
        prompt = dict(cls._require_mapping(prompt, "prompt_template"))
        prompt_type = prompt.get("type")
        if prompt_type and prompt_type in PROMPT_CONFIG_CLASSES:
            return PROMPT_CONFIG_CLASSES[prompt_type](**prompt)
        if prompt.get("lc") == 1:
            # LangChain's own serialisation, as stored by set_prompt() and written by exports
            return LangchainPromptTemplateConfig(definition=prompt)
        raise ValueError("Unknown or missing prompt type.")

    @classmethod
    def _convert_models(cls, models: dict[str, dict | BaseModel]) -> dict[str, BaseModel]:
        """Convert model configurations to instances of their respective Pydantic classes."""

        def convert_model(key: str, model: dict | BaseModel) -> BaseModel:
            if isinstance(model, BaseModel):
                return model
            model = dict(cls._require_mapping(model, f"models.{key}"))
            model_type = model.get("type")
            if model_type and model_type in MODEL_CONFIG_CLASSES:
                return MODEL_CONFIG_CLASSES[model_type](**model)
            raise ValueError(f"Unknown or missing model type for key: {key}")

        return {
            key: convert_model(key, model)
            for key, model in cls._require_mapping(models, "models").items()
        }

    @classmethod
    def _convert_questionnaire(cls, questionnaire: Questionnaire | dict) -> Questionnaire:
        """Convert questionnaire to an instance of Questionnaire."""
        if isinstance(questionnaire, Questionnaire):
            return questionnaire
        return Questionnaire(**cls._require_mapping(questionnaire, "questionnaire"))

    def _load_runnable_models(self, models: dict[str, Any]) -> dict[str, Any]:
        """Load models into runnable instances."""
        return {key: model.load_model() for key, model in models.items()}

    def _load_runnable_prompt(self, prompt_template: str | BaseModel) -> Any:
        """Load a prompt template into a runnable LangChain prompt."""
        if isinstance(prompt_template, str):
            return prompt_template  # Directly use if it's a string

        try:
            return prompt_template.load_prompt_template()  # type: ignore[attr-defined]
        except Exception as e:
            raise ValueError(f"Invalid prompt template: {e}") from e

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
