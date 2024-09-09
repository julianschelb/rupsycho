# ===========================================================================
#                           Data Model: Language Models
# ===========================================================================
# This file contains the data model for the language models configs.


import warnings
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union
from langchain_core.load import load
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from .prompt import (
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
)


# ------------------------------------------------
#                Generic Config
# ------------------------------------------------


# class ModelConfig(BaseModel):
#     """Base class for all model configurations."""
#     definition: Dict[str, Any] = Field(
#         {}, description="The definition of the model"
#     )
#     parameters: Dict[str, Any] = Field(
#         {}, description="The parameters for the model"
#     )
#     secrets_map: Dict[str, Any] = Field(
#         {}, description="The secrets map for the model"
#     )

#     prompt_template: Optional[Dict[str, Any]] = Field(
#         None, description="The prompt template used by the model"
#     )

# ------------------------------------------------
#                LangChain Config
# ------------------------------------------------


class LangChainModelConfig(BaseModel):
    """Configuration for a serialized LangChain model or any other runnable configuration."""

    type: str = Field("langchain",
                      description="The type of the model configuration.")

    definition: Dict[str, Any] = Field(
        ..., description="A dictionary containing the serialized LangChain model or other runnable configuration."
    )

    parameters: Dict = Field(
        {},
        description="The parameters for text generation"
    )

    prompt_template: LangchainPromptTemplateConfig = Field(
        None, description="The prompt template used by the model"
    )

    def load_model(self):
        """
        Loads and returns a LangChain model based on the provided serialized configuration.

        Parameters
        ----------
        config : LangChainModelConfig
            The configuration object containing the serialized LangChain model definition.

        Returns
        -------
        Any
            The deserialized LangChain model ready for use, or None if loading fails.
        """
        try:
            # Load the LangChain model using the provided definition
            model = load(self.definition)
            return model

        except Exception as e:
            warnings.warn(f"Failed to load LangChain model: {e}", UserWarning)
            return None


# ------------------------------------------------
#                Local HF Config
# ------------------------------------------------

class LocalHuggingFaceModelConfig(BaseModel):
    """Configuration for a local Hugging Face model."""

    name_or_path: str = Field(
        ..., description="The path to the local directory or the name of the Hugging Face model."
    )

    revision: Optional[str] = Field(
        None, description="The specific model version to use (e.g., a branch name, tag, or commit hash)."
    )

    tokenizer_name_or_path: Optional[str] = Field(
        None, description="The name or path to the tokenizer to use. Defaults to `model_name_or_path` if not specified."
    )

    cache_dir: Optional[str] = Field(
        None, description="Path to the directory where the downloaded model and tokenizer files will be cached."
    )

    use_auth_token: Optional[bool] = Field(
        False, description="Whether to use the token generated when running `huggingface-cli login` for private models."
    )

    device: Optional[int] = Field(
        -1, description="The device to load the model onto ('cpu' or 'cuda')."
    )

    task: Optional[str] = Field(
        "text-generation", description="The type of pipeline to create (e.g., 'text-generation', 'text-classification', etc.)."
    )

    parameters: Dict = Field(
        {},
        description="The parameters for text generation (e.g., max_length, temperature, etc.)."
    )

    prompt_template: Optional[Union[str, BaseModel]] = Field(
        None, description="The prompt template used by the model, if applicable."
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    def load_model(self):
        """
        Loads and returns a Hugging Face model pipeline wrapped in a LangChain HuggingFacePipeline.

        Returns
        -------
        HuggingFacePipeline
            The LangChain HuggingFacePipeline ready for integration into LangChain workflows.
        """
        try:
            # Create the Hugging Face pipeline using the `from_model_id` method
            llm = HuggingFacePipeline.from_model_id(
                model_id=self.name_or_path,
                task=self.task,
                pipeline_kwargs=dict(
                    **self.parameters  # Pass the text generation parameters from the config
                ),
                device=self.device  # TODO: Add device parameter
            )

            # Wrap the HuggingFacePipeline in a ChatHuggingFace object for LangChain integration
            return ChatHuggingFace(llm=llm)

        except Exception as e:
            raise ValueError(
                f"Failed to load the Hugging Face model: {str(e)}")


# ------------------------------------------------
#                Remote HF Config
# ------------------------------------------------

class RemoteHuggingFaceModelConfig(BaseModel):
    """Configuration for a remote Hugging Face model using the Inference Endpoint API."""

    type: str = Field("remote_huggingface",
                      description="The type of the model configuration.")

    repo_id: str = Field(
        ..., description="The repository ID of the Hugging Face model (e.g., 'HuggingFaceH4/zephyr-7b-beta')."
    )

    task: str = Field(
        ..., description="The task for the Hugging Face pipeline (e.g., 'text-generation', 'text-classification')."
    )

    huggingfacehub_api_token: str = Field(
        ..., description="The API token for accessing Hugging Face endpoints."
    )

    parameters: Dict = Field(
        {},
        description="The parameters for text generation (e.g., max_length, temperature, etc.)."
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    def load_model(self):
        """
        Loads and returns a Hugging Face model pipeline from a remote inference endpoint,
        wrapped in a LangChain HuggingFacePipeline.

        Returns
        -------
        HuggingFacePipeline
            The LangChain HuggingFacePipeline ready for integration into LangChain workflows.
        """
        try:
            # Create the Hugging Face Endpoint using the specified parameters
            endpoint = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                task=self.task,
                huggingfacehub_api_token=self.huggingfacehub_api_token,
                **self.parameters  # Pass the generation parameters
            )

            # Return the LangChain HuggingFacePipeline object with the endpoint
            return ChatHuggingFace(llm=endpoint)

        except Exception as e:
            raise ValueError(
                f"Failed to load the remote Hugging Face model: {str(e)}")

# ------------------------------------------------
#                Ollama Config
# ------------------------------------------------


class OllamaModelConfig(BaseModel):
    """
    Configuration for loading an Ollama model.
    This class holds all the necessary information for configuring and loading an OllamaLLM model.
    """

    type: str = Field("ollama",
                      description="The type of the model configuration.")

    model: str = Field(
        ..., description="The identifier for the Ollama model (e.g., 'gemma2:2b')."
    )

    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    parameters: Dict = Field(
        {},
        description="The parameters for text generation"
    )

    prompt_template: ChatPromptTemplateConfig = Field(
        None, description="The prompt template used by the model"
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    def load_model(self):
        """
        Loads and returns an Ollama model based on the provided configuration.

        Returns
        -------
        OllamaLLM
            The instantiated Ollama model ready for use.
        """
        try:
            # Create the Ollama model instance using the provided configuration
            model_ollama = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                **self.parameters  # Pass the text generation parameters
            )

            return model_ollama

        except Exception as e:
            raise ValueError(f"Failed to load the Ollama model: {str(e)}")

# ------------------------------------------------
#                OpenAI Config
# ------------------------------------------------


class OpenAIModelConfig(BaseModel):
    """Configuration for an OpenAI model."""

    type: str = Field(
        "openai", description="The type of the model configuration.")

    model: str = Field(
        ..., description="The identifier for the OpenAI model (e.g., 'gpt-4')."
    )

    api_key: Optional[str] = Field(
        None, description="The API key for accessing OpenAI's models."
    )

    base_url: Optional[str] = Field(
        None, description="The base URL for the OpenAI API endpoint."
    )

    organization: Optional[str] = Field(
        None, description="The organization ID associated with the OpenAI API key."
    )

    parameters: Dict = Field(
        {},
        description="The parameters for text generation"
    )

    prompt_template: ChatPromptTemplateConfig = Field(
        None, description="The prompt template used by the model"
    )

    class Config:
        arbitrary_types_allowed = True

    def load_model(self):
        """
        Loads and returns an OpenAI model based on the provided configuration.

        Parameters
        ----------
        config : OpenAIModelConfig
            The configuration object containing the details for loading the OpenAI model.

        Returns
        -------
        ChatOpenAI
            The instantiated OpenAI model ready for use.
        """
        try:
            # Create the OpenAI model instance using the provided config
            model_openai = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                **self.parameters  # Pass generation parameters
            )

            return model_openai

        except Exception as e:
            raise ValueError(f"Failed to load the OpenAI model: {e}")


# ------------------- Default Model -------------------

DEFAULT_MODEL_CONFIG_DICT = {
    "type": "local_huggingface",
            "name_or_path": "HuggingFaceTB/SmolLM-1.7b-Instruct",
            "task": "text-generation",
            "generation_kwargs": {
                "min_new_tokens": 2,
                "max_new_tokens": 128,
                "max_length": 1024,
                "do_sample": True,
                "repetition_penalty": 1.03,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95
            }
}

DEFAULT_MODEL_CONFIG = LocalHuggingFaceModelConfig(**DEFAULT_MODEL_CONFIG_DICT)
