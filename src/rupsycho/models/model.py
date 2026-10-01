# model.py
"""Data model: language model configurations.

Each configuration describes how to build one LangChain model and exposes ``load_model()``.
Provider SDKs are imported lazily inside ``load_model`` so that configurations can be
created, validated and serialised without the corresponding extra installed. Loading a model
whose extra is missing raises an ``ImportError`` naming the extra to install.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rupsycho._compat import load_serialized, require
from rupsycho.models.prompt import (
    ChatPromptTemplateConfig,
    LangchainPromptTemplateConfig,
)

__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_MODEL_CONFIG_DICT",
    "DeepSeekModelConfig",
    "GoogleModelConfig",
    "LangChainModelConfig",
    "LocalHuggingFaceModelConfig",
    "OllamaModelConfig",
    "OpenAIModelConfig",
    "RemoteHuggingFaceModelConfig",
]


# ------------------------------------------------
#                LangChain Config
# ------------------------------------------------


class LangChainModelConfig(BaseModel):
    """Configuration for a serialized LangChain model or any other runnable configuration.

    Attributes:
        definition: The serialized model as produced by ``langchain_core.load.dumpd``.
        parameters: Generation parameters (informational for serialized models).
        prompt_template: Optional prompt template of the model.
    """

    type: str = Field("langchain", description="The type of the model configuration.")

    definition: dict[str, Any] = Field(
        ...,
        description="A dictionary containing the serialized LangChain model or other runnable configuration.",
    )

    parameters: dict = Field({}, description="The parameters for text generation")

    prompt_template: LangchainPromptTemplateConfig | None = Field(
        None, description="The prompt template used by the model"
    )

    def load_model(self) -> Any:
        """Deserialize and return the LangChain model.

        Returns:
            The deserialized LangChain model.

        Raises:
            ValueError: If the definition cannot be deserialized.
        """
        try:
            return load_serialized(self.definition)
        except Exception as e:
            raise ValueError(f"Failed to load the LangChain model: {e}") from e


# ------------------------------------------------
#                Local HF Config
# ------------------------------------------------


class LocalHuggingFaceModelConfig(BaseModel):
    """Configuration for a Hugging Face model that runs in this process.

    Requires the ``huggingface`` extra. Pin ``revision`` to a commit hash to make the exact
    model version part of your experiment configuration.

    Example:
        ```python
        config = LocalHuggingFaceModelConfig(
            name_or_path="HuggingFaceTB/SmolLM-1.7b-Instruct",
            revision="main",
            device_map="cpu",
            parameters={"max_new_tokens": 64, "do_sample": True, "return_full_text": False},
        )
        model = config.load_model()
        ```
    """

    type: str = Field("local_huggingface", description="The type of the model configuration.")

    name_or_path: str = Field(
        ..., description="The path to the local directory or the name of the Hugging Face model."
    )

    revision: str | None = Field(
        None,
        description="The specific model version to use (e.g., a branch name, tag, or commit hash).",
    )

    tokenizer_name_or_path: str | None = Field(
        None,
        description="The name or path to the tokenizer to use. Defaults to `name_or_path` if not specified.",
    )

    cache_dir: str | None = Field(
        None,
        description="Path to the directory where the downloaded model and tokenizer files will be cached.",
    )

    huggingfacehub_api_token: str | None = Field(
        None, description="The API token for accessing gated or private Hugging Face models."
    )

    device_map: Any | None = Field(
        "auto",
        description="The device map to load the model onto ('cpu', 'cuda', or custom device map). See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map",
    )

    task: str | None = Field(
        "text-generation",
        description="The type of pipeline to create (e.g., 'text-generation').",
    )

    parameters: dict = Field(
        {}, description="The parameters for text generation (e.g., max_new_tokens, temperature)."
    )

    prompt_template: str | BaseModel | None = Field(
        None, description="The prompt template used by the model, if applicable."
    )

    bitsandbytes_config: dict | None = Field(
        None, description="Optional dictionary for bitsandbytes quantization configuration."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_model(self) -> Any:
        """Load the model, wrap it into a Transformers pipeline and return a chat model.

        Returns:
            A ``ChatHuggingFace`` around a ``HuggingFacePipeline``.

        Raises:
            ImportError: If the ``huggingface`` extra is not installed.
            ValueError: If the model cannot be loaded.
        """
        transformers = require("transformers", "huggingface", feature="Local Hugging Face models")
        lc_hf = require("langchain_huggingface", "huggingface", feature="Local Hugging Face models")

        hub_kwargs: dict[str, Any] = {}
        if self.revision:
            hub_kwargs["revision"] = self.revision
        if self.cache_dir:
            hub_kwargs["cache_dir"] = self.cache_dir
        if self.huggingfacehub_api_token:
            hub_kwargs["token"] = self.huggingfacehub_api_token

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path or self.name_or_path, **hub_kwargs
            )

            model_kwargs: dict[str, Any] = {"device_map": self.device_map, **hub_kwargs}
            if self.bitsandbytes_config:
                model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    **self.bitsandbytes_config
                )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.name_or_path, **model_kwargs
            )

            pipe = transformers.pipeline(
                task=self.task,
                model=model,
                tokenizer=tokenizer,
                **self.parameters,
            )

            return lc_hf.ChatHuggingFace(
                llm=lc_hf.HuggingFacePipeline(pipeline=pipe, model_id=self.name_or_path),
                tokenizer=tokenizer,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to load the Hugging Face model '{self.name_or_path}' "
                f"with task '{self.task}': {e}"
            ) from e


# ------------------------------------------------
#                Remote HF Config
# ------------------------------------------------


class RemoteHuggingFaceModelConfig(BaseModel):
    """Configuration for a model served by a Hugging Face Inference Endpoint.

    Requires the ``huggingface`` extra and a Hugging Face token in the environment
    (``HUGGINGFACEHUB_API_TOKEN``).
    """

    type: str = Field("remote_huggingface", description="The type of the model configuration.")

    repo_id: str = Field(
        ...,
        description="The repository ID of the Hugging Face model (e.g., 'HuggingFaceH4/zephyr-7b-beta').",
    )

    task: str = Field(
        ...,
        description="The task for the Hugging Face pipeline (e.g., 'text-generation').",
    )

    parameters: dict = Field(
        {}, description="The parameters for text generation (e.g., max_new_tokens, temperature)."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_model(self) -> Any:
        """Create the endpoint client and return it as a chat model.

        Returns:
            A ``ChatHuggingFace`` around a ``HuggingFaceEndpoint``.

        Raises:
            ImportError: If the ``huggingface`` extra is not installed.
            ValueError: If the endpoint cannot be created.
        """
        lc_hf = require("langchain_huggingface", "huggingface", feature="Hugging Face endpoints")
        try:
            endpoint = lc_hf.HuggingFaceEndpoint(
                repo_id=self.repo_id, task=self.task, **self.parameters
            )
            return lc_hf.ChatHuggingFace(llm=endpoint)
        except Exception as e:
            raise ValueError(f"Failed to load the remote Hugging Face model: {e}") from e


# ------------------------------------------------
#                Ollama Config
# ------------------------------------------------


class OllamaModelConfig(BaseModel):
    """Configuration for a model served by a local or remote Ollama server.

    Requires the ``ollama`` extra.
    """

    type: str = Field("ollama", description="The type of the model configuration.")

    model: str = Field(..., description="The identifier for the Ollama model (e.g., 'gemma2:2b').")

    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    parameters: dict = Field({}, description="The parameters for text generation")

    prompt_template: ChatPromptTemplateConfig | None = Field(
        None, description="The prompt template used by the model"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_model(self) -> Any:
        """Create the Ollama client.

        Returns:
            An ``OllamaLLM``.

        Raises:
            ImportError: If the ``ollama`` extra is not installed.
            ValueError: If the client cannot be created.
        """
        llms = require("langchain_ollama.llms", "ollama", feature="Ollama models")
        try:
            return llms.OllamaLLM(model=self.model, base_url=self.base_url, **self.parameters)
        except Exception as e:
            raise ValueError(f"Failed to load the Ollama model: {e}") from e


# ------------------------------------------------
#                OpenAI Config
# ------------------------------------------------


class OpenAIModelConfig(BaseModel):
    """Configuration for an OpenAI (or OpenAI-compatible) chat model.

    Requires the ``openai`` extra. The API key is read from ``api_key`` or, if unset, from the
    ``OPENAI_API_KEY`` environment variable. Prefer the environment variable so that keys never
    end up in shared configuration files.
    """

    type: str = Field("openai", description="The type of the model configuration.")

    name_or_path: str = Field(
        ..., description="The identifier for the OpenAI model (e.g., 'gpt-4')."
    )

    api_key: str | None = Field(None, description="The API key for accessing OpenAI's models.")

    base_url: str | None = Field(None, description="The base URL for the OpenAI API endpoint.")

    organization: str | None = Field(
        None, description="The organization ID associated with the OpenAI API key."
    )

    parameters: dict = Field({}, description="The parameters for text generation")

    prompt_template: ChatPromptTemplateConfig | None = Field(
        None, description="The prompt template used by the model"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_model(self) -> Any:
        """Create the OpenAI chat model.

        Returns:
            A ``ChatOpenAI``.

        Raises:
            ImportError: If the ``openai`` extra is not installed.
            ValueError: If the client cannot be created.
        """
        lc_openai = require("langchain_openai", "openai", feature="OpenAI models")
        try:
            return lc_openai.ChatOpenAI(
                model=self.name_or_path,
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                **self.parameters,
            )
        except Exception as e:
            raise ValueError(f"Failed to load the OpenAI model: {e}") from e


# ------------------------------------------------
#               Remote Google Config
# ------------------------------------------------


class GoogleModelConfig(BaseModel):
    """Configuration for a Google Gemini chat model.

    Requires the ``google`` extra. Gemini models cannot be seeded; see
    [`rupsycho.seeding`][rupsycho.seeding].
    """

    type: str = Field("google", description="The type of the model configuration.")

    name_or_path: str = Field(
        ..., description="The identifier for the Google model (e.g., 'gemini-2.0-flash')."
    )

    api_key: str | None = Field(None, description="The API key for accessing Google models.")

    parameters: dict = Field({}, description="The parameters for text generation")

    prompt_template: ChatPromptTemplateConfig | None = Field(
        None, description="The prompt template used by the model"
    )

    def load_model(self) -> Any:
        """Create the Google chat model.

        Returns:
            A ``ChatGoogleGenerativeAI``.

        Raises:
            ImportError: If the ``google`` extra is not installed.
            ValueError: If the client cannot be created.
        """
        lc_google = require("langchain_google_genai", "google", feature="Google models")
        try:
            return lc_google.ChatGoogleGenerativeAI(
                model=self.name_or_path, api_key=self.api_key, **self.parameters
            )
        except Exception as e:
            raise ValueError(f"Failed to load the Google model: {e}") from e


# ------------------------------------------------
#               Remote DeepSeek Config
# ------------------------------------------------


class DeepSeekModelConfig(BaseModel):
    """Configuration for a DeepSeek chat model.

    Requires the ``deepseek`` extra. The API key is read from ``api_key`` or, if unset, from
    the ``DEEPSEEK_API_KEY`` environment variable.
    """

    type: str = Field("deepseek", description="The type of the model configuration.")

    name_or_path: str = Field(
        ..., description="The identifier for the DeepSeek model (e.g. 'deepseek-chat')."
    )

    api_key: str | None = Field(None, description="The API key for accessing DeepSeek models.")

    parameters: dict = Field({}, description="The parameters for text generation")

    prompt_template: ChatPromptTemplateConfig | None = Field(
        None, description="The prompt template used by the model"
    )

    def load_model(self) -> Any:
        """Create the DeepSeek chat model.

        Returns:
            A ``ChatDeepSeek``.

        Raises:
            ImportError: If the ``deepseek`` extra is not installed.
            ValueError: If the client cannot be created.
        """
        lc_deepseek = require("langchain_deepseek", "deepseek", feature="DeepSeek models")
        kwargs: dict[str, Any] = dict(self.parameters)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            return lc_deepseek.ChatDeepSeek(model=self.name_or_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load the DeepSeek model: {e}") from e


# ------------------- Default Model -------------------

DEFAULT_MODEL_CONFIG_DICT: dict[str, Any] = {
    "type": "local_huggingface",
    "name_or_path": "HuggingFaceTB/SmolLM-1.7b-Instruct",
    "task": "text-generation",
    "device_map": "cpu",
    "parameters": {
        "max_new_tokens": 64,
        "temperature": 1.0,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False,
    },
}
"""Small default model used when a configuration defines no ``models`` key at all."""

DEFAULT_MODEL_CONFIG = LocalHuggingFaceModelConfig(**DEFAULT_MODEL_CONFIG_DICT)
