"""Model configurations: wiring of fields, lazy imports and error messages."""

import pytest
from pydantic import ValidationError

from rupsycho._compat import is_available, require
from rupsycho.models.model import (
    DEFAULT_MODEL_CONFIG,
    DeepSeekModelConfig,
    GoogleModelConfig,
    LangChainModelConfig,
    LocalHuggingFaceModelConfig,
    OllamaModelConfig,
    OpenAIModelConfig,
)


class TestDefaultModel:
    def test_generation_parameters_are_applied(self):
        """Regression: the defaults used to sit under an ignored 'pipeline_kwargs' key."""
        parameters = DEFAULT_MODEL_CONFIG.parameters
        assert parameters["max_new_tokens"] == 64
        assert parameters["do_sample"] is True
        assert parameters["return_full_text"] is False


class TestLocalHuggingFace:
    def test_loads_a_chat_model_from_a_directory(self, tiny_hf_config):
        model = LocalHuggingFaceModelConfig(**tiny_hf_config).load_model()
        assert type(model).__name__ == "ChatHuggingFace"
        assert model.invoke("hello").content != ""

    def test_hub_options_are_forwarded(self, tiny_hf_config, monkeypatch, tmp_path):
        import transformers

        seen = {}
        real_tokenizer = transformers.AutoTokenizer.from_pretrained
        real_model = transformers.AutoModelForCausalLM.from_pretrained

        def tokenizer(name, **kwargs):
            seen["tokenizer"] = kwargs
            return real_tokenizer(name, **kwargs)

        def model(name, **kwargs):
            seen["model"] = kwargs
            return real_model(name, **kwargs)

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)

        LocalHuggingFaceModelConfig(
            **tiny_hf_config,
            revision="main",
            cache_dir=str(tmp_path),
            huggingfacehub_api_token="hf_secret",
        ).load_model()

        for call in ("tokenizer", "model"):
            assert seen[call]["revision"] == "main"
            assert seen[call]["cache_dir"] == str(tmp_path)
            assert seen[call]["token"] == "hf_secret"

    def test_hub_options_are_omitted_when_unset(self, tiny_hf_config, monkeypatch):
        import transformers

        seen = {}
        real = transformers.AutoModelForCausalLM.from_pretrained

        def model(name, **kwargs):
            seen.update(kwargs)
            return real(name, **kwargs)

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)
        LocalHuggingFaceModelConfig(**tiny_hf_config).load_model()
        assert set(seen) == {"device_map"}

    def test_a_missing_model_raises_a_clear_error(self, tmp_path):
        config = LocalHuggingFaceModelConfig(name_or_path=str(tmp_path / "nope"))
        pytest.importorskip("transformers")
        with pytest.raises(ValueError, match="Failed to load the Hugging Face model"):
            config.load_model()

    def test_a_missing_extra_names_what_to_install(self, tiny_hf_config, monkeypatch):
        import importlib

        real = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(ImportError, match=r"pip install 'rupsycho\[huggingface\]'"):
            LocalHuggingFaceModelConfig(**tiny_hf_config).load_model()


class TestOtherBackends:
    def test_openai(self):
        pytest.importorskip("langchain_openai")
        model = OpenAIModelConfig(
            name_or_path="gpt-4o-mini", api_key="k", parameters={"temperature": 0.2}
        ).load_model()
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.2

    def test_deepseek_uses_the_configured_key(self):
        pytest.importorskip("langchain_deepseek")
        model = DeepSeekModelConfig(name_or_path="deepseek-chat", api_key="abc").load_model()
        assert model.api_key.get_secret_value() == "abc"

    def test_ollama(self):
        pytest.importorskip("langchain_ollama")
        model = OllamaModelConfig(model="llama3", base_url="http://example:1").load_model()
        assert (model.model, model.base_url) == ("llama3", "http://example:1")

    def test_google(self):
        pytest.importorskip("langchain_google_genai")
        model = GoogleModelConfig(name_or_path="gemini-2.0-flash", api_key="k").load_model()
        assert "gemini-2.0-flash" in model.model

    def test_serialized_langchain_models_round_trip(self, monkeypatch):
        pytest.importorskip("langchain_openai")
        from langchain_core.load import dumpd
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        config = LangChainModelConfig(definition=dumpd(ChatOpenAI(model="gpt-4o-mini")))
        assert config.load_model().model_name == "gpt-4o-mini"

    def test_invalid_serialized_definitions_raise(self):
        definition = {"lc": 1, "type": "constructor", "id": ["no", "such", "Model"], "kwargs": {}}
        with pytest.raises(ValueError, match="Failed to load the LangChain model"):
            LangChainModelConfig(definition=definition).load_model()

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            OpenAIModelConfig()


class TestOptionalDependencyHelpers:
    def test_require_returns_the_module(self):
        assert require("json", "none").dumps({}) == "{}"

    def test_require_explains_how_to_install(self):
        with pytest.raises(ImportError, match=r"pip install 'rupsycho\[openai\]'"):
            require("definitely_not_installed_xyz", "openai", feature="OpenAI models")

    def test_is_available(self):
        assert is_available("json") is True
        assert is_available("definitely_not_installed_xyz") is False
