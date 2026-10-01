"""Seeds must reach the model: local pipelines via set_seed, API models via a seed field."""

import copy
import warnings

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import rupsycho as rup
from rupsycho import seeding
from rupsycho.seeding import is_thread_safe, register_seeder, seed_model, supports_seeding

PROMPT = ChatPromptTemplate.from_messages([("user", "{x}")])


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test starts with the built-in seeders and a fresh 'warned once' set."""
    monkeypatch.setattr(seeding, "_REGISTRY", [])
    monkeypatch.setattr(seeding, "_WARNED", set())


def sample(model, seed, text="hello"):
    chain = PROMPT | seed_model(model, seed) | StrOutputParser()
    return chain.invoke({"x": text})


class TestLocalHuggingFace:
    @pytest.fixture
    def model(self, tiny_hf_config):
        from rupsycho.models.model import LocalHuggingFaceModelConfig

        return LocalHuggingFaceModelConfig(**tiny_hf_config).load_model()

    def test_same_seed_reproduces_the_answer(self, model):
        assert sample(model, 1) == sample(model, 1)

    def test_different_seeds_give_different_answers(self, model):
        assert sample(model, 1) != sample(model, 2)

    def test_seed_does_not_depend_on_earlier_calls(self, model):
        first = sample(model, 1)
        sample(model, 2)
        sample(model, 3)
        assert sample(model, 1) == first

    def test_unseeded_sampling_is_not_reproducible(self, model):
        """Documents the problem seeding solves: global RNG state advances between calls."""
        chain = PROMPT | model | StrOutputParser()
        assert chain.invoke({"x": "hello"}) != chain.invoke({"x": "hello"})

    def test_model_is_not_thread_safe(self, model):
        assert is_thread_safe(model) is False
        assert supports_seeding(model) is True

    def test_experiment_run_is_reproducible(self, config_dict, tiny_hf_config):
        def answers(seeds):
            config = copy.deepcopy(config_dict)
            config["models"] = {"tiny": tiny_hf_config}
            config["parameters"] = {"seeds": seeds}
            experiment = rup.experiment_from_dict(config)
            experiment.run(show_progress=False)
            return experiment.get_answers_as_dataframe()["Answer"].tolist()

        assert answers(["1"]) == answers(["1"])
        assert answers(["1"]) != answers(["2"])


class TestApiModels:
    def test_openai_receives_the_seed(self):
        pytest.importorskip("langchain_openai")
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(api_key="test", model="gpt-4o-mini")
        seeded = seed_model(model, 7)

        assert seeded.seed == 7
        assert model.seed is None  # the original is untouched
        assert seeded._get_request_payload([HumanMessage("hi")])["seed"] == 7

    def test_deepseek_receives_the_seed(self):
        pytest.importorskip("langchain_deepseek")
        from langchain_deepseek import ChatDeepSeek

        seeded = seed_model(ChatDeepSeek(api_key="test", model="deepseek-chat"), 3)
        assert seeded.seed == 3

    def test_ollama_receives_the_seed_as_an_option(self):
        """bind(seed=...) would become an unsupported keyword of the Ollama client."""
        pytest.importorskip("langchain_ollama")
        from langchain_ollama.llms import OllamaLLM

        seeded = seed_model(OllamaLLM(model="llama3"), 11)

        params = seeded._generate_params("hi")
        assert params["options"]["seed"] == 11
        assert "seed" not in params

    def test_huggingface_endpoint_receives_the_seed_in_the_request(self, monkeypatch):
        pytest.importorskip("langchain_huggingface")
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id="org/model", task="text-generation", huggingfacehub_api_token="token"
        )
        chat = ChatHuggingFace(llm=endpoint)
        seeded = seed_model(chat, 5)
        assert supports_seeding(chat) is True
        assert "seed" not in (chat.model_kwargs or {})  # the original is untouched

        sent = {}

        def fake_chat_completion(messages, **params):
            sent.update(params)
            raise RuntimeError("stop here")

        monkeypatch.setattr(endpoint.client, "chat_completion", fake_chat_completion)
        with pytest.raises(RuntimeError, match="stop here"):
            seeded.invoke("hello")
        assert sent["seed"] == 5

    def test_models_without_seed_support_warn_once(self):
        pytest.importorskip("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="test")
        assert supports_seeding(model) is False
        with pytest.warns(UserWarning, match="cannot be seeded"):
            assert seed_model(model, 1) is model
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            seed_model(model, 2)  # second time: silent

    def test_api_models_are_thread_safe(self):
        assert is_thread_safe(FakeListLLM(responses=["x"])) is True


class TestRegistry:
    def test_unknown_models_are_returned_unchanged_with_a_warning(self):
        model = FakeListLLM(responses=["x"])
        with pytest.warns(UserWarning, match="FakeListLLM"):
            assert seed_model(model, 1) is model

    def test_registered_seeder_takes_precedence(self):
        seen = []

        def seeder(model, seed):
            seen.append(seed)
            return model

        register_seeder(FakeListLLM, seeder)
        model = FakeListLLM(responses=["x"])

        assert supports_seeding(model) is True
        assert seed_model(model, "5") is model  # the seed is normalised to int
        assert seen == [5]

    def test_latest_registration_wins(self):
        calls = []
        register_seeder(FakeListLLM, lambda m, s: calls.append("first") or m)
        register_seeder(FakeListLLM, lambda m, s: calls.append("second") or m)
        seed_model(FakeListLLM(responses=["x"]), 1)
        assert calls == ["second"]
