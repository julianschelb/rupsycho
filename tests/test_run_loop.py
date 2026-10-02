"""Tests for the run loop in ``rupsycho.mixins.experiment_processing``.

Everything runs against fake LLMs: the recording model below counts the chain invocations and
captures the prompt and the seed of every call. The policy for failed calls, concurrency and
repeated runs is covered in ``test_run_semantics.py``; this file tests the loop in detail.
"""

import copy
import logging
import subprocess
import sys
import textwrap
import threading
import warnings
from collections import namedtuple
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.llms import LLM
from langchain_core.load import dumpd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBinding, RunnableLambda
from pydantic import Field, PrivateAttr, ValidationError

import rupsycho as rup
from rupsycho import seeding
from rupsycho.callbacks import Callback
from rupsycho.mixins.experiment_processing import (
    DEFAULT_SEED,
    ExperimentProcessingMixin,
    RunSummary,
)
from rupsycho.models.model import LangChainModelConfig, LocalHuggingFaceModelConfig
from rupsycho.models.questionnaire import InstructionItem

LOGGER_NAME = "rupsycho.mixins.experiment_processing"

# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------

Call = namedtuple("Call", ["prompt", "seed", "kwargs"])


def numbered(n: int = 300, prefix: str = "a") -> list[str]:
    """Distinct responses ``a0, a1, ...``: the k-th call of a model answers ``a<k>``."""
    return [f"{prefix}{i}" for i in range(n)]


class RecordingLLM(LLM):
    """A fake LLM that records every call it receives.

    It has a ``seed`` field, which makes it a model that can be seeded: a run hands the seed of
    each repetition over on a *copy* of the model (see ``rupsycho.seeding``). All copies share
    ``log`` and ``trace``, and the k-th call overall - not per copy - is answered with
    ``responses[k]`` (the last response repeats). ``trace`` can be shared between models to see
    the global call order. ``failing_calls`` are the call numbers that raise (a set, or a
    ``{number: message}`` dictionary).
    """

    responses: list[str] = Field(default_factory=lambda: ["x"])
    tag: str = ""
    seed: int | None = None
    log: Any = Field(default_factory=list)
    trace: Any = None
    failing_calls: Any = ()
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        with self._lock:
            call_number = len(self.log)
            self.log.append(Call(prompt, self.seed, kwargs))
            if self.trace is not None:
                self.trace.append((self.tag, self.seed))
        if call_number in self.failing_calls:
            failing = self.failing_calls
            raise RuntimeError(failing[call_number] if isinstance(failing, dict) else "boom")
        return self.responses[min(call_number, len(self.responses) - 1)]


class LengthLLM(LLM):
    """Answers ``len<number of characters>``: a pure function of the prompt it receives."""

    seed: int | None = None

    @property
    def _llm_type(self) -> str:
        return "length"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return f"len{len(prompt)}"


class SeedEchoLLM(LLM):
    """Answers ``seed-<seed it was given>``."""

    seed: int | None = None

    @property
    def _llm_type(self) -> str:
        return "seed-echo"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return f"seed-{self.seed}"


class RecordingCallback(Callback):
    """Records every argument of ``save_answer``, plus the answers the item held at that time."""

    def __init__(self):
        self.calls = []

    def save_answer(
        self,
        experiment,
        instruction_item_id,
        instruction_item,
        model_id,
        profile_id,
        random_seed,
        time,
        answer,
    ):
        self.calls.append(
            SimpleNamespace(
                experiment=experiment,
                item_id=instruction_item_id,
                item=instruction_item,
                model_id=model_id,
                profile_id=profile_id,
                random_seed=random_seed,
                time=time,
                answer=answer,
                stored=copy.deepcopy(instruction_item.model_dump()["answers"]),
            )
        )


class ExplodingCallback(Callback):
    def save_answer(self, *args):
        raise RuntimeError("callback boom")


class CountingBar:
    """Minimal stand-in for a tqdm progress bar."""

    def __init__(self):
        self.updates = 0

    def update(self, n=1):
        self.updates += n


def make_experiment(config_dict, models=None, seeds=None, **parameters):
    """Build a BFI experiment (2 personas, 4 items) and add the given ``{id: model}`` models."""
    config = copy.deepcopy(config_dict)
    if seeds is not None:
        config["parameters"]["seeds"] = list(seeds)
    config["parameters"].update(parameters)
    experiment = rup.experiment_from_dict(config)
    for identifier, model in (models or {}).items():
        experiment.add_model(model, identifier=identifier)
    return experiment


def grid_of(experiment):
    """``(number of items, persona ids)`` of an experiment."""
    return (
        len(experiment.questionnaire.instruction_items),
        list(experiment.demographic_profiles),
    )


def cumulative_config(config_dict):
    """The BFI config with a user template that only contains ``{question}``."""
    config = copy.deepcopy(config_dict)
    system = config["prompt_template"]["messages"][0]["content"]
    config["prompt_template"]["messages"] = [
        {"role": "system", "content": system + "\nAnswer Options: {answer_options}"},
        {"role": "user", "content": "Question: {question}\nAnswer:"},
    ]
    return config


def human_part(prompt: str) -> str:
    """The user message of a rendered chat prompt (``System: ...\\nHuman: ...``)."""
    return prompt.split("\nHuman: ", 1)[1]


# ------------------------------------------------------------------------------------------
# Iteration order and number of chain invocations
# ------------------------------------------------------------------------------------------


def test_iteration_order_is_model_seed_item_persona(config_dict):
    trace: list = []
    models = {
        model_id: RecordingLLM(responses=numbered(), tag=model_id, trace=trace)
        for model_id in ("m1", "m2")
    }
    experiment = make_experiment(config_dict, models, seeds=["1", "2"])
    callback = RecordingCallback()

    experiment.run(callbacks=[callback], show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    expected = [
        (model_id, seed, item_id, persona_id)
        for model_id in ("m1", "m2")
        for seed in ("1", "2")
        for item_id in range(n_items)
        for persona_id in persona_ids
    ]
    got = [(c.model_id, c.random_seed, c.item_id, c.profile_id) for c in callback.calls]
    assert got == expected
    # the models themselves were called in the same order, with the seed as an int
    assert trace == [(model_id, int(seed)) for model_id, seed, _, _ in expected]


@pytest.mark.parametrize(
    "model_ids, seeds",
    [
        (["m"], ["1"]),
        (["m"], ["1", "2", "3"]),
        (["m1", "m2"], ["5"]),
        (["m1", "m2", "m3"], ["5", "6"]),
    ],
)
def test_number_of_chain_invocations(config_dict, model_ids, seeds):
    models = {model_id: RecordingLLM(responses=numbered()) for model_id in model_ids}
    experiment = make_experiment(config_dict, models, seeds=seeds)
    n_items, persona_ids = grid_of(experiment)
    expected = len(model_ids) * len(seeds) * n_items * len(persona_ids)
    assert experiment._calculate_total_iterations() == expected

    bar = CountingBar()
    summary = experiment.process_single_experiment(cumulative=False, pbar=bar)

    assert [len(model.log) for model in models.values()] == [
        len(seeds) * n_items * len(persona_ids)
    ] * len(model_ids)
    assert sum(len(model.log) for model in models.values()) == expected
    assert bar.updates == expected
    assert isinstance(summary, RunSummary) and summary.n_calls == expected


def test_process_single_experiment_creates_its_own_progress_bar(config_dict, capsys):
    experiment = make_experiment(config_dict, {"m": RecordingLLM()})

    summary = experiment.process_single_experiment(cumulative=False)  # pbar=None creates a bar

    assert summary.n_calls == len(experiment.get_answers_as_dataframe()) == 8
    assert "8/8" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name, shown", [("My study", "My study"), (None, "Experiment")], ids=["named", "unnamed"]
)
def test_the_progress_bar_shows_the_experiment_name_and_counts_the_calls(
    config_dict, capsys, name, shown
):
    experiment = make_experiment(config_dict, {"m": RecordingLLM()})
    experiment.name = name

    experiment.run()  # the progress bar is on by default

    err = capsys.readouterr().err
    assert f"{shown}:" in err and "8/8" in err and "prompts" in err


def test_the_progress_bar_counts_failed_calls_too(config_dict):
    experiment = make_experiment(config_dict, {"m": RecordingLLM(failing_calls={2, 3})})
    bar = CountingBar()

    summary = experiment.process_single_experiment(cumulative=False, pbar=bar, on_error="ignore")

    assert summary.n_failed == 2
    assert bar.updates == 8


def test_multiple_models_store_answers_under_their_own_id(config_dict):
    models = {
        "first": RecordingLLM(responses=["one"]),
        "second": RecordingLLM(responses=["two"]),
    }
    experiment = make_experiment(config_dict, models)
    experiment.run(show_progress=False)

    _, persona_ids = grid_of(experiment)
    for item in experiment.questionnaire.instruction_items:
        assert item.get_all_answers() == {
            "first": {persona_id: {"7": "one"} for persona_id in persona_ids},
            "second": {persona_id: {"7": "two"} for persona_id in persona_ids},
        }


def test_models_run_in_the_order_they_were_added_and_lazy_ones_load_just_in_time(
    config_dict, monkeypatch
):
    events: list = []

    def load_model(self):
        events.append(("load", self.name_or_path))
        return RecordingLLM(tag=self.name_or_path, trace=events)

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    config = copy.deepcopy(config_dict)
    config["models"] = {
        "first": {"type": "local_huggingface", "name_or_path": "org/first"},
        "second": {"type": "local_huggingface", "name_or_path": "org/second"},
    }
    experiment = rup.experiment_from_dict(config)
    assert events == []  # nothing is loaded when the experiment is created

    experiment.run(show_progress=False)

    # one model at a time: it is loaded right before its first call and not before
    assert events == (
        [("load", "org/first")]
        + [("org/first", 7)] * 8
        + [("load", "org/second")]
        + [("org/second", 7)] * 8
    )


def test_models_stay_available_after_the_run(config_dict):
    models = {"m1": RecordingLLM(), "m2": RecordingLLM()}
    experiment = make_experiment(config_dict, models)

    experiment.run(show_progress=False)

    assert list(experiment.runnable_models) == ["m1", "m2"]
    assert experiment.get_model("m1") is models["m1"]
    assert experiment.get_model("m2") is models["m2"]
    assert experiment.list_models() == ["m1", "m2"]


def test_an_experiment_can_be_run_repeatedly(config_dict):
    llm = RecordingLLM(responses=numbered(), tag="m")
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["1"])

    first = experiment.run(show_progress=False)
    second = experiment.run(show_progress=False)

    assert (first.n_calls, second.n_calls) == (8, 8) and len(llm.log) == 16
    # a repetition answers the same slots again: the answers are replaced, not duplicated
    df = experiment.get_answers_as_dataframe()
    assert len(df) == 8
    assert df["Answer"].tolist() == [f"a{8 + k}" for k in range(8)]


def test_prompt_inputs_are_built_once_per_run(config_dict, monkeypatch):
    calls = []
    original = ExperimentProcessingMixin._create_input_dict

    def spy(self, profile, item):
        calls.append(item.question)
        return original(self, profile, item)

    monkeypatch.setattr(ExperimentProcessingMixin, "_create_input_dict", spy)
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2")}
    experiment = make_experiment(config_dict, models, seeds=["1", "2", "3"])
    experiment.run(show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    # they depend on neither the model nor the seed
    assert len(calls) == n_items * len(persona_ids)


def test_chain_is_built_once_per_model_and_seed(config_dict, monkeypatch):
    seen = []
    original = ExperimentProcessingMixin._get_chain

    def spy(self, *args, **kwargs):
        seen.append((args[3], kwargs["seed"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentProcessingMixin, "_get_chain", spy)
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2")}
    experiment = make_experiment(config_dict, models, seeds=["1", "2", "3"])
    experiment.run(show_progress=False)

    assert seen == [(mid, seed) for mid in ("m1", "m2") for seed in (1, 2, 3)]


def test_experiment_without_personas_generates_nothing(config_dict):
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm})
    experiment.clear_personas()

    summary = experiment.run(show_progress=False)

    assert llm.log == []
    assert (summary.n_calls, summary.n_failed) == (0, 0)
    assert len(experiment.get_answers_as_dataframe()) == 0


# ------------------------------------------------------------------------------------------
# Seeds
# ------------------------------------------------------------------------------------------


def test_the_seed_of_each_repetition_reaches_the_model(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["11", "22"])

    experiment.run(show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    per_seed = n_items * len(persona_ids)
    assert [call.seed for call in llm.log] == [11] * per_seed + [22] * per_seed
    assert all(type(call.seed) is int for call in llm.log)
    assert all(call.kwargs == {} for call in llm.log)  # nothing but the seed is applied
    assert llm.seed is None  # the model that was added is never modified: calls use a copy


def test_each_call_gets_the_seed_of_its_own_repetition_when_calls_run_in_parallel(config_dict):
    def run(workers):
        experiment = make_experiment(config_dict, {"m": SeedEchoLLM()}, seeds=["1", "2", "3"])
        experiment.run(max_concurrency=workers, show_progress=False)
        return experiment.get_answers_as_dataframe()

    parallel = run(4)

    assert (parallel["Answer"] == "seed-" + parallel["Run Seed"]).all()
    assert len(parallel) == 3 * 8
    assert parallel.equals(run(1))


@pytest.mark.parametrize(
    "seeds, stored, given",
    [
        ([1, 2], ["1", "2"], [1, 2]),
        ([" 5 ", "-3"], ["5", "-3"], [5, -3]),
        ([0, "7"], ["0", "7"], [0, 7]),
    ],
    ids=["ints", "whitespace-and-negative", "mixed"],
)
def test_integer_seeds_are_stored_as_strings_and_reach_the_model_as_integers(
    config_dict, seeds, stored, given
):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config_dict, {"m": llm}, seeds=seeds)

    experiment.run(show_progress=False)

    assert experiment.parameters.seeds == stored
    assert [call.seed for call in llm.log[::8]] == given  # one call per repetition and 8 calls each
    persona_id = next(iter(experiment.demographic_profiles))
    assert set(
        experiment.questionnaire.instruction_items[0].get_all_answers()["m"][persona_id]
    ) == set(stored)


def test_models_without_seed_support_are_used_as_they_are_and_warn_once(config_dict, monkeypatch):
    monkeypatch.setattr(seeding, "_WARNED", set())
    llm = FakeListLLM(responses=["3"])
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["1", "2"])

    with pytest.warns(UserWarning, match="Models of type FakeListLLM cannot be seeded") as record:
        experiment.run(show_progress=False)

    ours = [w for w in record if "cannot be seeded" in str(w.message)]
    assert len(ours) == 1  # once per model type, not once per repetition or call
    assert experiment.get_model("m") is llm
    assert len(experiment.get_answers_as_dataframe()) == 16


def test_a_registered_seeder_decides_how_the_seed_is_applied(config_dict, monkeypatch):
    class Registered(FakeListLLM):
        pass

    seen = []

    def seeder(model, seed):
        seen.append(seed)
        return model

    monkeypatch.setattr(seeding, "_REGISTRY", [])
    seeding.register_seeder(Registered, seeder)
    experiment = make_experiment(config_dict, {"m": Registered(responses=["3"])}, seeds=["3", "4"])

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*cannot be seeded")  # it *can* be seeded now
        experiment.run(show_progress=False)

    assert seen == [3, 4]  # once per repetition, as an int


def test_answers_are_keyed_by_the_seed_string_from_the_configuration(config_dict):
    experiment = make_experiment(
        config_dict, {"m": RecordingLLM(responses=["x"])}, seeds=["7", "8"]
    )
    experiment.run(show_progress=False)

    item = experiment.questionnaire.instruction_items[0]
    persona_id = next(iter(experiment.demographic_profiles))
    assert item.get_answer("m", persona_id, "7") == "x"
    assert item.get_answer("m", persona_id, "8") == "x"
    assert set(item.get_all_answers()["m"][persona_id]) == {"7", "8"}


@pytest.mark.parametrize("seeds", [None, []], ids=["null", "empty"])
def test_default_seed_is_used_when_no_seeds_are_configured(config_dict, seeds):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = seeds
    experiment = rup.experiment_from_dict(config)
    llm = RecordingLLM(responses=["x"])
    experiment.add_model(llm, identifier="m")
    assert DEFAULT_SEED == 42
    assert [str(seed) for seed in experiment._get_seed_values()] == ["42"]

    experiment.run(show_progress=False)

    assert {call.seed for call in llm.log} == {42}
    persona_id = next(iter(experiment.demographic_profiles))
    answers = experiment.questionnaire.instruction_items[0].get_all_answers()["m"][persona_id]
    assert {str(seed): answer for seed, answer in answers.items()} == {"42": "x"}


@pytest.mark.parametrize("seeds", [["abc"], ["1.5"], ["1", ""], [None]], ids=str)
def test_seed_that_is_not_an_integer_is_rejected_when_the_experiment_is_loaded(config_dict, seeds):
    """It used to stop the run with "invalid literal for int()" after the first model was loaded."""
    with pytest.raises(ValidationError, match="seeds"):
        make_experiment(config_dict, {"m": RecordingLLM()}, seeds=seeds)


# ------------------------------------------------------------------------------------------
# Parameters and lazy loading of model configurations
# ------------------------------------------------------------------------------------------


@pytest.fixture
def lazy_config(config_dict):
    """BFI config with one lazily loaded Hugging Face model config that carries parameters."""
    config = copy.deepcopy(config_dict)
    config["parameters"].update({"temperature": 0.9, "top_k": 3})
    config["models"] = {
        "lazy": {
            "type": "local_huggingface",
            "name_or_path": "org/model",
            "parameters": {"temperature": 0.1, "max_new_tokens": 5},
        }
    }
    return config


def test_lazy_model_config_is_loaded_when_its_turn_comes(lazy_config, monkeypatch):
    loaded = []
    llm = RecordingLLM(responses=["lazy-answer"])

    def load_model(self):
        loaded.append(self.name_or_path)
        return llm

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    experiment = rup.experiment_from_dict(lazy_config)

    # nothing is loaded at construction time; runnable_models holds the (non-Runnable) config
    assert loaded == []
    assert isinstance(experiment.runnable_models["lazy"], LocalHuggingFaceModelConfig)

    experiment.run(show_progress=False)

    assert loaded == ["org/model"]
    assert len(llm.log) == 8
    assert experiment.get_answers()[0] == {
        "lazy": {pid: {"7": "lazy-answer"} for pid in experiment.demographic_profiles}
    }
    # the loaded model is released, the definition stays
    assert experiment.runnable_models["lazy"] is experiment.models["lazy"]


def test_a_lazy_model_is_loaded_again_for_every_run(lazy_config, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        LocalHuggingFaceModelConfig,
        "load_model",
        lambda self: loaded.append(self.name_or_path) or RecordingLLM(responses=["x"]),
    )
    experiment = rup.experiment_from_dict(lazy_config)

    experiment.run(show_progress=False)
    experiment.run(show_progress=False)

    assert loaded == ["org/model", "org/model"]


def test_model_parameters_come_from_the_model_configuration_only(lazy_config, monkeypatch):
    seen = []

    def load_model(self):
        seen.append(dict(self.parameters))
        return RecordingLLM(responses=["x"])

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    experiment = rup.experiment_from_dict(lazy_config)

    experiment.run(show_progress=False)

    # the experiment-level temperature (0.9) and top_k are not merged into the model's own
    assert seen == [{"temperature": 0.1, "max_new_tokens": 5}]
    assert experiment.models["lazy"].parameters == {"temperature": 0.1, "max_new_tokens": 5}
    assert experiment.parameters.temperature == 0.9 and experiment.parameters.top_k == 3


def test_experiment_level_parameters_are_stored_but_not_passed_to_the_models(config_dict):
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["1"], top_p=0.5)

    experiment.run(show_progress=False)

    assert experiment.parameters.top_p == 0.5
    assert all(call.kwargs == {} for call in llm.log)
    assert not hasattr(llm, "top_p")


def test_eager_loading_loads_the_models_when_the_experiment_is_created(lazy_config, monkeypatch):
    lazy_config["parameters"]["lazy_load_models"] = False
    loaded = []
    llm = RecordingLLM(responses=["x"])

    def load_model(self):
        loaded.append(self.name_or_path)
        return llm

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)

    experiment = rup.experiment_from_dict(lazy_config)

    assert loaded == ["org/model"] and experiment.runnable_models == {"lazy": llm}
    assert isinstance(experiment.models["lazy"], LocalHuggingFaceModelConfig)  # the config is kept

    experiment.run(show_progress=False)
    experiment.run(show_progress=False)

    assert len(llm.log) == 16
    assert loaded == ["org/model"]  # an eagerly loaded model is loaded once and kept


def test_eager_loading_reports_a_model_that_cannot_be_loaded(lazy_config, monkeypatch):
    """The documentation used to promise that such a model is skipped with a printed message."""
    lazy_config["parameters"]["lazy_load_models"] = False
    lazy_config["models"]["broken"] = {"type": "local_huggingface", "name_or_path": "org/broken"}

    def load_model(self):
        if self.name_or_path == "org/broken":
            raise ValueError("no such model")
        return RecordingLLM(responses=["x"])

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)

    with pytest.raises(ValueError, match="no such model"):
        rup.experiment_from_dict(lazy_config)


def test_error_while_loading_a_model_stops_the_run_but_keeps_earlier_answers(
    config_dict, monkeypatch
):
    """Only failing model *calls* are tolerated; a model that cannot be loaded is raised."""

    def load_model(self):
        raise ValueError("cannot load")

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    good = RecordingLLM(responses=["ok"])
    experiment = make_experiment(config_dict, {"good": good})
    experiment.models["bad"] = LocalHuggingFaceModelConfig(name_or_path="org/bad")
    experiment.runnable_models["bad"] = experiment.models["bad"]

    with pytest.raises(ValueError, match="cannot load"):
        experiment.run(show_progress=False)

    assert len(experiment.get_answers_as_dataframe()) == 8  # the first model's answers survive
    assert experiment.runnable_models["good"] is good
    assert experiment.runnable_models["bad"] is experiment.models["bad"]


def test_unloadable_langchain_definition_fails_the_run_with_the_reason(config_dict):
    config = copy.deepcopy(config_dict)
    # dumpd() cannot serialise a fake LLM: the definition is a "not_implemented" stub
    config["models"] = {
        "broken": {"type": "langchain", "definition": dumpd(FakeListLLM(responses=["x"]))}
    }
    experiment = rup.experiment_from_dict(config)

    with pytest.raises(ValueError, match="Failed to load the LangChain model: .*implement"):
        experiment.run(show_progress=False)


def test_load_model_only_loads_non_runnables(fake_experiment):
    llm = FakeListLLM(responses=["x"])
    assert fake_experiment._is_runnable(llm) is True
    assert fake_experiment._load_model(llm) is llm

    config = LangChainModelConfig(definition={})
    assert fake_experiment._is_runnable(config) is False
    assert fake_experiment._is_runnable(None) is False


# ------------------------------------------------------------------------------------------
# Failing model calls
# ------------------------------------------------------------------------------------------


def test_failing_calls_give_no_answer_and_the_run_continues(config_dict, caplog):
    llm = RecordingLLM(responses=numbered(), failing_calls={1, 4})
    experiment = make_experiment(config_dict, {"m": llm})
    callback = RecordingCallback()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with pytest.warns(RuntimeWarning, match=r"2 of 8 model calls failed .*RuntimeError: boom"):
            summary = experiment.run(callbacks=[callback], show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    total = n_items * len(persona_ids)
    assert len(llm.log) == total  # every slot was tried: the run did not stop
    assert (summary.n_calls, summary.n_failed, summary.n_succeeded) == (total, 2, total - 2)
    assert summary.errors == ["RuntimeError: boom"]  # distinct messages
    answers = [c.answer for c in callback.calls]
    # the failing slots (calls 1 and 4) are reported to the callbacks as None ...
    assert answers[1] is None and answers[4] is None
    # ... the others got the answer of their call number ...
    assert [a for a in answers if a is not None] == [
        f"a{k}" for k in range(total) if k not in (1, 4)
    ]
    # ... and nothing is stored on the item for the failed ones
    df = experiment.get_answers_as_dataframe()
    assert len(df) == total - 2
    missing = {(callback.calls[i].item_id, callback.calls[i].profile_id) for i in (1, 4)}
    assert missing == {(0, persona_ids[1]), (2, persona_ids[0])}
    stored = {(row["Instruction ID"], row["Persona ID"]) for _, row in df.iterrows()}
    assert stored.isdisjoint(missing)
    # every failure is logged where it happened
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
    assert messages == [
        "Model call failed (model=m, persona=Conservative Persona, item=0, seed=7): "
        "RuntimeError: boom",
        "Model call failed (model=m, persona=Optimistic Persona, item=2, seed=7): "
        "RuntimeError: boom",
    ]


def test_failing_calls_still_report_the_elapsed_time(config_dict):
    llm = RecordingLLM(responses=["x"], failing_calls=set(range(100)))
    experiment = make_experiment(config_dict, {"m": llm})
    callback = RecordingCallback()

    summary = experiment.run(callbacks=[callback], on_error="ignore", show_progress=False)

    assert summary.n_failed == 8
    assert len(callback.calls) == 8
    assert all(c.answer is None for c in callback.calls)
    assert all(isinstance(c.time, float) and c.time >= 0 for c in callback.calls)
    assert len(experiment.get_answers_as_dataframe()) == 0


def test_failing_parser_gives_none(config_dict, caplog):
    def explode(text):
        raise ValueError(f"cannot parse {text}")

    experiment = make_experiment(config_dict, {"m": RecordingLLM(responses=["x"])})
    experiment.set_parser(RunnableLambda(explode))
    callback = RecordingCallback()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with pytest.warns(RuntimeWarning, match="8 of 8 model calls failed"):
            summary = experiment.run(callbacks=[callback], show_progress=False)

    assert [c.answer for c in callback.calls] == [None] * 8
    assert summary.errors == ["ValueError: cannot parse x"]
    assert "ValueError: cannot parse x" in caplog.text


def test_prompt_with_unknown_placeholder_fails_every_call(fake_experiment, caplog):
    fake_experiment.set_prompt(
        ChatPromptTemplate.from_messages([("user", "{question} {no_such_placeholder}")])
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with pytest.warns(RuntimeWarning, match="8 of 8 model calls failed") as record:
            summary = fake_experiment.run(show_progress=False)

    assert len(fake_experiment.get_answers_as_dataframe()) == 0
    assert caplog.text.count("Model call failed") == 8
    assert "no_such_placeholder" in caplog.text
    assert summary.errors[0].startswith("KeyError") and len(summary.errors) == 1
    assert sum("no_such_placeholder" in str(w.message) for w in record) == 1  # warned once


def test_the_raise_policy_stops_at_the_failing_call_and_keeps_the_answers_so_far(config_dict):
    llm = RecordingLLM(responses=numbered(), failing_calls={3: "stop here"})
    experiment = make_experiment(config_dict, {"m": llm})
    callback = RecordingCallback()

    with pytest.raises(RuntimeError, match="stop here"):
        experiment.run(callbacks=[callback], on_error="raise", show_progress=False)

    assert len(llm.log) == 4  # nothing was asked after the failure
    assert [c.answer for c in callback.calls] == ["a0", "a1", "a2"]  # the failure is not reported
    assert len(experiment.get_answers_as_dataframe()) == 3
    assert experiment.runnable_models["m"] is llm  # the model is still there for the next try


def test_summary_keeps_the_first_five_distinct_error_messages(config_dict):
    llm = RecordingLLM(failing_calls={k: f"fail {k}" for k in range(8)})
    experiment = make_experiment(config_dict, {"m": llm})

    summary = experiment.run(on_error="ignore", show_progress=False)

    assert summary.n_failed == 8
    assert summary.errors == [f"RuntimeError: fail {k}" for k in range(5)]


# ------------------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------------------


def test_callbacks_receive_the_documented_arguments(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["3", "4"])
    callback = RecordingCallback()

    experiment.run(callbacks=[callback], show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    assert len(callback.calls) == 2 * n_items * len(persona_ids)
    items = experiment.questionnaire.instruction_items
    for k, call in enumerate(callback.calls):
        assert call.experiment is experiment
        assert call.item is items[call.item_id]
        assert call.item.question == items[call.item_id].question
        assert call.model_id == "m"
        assert call.random_seed in ("3", "4")
        assert isinstance(call.time, float) and call.time >= 0
        assert call.answer == f"a{k}"
        # the answer is already stored on the item when the callback runs
        assert call.stored["m"][call.profile_id][call.random_seed] == call.answer
    assert [c.item_id for c in callback.calls[: n_items * len(persona_ids)]] == [
        i for i in range(n_items) for _ in persona_ids
    ]


def test_all_callbacks_are_called_for_every_answer_in_order(fake_experiment):
    first, second = RecordingCallback(), RecordingCallback()
    fake_experiment.run(callbacks=[first, second], show_progress=False)

    assert len(first.calls) == len(second.calls) == 8
    for one, two in zip(first.calls, second.calls):
        assert (one.item_id, one.model_id, one.profile_id, one.random_seed, one.answer) == (
            two.item_id,
            two.model_id,
            two.profile_id,
            two.random_seed,
            two.answer,
        )


def test_failing_callback_only_warns_and_other_callbacks_still_run(fake_experiment):
    after = RecordingCallback()

    with pytest.warns(UserWarning) as record:
        fake_experiment.run(callbacks=[ExplodingCallback(), after], show_progress=False)

    messages = [str(w.message) for w in record if "Error while saving answer" in str(w.message)]
    assert messages == ["Error while saving answer: callback boom"] * 8
    assert len(after.calls) == 8
    assert len(fake_experiment.get_answers_as_dataframe()) == 8  # the run itself is unaffected


def test_run_without_callbacks_by_default(fake_experiment):
    fake_experiment.run(show_progress=False)
    assert len(fake_experiment.get_answers_as_dataframe()) == 8


# ------------------------------------------------------------------------------------------
# Cumulative ("response memory") mode
# ------------------------------------------------------------------------------------------


def test_cumulative_prompts_carry_the_previous_answers_of_the_same_persona(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})
    questions = [item.question for item in experiment.questionnaire.instruction_items]
    n_personas = len(experiment.demographic_profiles)

    experiment.run(cumulative=True, show_progress=False)

    assert len(llm.log) == len(questions) * n_personas
    for item_id in range(len(questions)):
        for persona_no in range(n_personas):
            prompt = llm.log[item_id * n_personas + persona_no].prompt
            history = "".join(
                f"Question: {questions[earlier]}\nAnswer: a{earlier * n_personas + persona_no}\n"
                for earlier in range(item_id)
            )
            assert human_part(prompt) == history + f"Question: {questions[item_id]}\nAnswer:"


def test_cumulative_second_question_sees_the_first_answer_only_of_its_own_persona(config_dict):
    llm = RecordingLLM(responses=["first-p0", "first-p1", "second-p0", "second-p1"] + numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True, show_progress=False)

    prompts = [call.prompt for call in llm.log]
    q = [item.question for item in experiment.questionnaire.instruction_items]
    assert "first-p" not in prompts[0] and "first-p" not in prompts[1]  # nothing to remember yet
    assert f"Answer: first-p0\nQuestion: {q[1]}" in human_part(prompts[2])
    assert "first-p1" not in prompts[2]
    assert f"Answer: first-p1\nQuestion: {q[1]}" in human_part(prompts[3])
    assert "first-p0" not in prompts[3]
    assert f"Answer: second-p0\nQuestion: {q[2]}" in human_part(prompts[4])


def test_cumulative_system_prompt_stays_the_same_for_a_persona(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})
    experiment.run(cumulative=True, show_progress=False)

    systems = [call.prompt.split("\nHuman: ", 1)[0] for call in llm.log]
    n_personas = len(experiment.demographic_profiles)
    for persona_no in range(n_personas):
        assert len({systems[i] for i in range(persona_no, len(systems), n_personas)}) == 1
    assert systems[0] != systems[1]  # different persona descriptions
    assert "Ms Muller is 18 years old" in systems[0]
    assert "Mr Grueber is 65 years old" in systems[1]


def test_cumulative_memory_starts_again_for_every_seed(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm}, seeds=["1", "2"])

    experiment.run(cumulative=True, show_progress=False)

    n_items, persona_ids = grid_of(experiment)
    per_seed = n_items * len(persona_ids)
    first_question = experiment.questionnaire.instruction_items[0].question
    for seed_no in range(2):
        for persona_no in range(len(persona_ids)):
            call = llm.log[seed_no * per_seed + persona_no]
            assert call.seed == seed_no + 1
            assert human_part(call.prompt) == f"Question: {first_question}\nAnswer:"


def test_cumulative_mode_builds_a_chain_for_every_call(config_dict, monkeypatch):
    seen = []
    original = ExperimentProcessingMixin._get_chain

    def spy(self, *args, **kwargs):
        seen.append(kwargs["seed"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentProcessingMixin, "_get_chain", spy)
    experiment = make_experiment(
        cumulative_config(config_dict), {"m": RecordingLLM(responses=["x"])}
    )
    experiment.run(cumulative=True, show_progress=False)

    assert len(seen) == 8  # the prompt changes with every answer


def test_cumulative_mode_reports_the_same_callbacks_as_the_normal_mode(config_dict):
    normal, cumulative = RecordingCallback(), RecordingCallback()
    make_experiment(cumulative_config(config_dict), {"m": RecordingLLM(responses=numbered())}).run(
        callbacks=[normal], show_progress=False
    )
    make_experiment(cumulative_config(config_dict), {"m": RecordingLLM(responses=numbered())}).run(
        cumulative=True, callbacks=[cumulative], show_progress=False
    )

    def key(c):
        return (c.item_id, c.model_id, c.profile_id, c.random_seed, c.answer)

    assert [key(c) for c in normal.calls] == [key(c) for c in cumulative.calls]


def test_cumulative_run_survives_a_failing_call(config_dict):
    llm = RecordingLLM(responses=numbered(), failing_calls={0})
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    summary = experiment.run(cumulative=True, on_error="ignore", show_progress=False)

    assert len(llm.log) == 8
    assert (summary.n_calls, summary.n_failed) == (8, 1)
    assert len(experiment.get_answers_as_dataframe()) == 7


def test_cumulative_memory_does_not_contain_the_text_none_for_a_failed_call(config_dict):
    llm = RecordingLLM(responses=numbered(), failing_calls={0})
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True, on_error="ignore", show_progress=False)

    # call 0 (item 0, first persona) failed, call 2 asks item 1 of the same persona
    assert "None" not in human_part(llm.log[2].prompt)
    assert human_part(llm.log[2].prompt).count("Question:") == 1  # nothing was remembered


def test_cumulative_mode_handles_answers_with_curly_braces(config_dict):
    answer = '{answer: "3"}'  # the answer format the BFI prompts ask for
    llm = RecordingLLM(responses=[answer])
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    summary = experiment.run(cumulative=True, show_progress=False)

    assert summary.n_failed == 0 and len(llm.log) == 8
    assert f"Answer: {answer}\nQuestion:" in human_part(llm.log[2].prompt)
    assert len(experiment.get_answers_as_dataframe()) == 8


def test_cumulative_mode_handles_curly_braces_in_the_questions(config_dict):
    config = cumulative_config(config_dict)
    config["questionnaire"]["instruction_items"][0]["question"] = "Do you like {curly} braces?"
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config, {"m": llm})

    summary = experiment.run(cumulative=True, show_progress=False)

    assert summary.n_failed == 0
    assert "Question: Do you like {curly} braces?\nAnswer: a0\n" in human_part(llm.log[2].prompt)


def test_cumulative_mode_keeps_the_messages_after_the_user_message(config_dict):
    config = cumulative_config(config_dict)
    config["prompt_template"]["messages"].append({"role": "assistant", "content": "My choice:"})
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config, {"m": llm})
    q = [item.question for item in experiment.questionnaire.instruction_items]

    experiment.run(cumulative=True, show_progress=False)

    assert all(call.prompt.endswith("\nAI: My choice:") for call in llm.log)
    # the memory sits in the user message, in front of the live question and before the rest
    assert human_part(llm.log[2].prompt) == (
        f"Question: {q[0]}\nAnswer: a0\nQuestion: {q[1]}\nAnswer:\nAI: My choice:"
    )


def test_cumulative_mode_remembers_questions_with_the_answer_options_they_were_asked_with(
    config_dict,
):
    config = copy.deepcopy(config_dict)  # its user message contains {answer_options}
    items = config["questionnaire"]["instruction_items"]
    items[0]["answer_options"] = {"a": {"text": "yes"}, "b": {"text": "no"}}
    items[1]["answer_options"] = {"x": {"text": "up"}, "y": {"text": "down"}}
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config, {"m": llm})
    q = [item.question for item in experiment.questionnaire.instruction_items]

    experiment.run(cumulative=True, show_progress=False)

    # item 1 of the first persona (call 2): the first question as it was asked, then the live one
    assert human_part(llm.log[2].prompt) == (
        f"Question: {q[0]}\nAnswer Options: yes, no\nAnswer: a0\n"
        f"Question: {q[1]}\nAnswer Options: up, down\nAnswer:"
    )


def test_cumulative_mode_gives_the_same_results_when_the_personas_run_in_parallel(config_dict):
    def run(workers):
        experiment = make_experiment(
            cumulative_config(config_dict), {"m": LengthLLM()}, seeds=["1", "2"]
        )
        callback = RecordingCallback()
        experiment.run(
            cumulative=True, callbacks=[callback], max_concurrency=workers, show_progress=False
        )
        return [c.answer for c in callback.calls], experiment.get_answers_as_dataframe()

    sequential_answers, sequential_df = run(1)
    parallel_answers, parallel_df = run(4)

    assert parallel_answers == sequential_answers
    assert parallel_df.equals(sequential_df)
    # the answer is the length of the prompt, so it grows with the memory: item by item, for
    # every persona and seed (calls come item by item, two personas each, 8 calls per seed)
    lengths = [int(answer.removeprefix("len")) for answer in sequential_answers]
    for seed_no in range(2):
        for persona_no in range(2):
            per_item = lengths[seed_no * 8 + persona_no : seed_no * 8 + 8 : 2]
            assert len(per_item) == 4 and per_item == sorted(set(per_item))


def test_cumulative_mode_needs_a_system_and_a_user_message(config_dict):
    config = copy.deepcopy(config_dict)
    config["prompt_template"] = {
        "type": "chat",
        "messages": [{"role": "user", "content": "{question} {answer_options}"}],
    }
    experiment = make_experiment(config, {"m": RecordingLLM()})

    with pytest.raises(ValueError, match="system message followed by a user message"):
        experiment.run(cumulative=True, show_progress=False)


# ------------------------------------------------------------------------------------------
# _ensure_requirements_to_run
# ------------------------------------------------------------------------------------------


def _no_models(experiment):
    experiment.clear_models()


def _no_prompt(experiment):
    experiment.reset_prompt()


def _no_questionnaire(experiment):
    experiment.questionnaire = None


def _no_profiles(experiment):
    experiment.demographic_profiles = None


def test_questionnaire_without_items_fails_with_a_clear_error(config_dict):
    """A misspelt 'instruction_items' key leaves the questionnaire empty."""
    config = copy.deepcopy(config_dict)
    config["questionnaire"]["instruction_item"] = config["questionnaire"].pop("instruction_items")
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["x"]), identifier="m")

    assert experiment.questionnaire.instruction_items is None
    with pytest.raises(ValueError, match="no instruction_items"):
        experiment.run(show_progress=False)


def test_a_persona_template_with_an_unknown_placeholder_fails_before_any_call(config_dict):
    """docs/configuration.md: the prompts are built up front, so no model call is wasted."""
    config = copy.deepcopy(config_dict)
    config["demographic_profiles"]["Broken"] = {
        "attributes": {"name": "X"},
        "template": "{name} {nowhere}",
    }
    llm = RecordingLLM()
    experiment = make_experiment(config, {"m": llm})

    with pytest.raises(ValueError, match="attribute 'nowhere'"):
        experiment.run(show_progress=False)

    assert llm.log == [] and experiment.get_answers() == [{}, {}, {}, {}]


@pytest.mark.parametrize(
    "break_experiment, message",
    [
        (_no_models, "No models have been set in runnable_models."),
        (_no_prompt, "runnable_prompt has not been set."),
        (_no_questionnaire, "questionnaire has not been set."),
        (_no_profiles, "demographic_profiles has not been set."),
    ],
    ids=["models", "prompt", "questionnaire", "profiles"],
)
def test_missing_requirements_give_a_clear_error(fake_experiment, break_experiment, message):
    assert fake_experiment._ensure_requirements_to_run() is True

    break_experiment(fake_experiment)

    with pytest.raises(ValueError) as direct:
        fake_experiment._ensure_requirements_to_run()
    assert str(direct.value) == message
    with pytest.raises(ValueError) as from_run:
        fake_experiment.run(show_progress=False)
    assert str(from_run.value) == message
    with pytest.raises(ValueError) as from_process:
        fake_experiment.process_single_experiment(cumulative=False)
    assert str(from_process.value) == message


def test_models_are_checked_before_the_other_requirements(fake_experiment):
    fake_experiment.clear_models()
    fake_experiment.reset_prompt()
    fake_experiment.questionnaire = None

    with pytest.raises(ValueError, match="No models have been set"):
        fake_experiment._ensure_requirements_to_run()


def test_a_failed_check_leaves_the_experiment_untouched(fake_experiment):
    fake_experiment.reset_prompt()

    with pytest.raises(ValueError, match="runnable_prompt has not been set."):
        fake_experiment.run(show_progress=False)

    assert fake_experiment.list_models() == ["fake"]
    assert fake_experiment.get_model("fake") is not None
    assert fake_experiment.get_answers() == [{}, {}, {}, {}]


# ------------------------------------------------------------------------------------------
# assemble_prompt / print_assembled_prompt
# ------------------------------------------------------------------------------------------


def test_assemble_prompt_is_exactly_what_the_model_receives(config_dict):
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm})
    n_items, persona_ids = grid_of(experiment)

    experiment.run(show_progress=False)

    for item_idx in range(n_items):
        for persona_idx in range(len(persona_ids)):
            sent = llm.log[item_idx * len(persona_ids) + persona_idx].prompt
            assert experiment.assemble_prompt(item_idx, persona_idx) == sent


def test_assemble_prompt_defaults_to_the_first_item_and_persona(fake_experiment):
    prompt = fake_experiment.assemble_prompt()

    assert prompt == fake_experiment.assemble_prompt(item_idx=0, persona_idx=0)
    assert "I see myself as someone who..." in prompt and "Ms Muller is 18 years old" in prompt
    assert prompt.startswith("System: ") and "\nHuman: " in prompt


def test_assemble_prompt_works_for_a_prompt_that_is_no_chat_prompt(config_dict):
    config = copy.deepcopy(config_dict)
    config["prompt_template"] = {"type": "normal", "template": "{question} -> {answer_options}"}
    experiment = rup.experiment_from_dict(config)

    assert experiment.assemble_prompt(1, 1).startswith("Tends to find fault with others -> 1. ")


@pytest.mark.parametrize("item_idx, persona_idx", [(99, 0), (0, 99), (-99, 0)])
def test_assemble_prompt_raises_an_index_error_for_a_missing_item_or_persona(
    fake_experiment, item_idx, persona_idx
):
    with pytest.raises(IndexError):
        fake_experiment.assemble_prompt(item_idx, persona_idx)


@pytest.mark.parametrize(
    "break_experiment", [_no_prompt, _no_questionnaire], ids=["prompt", "items"]
)
def test_assemble_prompt_needs_a_questionnaire_and_a_prompt(fake_experiment, break_experiment):
    break_experiment(fake_experiment)

    with pytest.raises(ValueError, match="needs a questionnaire and a prompt template"):
        fake_experiment.assemble_prompt()


def test_print_assembled_prompt_shows_exactly_what_the_model_receives(config_dict, capsys):
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm})
    experiment.print_assembled_prompt(item_idx=1, persona_idx=1)
    out = capsys.readouterr().out

    header = "\n++++++++++++++++++ assembled prompt (item 1, persona 1) ++++++++++++++++++\n"
    footer = "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    assert out.startswith(header)
    assert out.endswith(footer)
    assembled = out[len(header) : -len(footer)]

    experiment.run(show_progress=False)
    n_personas = len(experiment.demographic_profiles)
    assert assembled == llm.log[1 * n_personas + 1].prompt == experiment.assemble_prompt(1, 1)


def test_print_assembled_prompt_defaults_to_the_first_item_and_persona(fake_experiment, capsys):
    fake_experiment.print_assembled_prompt()
    out = capsys.readouterr().out

    assert "assembled prompt (item 0, persona 0)" in out
    assert "I see myself as someone who..." in out
    assert '{answer: "answer option"}' in out  # the doubled braces of the template are literal
    assert "Ms Muller is 18 years old" in out
    assert "1. Disagree strongly, 2. Disagree a little" in out


@pytest.mark.parametrize(
    "item_idx, persona_idx, persona_text, question",
    [
        (0, 1, "Mr Grueber is 65 years old", "I see myself as someone who..."),
        (3, 0, "Ms Muller is 18 years old", "Is depressed, blue"),
    ],
)
def test_print_assembled_prompt_selects_item_and_persona(
    fake_experiment, capsys, item_idx, persona_idx, persona_text, question
):
    fake_experiment.print_assembled_prompt(item_idx=item_idx, persona_idx=persona_idx)
    out = capsys.readouterr().out

    assert f"(item {item_idx}, persona {persona_idx})" in out
    assert persona_text in out
    assert question in out


@pytest.mark.parametrize("item_idx, persona_idx", [(99, 0), (0, 99)])
def test_print_assembled_prompt_warns_instead_of_raising(
    fake_experiment, capsys, item_idx, persona_idx
):
    with pytest.warns(UserWarning, match="Failed to assemble prompt: list index out of range"):
        fake_experiment.print_assembled_prompt(item_idx, persona_idx)
    assert capsys.readouterr().out == ""


def test_print_assembled_prompt_warns_without_answer_options(fake_experiment, capsys):
    fake_experiment.questionnaire.default_answer_options = None

    with pytest.warns(UserWarning, match="Failed to assemble prompt"):
        fake_experiment.print_assembled_prompt()
    assert capsys.readouterr().out == ""


def test_print_assembled_prompt_warns_without_a_prompt(fake_experiment, capsys):
    fake_experiment.reset_prompt()

    with pytest.warns(UserWarning, match="Failed to assemble prompt: .*needs a questionnaire"):
        fake_experiment.print_assembled_prompt()
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------------------


def test_create_input_dict_uses_the_default_answer_options(fake_experiment):
    profile = fake_experiment.demographic_profiles["Optimistic Persona"]
    item = fake_experiment.questionnaire.instruction_items[0]

    values = fake_experiment._create_input_dict(profile, item)

    assert values == {
        "general_instruction": fake_experiment.questionnaire.general_instruction,
        "persona_description": (
            "Ms Muller is 18 years old and very open minded with a optimistic personality."
        ),
        "question": "I see myself as someone who...",
        "answer_options": (
            "1. Disagree strongly, 2. Disagree a little, 3. Neither agree nor disagree, "
            "4. Agree a little, 5. Agree strongly"
        ),
    }


def test_create_input_dict_prefers_the_item_answer_options(fake_experiment):
    profile = fake_experiment.demographic_profiles["Optimistic Persona"]
    item = InstructionItem(
        question="own options?",
        answer_options={
            "options": {"1": {"text": "yes"}, "2": {"text": "no"}},
            "delimiter": " | ",
        },
    )

    values = fake_experiment._create_input_dict(profile, item)

    assert values["answer_options"] == "yes | no"
    assert values["question"] == "own options?"


@pytest.mark.parametrize("empty", [{}, {"options": {}}], ids=["flat", "wrapped"])
def test_create_input_dict_uses_the_defaults_for_an_item_with_empty_answer_options(
    fake_experiment, empty
):
    profile = fake_experiment.demographic_profiles["Optimistic Persona"]
    plain = InstructionItem(question="q")
    empty_options = InstructionItem(question="q", answer_options=empty)

    assert (
        fake_experiment._create_input_dict(profile, empty_options)["answer_options"]
        == fake_experiment._create_input_dict(profile, plain)["answer_options"]
        != ""
    )


def test_an_item_with_empty_answer_options_is_asked_with_the_default_options(config_dict):
    config = copy.deepcopy(config_dict)
    config["questionnaire"]["instruction_items"][1]["answer_options"] = {}
    llm = RecordingLLM()
    experiment = make_experiment(config, {"m": llm})

    experiment.run(show_progress=False)

    default_options = experiment.questionnaire.default_answer_options.join_options()
    assert f"Answer Options: {default_options}\nAnswer:" in llm.log[2].prompt  # item 1, persona 0


def test_create_input_dict_without_any_answer_options_is_the_documented_attribute_error(
    fake_experiment,
):
    """docs/tutorials/running-experiments.md troubleshooting table."""
    fake_experiment.questionnaire.default_answer_options = None
    profile = fake_experiment.demographic_profiles["Optimistic Persona"]
    item = fake_experiment.questionnaire.instruction_items[0]

    with pytest.raises(AttributeError, match="join_options"):
        fake_experiment._create_input_dict(profile, item)


def test_invoke_returns_the_answer_the_elapsed_time_and_no_error(fake_experiment):
    chain = RunnableLambda(lambda values: values["question"].upper())

    answer, elapsed, error = fake_experiment._invoke(chain, {"question": "hi"})

    assert answer == "HI" and error is None
    assert isinstance(elapsed, float) and elapsed >= 0
    assert elapsed == round(elapsed, 3)


def test_invoke_returns_the_error_instead_of_raising(fake_experiment):
    chain = RunnableLambda(lambda values: 1 / 0)

    answer, elapsed, error = fake_experiment._invoke(chain, {})

    assert answer is None and elapsed >= 0
    assert isinstance(error, ZeroDivisionError)


def test_generate_answer_returns_the_answer_and_the_elapsed_time(fake_experiment):
    chain = RunnableLambda(lambda values: values["question"].upper())

    answer, elapsed = fake_experiment._generate_answer(chain, {"question": "hi"})

    assert answer == "HI"
    assert isinstance(elapsed, float) and elapsed >= 0
    assert elapsed == round(elapsed, 3)


def test_generate_answer_returns_none_and_logs_when_the_chain_fails(fake_experiment, caplog):
    chain = RunnableLambda(lambda values: 1 / 0)

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        answer, elapsed = fake_experiment._generate_answer(chain, {})

    assert answer is None
    assert elapsed >= 0
    assert "Error invoking chain for run: division by zero" in caplog.text


@pytest.mark.parametrize(
    "seeds, expected",
    [(["1", "2"], ["1", "2"]), ([], ["42"]), (None, ["42"])],
)
def test_get_seed_values(config_dict, seeds, expected):
    experiment = make_experiment(config_dict)
    experiment.parameters.seeds = seeds
    assert [str(seed) for seed in experiment._get_seed_values()] == expected


def test_total_iterations_multiplies_items_models_personas_and_seeds(config_dict):
    models = {mid: FakeListLLM(responses=["x"]) for mid in ("a", "b", "c")}
    experiment = make_experiment(config_dict, models, seeds=["1", "2"])
    assert experiment._calculate_total_iterations() == 4 * 3 * 2 * 2


# ------------------------------------------------------------------------------------------
# _get_chain
# ------------------------------------------------------------------------------------------


def test_get_chain_requires_a_prompt_and_a_model(fake_experiment):
    llm = FakeListLLM(responses=["x"])
    with pytest.raises(ValueError, match="Prompt template not set."):
        fake_experiment._get_chain(None, llm, None, "name")
    with pytest.raises(ValueError, match="Model not set."):
        fake_experiment._get_chain(fake_experiment.runnable_prompt, None, None, "name")


def test_get_chain_seeds_the_model_and_names_the_run(fake_experiment):
    llm = RecordingLLM()

    chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, llm, None, "my-run", seed=5)

    assert isinstance(chain, RunnableBinding)
    assert chain.config == {"run_name": "my-run"}
    steps = chain.bound.steps
    assert [type(step).__name__ for step in steps] == [
        "RunnablePassthrough",
        "ChatPromptTemplate",
        "RecordingLLM",
        "StrOutputParser",
    ]
    assert steps[2].seed == 5 and steps[2] is not llm and llm.seed is None


def test_get_chain_uses_the_default_seed_and_the_given_parser(fake_experiment):
    parser = StrOutputParser()

    chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, RecordingLLM(), parser, "n")

    assert chain.bound.steps[2].seed == DEFAULT_SEED
    assert chain.bound.steps[3] is parser


@pytest.mark.parametrize("empty_parser", [None, {}], ids=["none", "empty-dict"])
def test_get_chain_falls_back_to_a_string_parser(fake_experiment, empty_parser):
    """The experiment's ``runnable_parser`` defaults to ``{}``, which must also mean 'no parser'."""
    chain = fake_experiment._get_chain(
        fake_experiment.runnable_prompt, FakeListLLM(responses=["x"]), empty_parser, "n"
    )
    assert isinstance(chain.bound.steps[3], StrOutputParser)


def test_get_chain_leaves_models_that_cannot_be_seeded_untouched(fake_experiment, monkeypatch):
    monkeypatch.setattr(seeding, "_WARNED", set())
    llm = FakeListLLM(responses=["x"])

    with pytest.warns(UserWarning, match="Models of type FakeListLLM cannot be seeded"):
        chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, llm, None, "n", seed=9)

    assert chain.bound.steps[2] is llm


def test_get_chain_runs_prompt_model_and_parser_in_order(fake_experiment):
    llm = RecordingLLM(responses=["raw answer"])
    chain = fake_experiment._get_chain(
        fake_experiment.runnable_prompt,
        llm,
        RunnableLambda(lambda text: f"<{text}>"),
        "n",
        seed=3,
    )
    values = fake_experiment._create_input_dict(
        fake_experiment.demographic_profiles["Optimistic Persona"],
        fake_experiment.questionnaire.instruction_items[0],
    )

    assert chain.invoke(values) == "<raw answer>"
    (call,) = llm.log
    assert "I see myself as someone who..." in call.prompt
    assert call.seed == 3 and call.kwargs == {}


def test_default_parser_is_empty_and_answers_are_plain_strings(fake_experiment):
    assert fake_experiment.runnable_parser == {}
    fake_experiment.run(show_progress=False)
    assert {type(a) for a in fake_experiment.get_answers_as_dataframe()["Answer"]} == {str}


def test_set_parser_transforms_every_answer(fake_experiment):
    fake_experiment.set_parser(RunnableLambda(lambda text: f"<{text}>"))
    fake_experiment.run(show_progress=False)
    assert set(fake_experiment.get_answers_as_dataframe()["Answer"]) == {"<3>"}


# ------------------------------------------------------------------------------------------
# Memory management
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cuda_available", [True, False])
def test_cleanup_memory_only_empties_the_cuda_cache_when_cuda_is_available(
    fake_experiment, monkeypatch, cuda_available
):
    emptied = []
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available, empty_cache=lambda: emptied.append(True)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    fake_experiment._cleanup_memory()

    assert emptied == ([True] if cuda_available else [])


def test_cleanup_memory_does_not_import_pytorch(fake_experiment, monkeypatch):
    """Cleaning up must work (and be cheap) when PyTorch is not installed or not in use."""
    monkeypatch.setitem(sys.modules, "torch", None)  # any "import torch" would now fail

    fake_experiment._cleanup_memory()  # no ImportError


def lazy_models(config_dict, monkeypatch, *names, failing_calls=()):
    """An experiment whose models are lazy configurations; loading one returns a fake LLM."""
    loaded = []

    def load_model(self):
        loaded.append(self.name_or_path)
        return RecordingLLM(tag=self.name_or_path, failing_calls=failing_calls)

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    config = copy.deepcopy(config_dict)
    config["models"] = {name: {"type": "local_huggingface", "name_or_path": name} for name in names}
    return rup.experiment_from_dict(config), loaded


def test_memory_is_cleaned_after_every_model_that_was_loaded_for_the_run(config_dict, monkeypatch):
    cleaned = []
    monkeypatch.setattr(
        ExperimentProcessingMixin, "_cleanup_memory", lambda self: cleaned.append(1)
    )
    experiment, loaded = lazy_models(config_dict, monkeypatch, "m1", "m2", "m3")

    experiment.run(show_progress=False)

    assert loaded == ["m1", "m2", "m3"]
    assert cleaned == [1, 1, 1]


def test_models_handed_in_by_the_user_are_not_garbage_collected(config_dict, monkeypatch):
    """Only what the run loaded is released and cleaned up; the user still holds the others."""
    cleaned = []
    monkeypatch.setattr(
        ExperimentProcessingMixin, "_cleanup_memory", lambda self: cleaned.append(1)
    )
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2")}
    experiment = make_experiment(config_dict, models)

    experiment.run(show_progress=False)

    assert cleaned == []
    assert all(experiment.get_model(mid) is models[mid] for mid in models)


def test_memory_is_cleaned_and_the_definition_kept_when_the_run_fails(config_dict, monkeypatch):
    cleaned = []
    monkeypatch.setattr(
        ExperimentProcessingMixin, "_cleanup_memory", lambda self: cleaned.append(1)
    )
    experiment, _ = lazy_models(config_dict, monkeypatch, "m1", failing_calls={0: "boom"})

    with pytest.raises(RuntimeError, match="boom"):
        experiment.run(on_error="raise", show_progress=False)

    assert cleaned == [1]
    assert experiment.runnable_models["m1"] is experiment.models["m1"]


def test_a_live_model_stays_in_the_experiment_when_the_run_fails(config_dict):
    llm = RecordingLLM(failing_calls={0: "boom"})
    experiment = make_experiment(config_dict, {"m": llm})

    with pytest.raises(RuntimeError, match="boom"):
        experiment.run(on_error="raise", show_progress=False)

    assert experiment.runnable_models["m"] is llm


# ------------------------------------------------------------------------------------------
# Optional back-ends are needed when a model is loaded - not before
# ------------------------------------------------------------------------------------------

BACKEND_MODULES = [
    "torch",
    "transformers",
    "scipy",
    "langchain_huggingface",
    "langchain_openai",
    "langchain_ollama",
    "langchain_google_genai",
    "langchain_deepseek",
    "huggingface_hub",
    "openai",
    "IPython",
]

BACKEND_FREE_FLOW = """
import sys

for name in {backends!r}:
    sys.modules[name] = None  # "import name" now fails, even where the package is installed

import rupsycho as rup
from langchain_core.language_models.fake import FakeListLLM

config = rup.load_example_config("bfi")
config["models"] = {{
    "local": {{"type": "local_huggingface", "name_or_path": "org/model"}},
    "remote": {{"type": "remote_huggingface", "repo_id": "org/m", "task": "text-generation"}},
    "ollama": {{"type": "ollama", "model": "llama3"}},
    "gpt": {{"type": "openai", "name_or_path": "gpt-4o-mini", "api_key": "key"}},
    "gemini": {{"type": "google", "name_or_path": "gemini-2.0-flash"}},
    "deepseek": {{"type": "deepseek", "name_or_path": "deepseek-chat"}},
}}
experiment = rup.experiment_from_dict(config)  # configured, not loaded
experiment.clear_models()
experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
summary = experiment.run(show_progress=False)  # a model of your own needs no back-end
assert summary.n_failed == 0, summary
experiment.export_to_file(sys.argv[1])
print("done")
"""


def test_configuring_running_and_exporting_an_experiment_needs_no_model_backend(tmp_path):
    """Everything but loading a configured model works with the optional extras not installed.

    (This is what the "core" CI job checks by installing nothing; here the back-ends are
    blocked, so the guarantee also holds on a developer machine that has them.)
    """
    code = textwrap.dedent(BACKEND_FREE_FLOW).format(backends=BACKEND_MODULES)

    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code, str(tmp_path / "out.json")],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "done"
    assert (tmp_path / "out.json").is_file()


@pytest.mark.parametrize(
    "config, module, extra",
    [
        ({"type": "local_huggingface", "name_or_path": "org/model"}, "transformers", "huggingface"),
        ({"type": "openai", "name_or_path": "gpt-4o-mini"}, "langchain_openai", "openai"),
    ],
    ids=["local_huggingface", "openai"],
)
def test_running_a_configured_model_without_its_extra_says_what_to_install(
    config_dict, monkeypatch, config, module, extra
):
    """The failure is a clear ImportError when the run reaches the model, not at creation."""
    monkeypatch.setitem(sys.modules, module, None)  # "import <module>" fails
    experiment_config = {**copy.deepcopy(config_dict), "models": {"m": config}}
    experiment = rup.experiment_from_dict(experiment_config)  # creating it is fine

    with pytest.raises(ImportError, match=rf"pip install 'rupsycho\[{extra}\]'"):
        experiment.run(show_progress=False)

    assert experiment.get_answers() == [{}, {}, {}, {}]
