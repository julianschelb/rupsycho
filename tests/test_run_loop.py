"""Tests for the run loop in ``rupsycho.mixins.experiment_processing``.

Everything runs against fake LLMs: the recording models below count the chain invocations and
capture the prompts and keyword arguments (the seed) of every call.
"""

import copy
import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.load import dumpd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBinding, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

import rupsycho as rup
from rupsycho.callbacks import Callback
from rupsycho.mixins import experiment_processing
from rupsycho.mixins.experiment_processing import DEFAULT_SEED, ExperimentProcessingMixin
from rupsycho.models.model import LangChainModelConfig, LocalHuggingFaceModelConfig
from rupsycho.models.questionnaire import InstructionItem

LOGGER_NAME = "rupsycho.mixins.experiment_processing"


@pytest.fixture(autouse=True)
def _quiet_progress_bars(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------


def numbered(n: int = 300, prefix: str = "a") -> list[str]:
    """Distinct responses ``a0, a1, ...``: the k-th call of a model answers ``a<k>``."""
    return [f"{prefix}{i}" for i in range(n)]


class RecordingLLM(FakeListLLM):
    """A fake LLM that records ``(prompt, call kwargs)`` of every call it receives.

    The langchain chain binds ``seed=<int>`` to the model, so the recorded kwargs show which
    seed each call used. ``trace`` can be shared between models to see the global call order.
    """

    tag: str = ""
    log: Any = Field(default_factory=list)
    trace: Any = None
    failing_calls: Any = ()

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        call_number = len(self.log)
        self.log.append((prompt, kwargs))
        if self.trace is not None:
            self.trace.append((self.tag, kwargs.get("seed")))
        if call_number in self.failing_calls:
            raise RuntimeError("boom")
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


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
    """The BFI config with a user template that only contains ``{question}`` (see the docs)."""
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

    experiment.run(callbacks=[callback])

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
    experiment.process_single_experiment(cumulative=False, pbar=bar)

    assert [len(model.log) for model in models.values()] == [
        len(seeds) * n_items * len(persona_ids)
    ] * len(model_ids)
    assert sum(len(model.log) for model in models.values()) == expected
    assert bar.updates == expected


def test_run_creates_its_own_progress_bar(config_dict):
    experiment = make_experiment(config_dict, {"m": RecordingLLM(responses=["x"])})
    experiment.process_single_experiment(cumulative=False)  # pbar=None creates a tqdm bar
    assert len(experiment.get_answers_as_dataframe()) == 8


def test_multiple_models_store_answers_under_their_own_id(config_dict):
    models = {
        "first": RecordingLLM(responses=["one"]),
        "second": RecordingLLM(responses=["two"]),
    }
    experiment = make_experiment(config_dict, models)
    experiment.run()

    _, persona_ids = grid_of(experiment)
    for item in experiment.questionnaire.instruction_items:
        assert item.get_all_answers() == {
            "first": {persona_id: {"7": "one"} for persona_id in persona_ids},
            "second": {persona_id: {"7": "two"} for persona_id in persona_ids},
        }


def test_models_are_released_after_the_run(config_dict):
    models = {"m1": RecordingLLM(responses=["x"]), "m2": RecordingLLM(responses=["x"])}
    experiment = make_experiment(config_dict, models)
    experiment.run()

    assert experiment.runnable_models == {"m1": None, "m2": None}
    assert experiment.get_model("m1") is None
    # the configuration is kept
    assert experiment.list_models() == ["m1", "m2"]


def test_experiment_can_only_run_once_documented_limitation(config_dict):
    """docs/tutorials/running-experiments.md: a second run() fails because models are released."""
    experiment = make_experiment(config_dict, {"m": RecordingLLM(responses=["x"])})
    experiment.run()
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'load_model'"):
        experiment.run()


def test_input_grid_is_built_once_per_model_not_per_seed(config_dict, monkeypatch):
    calls = []
    original = ExperimentProcessingMixin._create_input_dict

    def spy(self, profile, item):
        calls.append(item.question)
        return original(self, profile, item)

    monkeypatch.setattr(ExperimentProcessingMixin, "_create_input_dict", spy)
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2")}
    experiment = make_experiment(config_dict, models, seeds=["1", "2", "3"])
    experiment.run()

    n_items, persona_ids = grid_of(experiment)
    assert len(calls) == 2 * n_items * len(persona_ids)  # per model, independent of the 3 seeds


def test_chain_is_built_once_per_model_and_seed(config_dict, monkeypatch):
    seen = []
    original = ExperimentProcessingMixin._get_chain

    def spy(self, *args, **kwargs):
        seen.append((args[3], kwargs["seed"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentProcessingMixin, "_get_chain", spy)
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2")}
    experiment = make_experiment(config_dict, models, seeds=["1", "2", "3"])
    experiment.run()

    assert seen == [(mid, seed) for mid in ("m1", "m2") for seed in (1, 2, 3)]


def test_experiment_without_personas_generates_nothing(config_dict):
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm})
    experiment.clear_personas()

    experiment.run()

    assert llm.log == []
    assert len(experiment.get_answers_as_dataframe()) == 0


# ------------------------------------------------------------------------------------------
# Seeds
# ------------------------------------------------------------------------------------------


def test_seed_is_bound_to_every_model_call(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["11", "22"])
    experiment.run()

    n_items, persona_ids = grid_of(experiment)
    per_seed = n_items * len(persona_ids)
    assert [kwargs for _, kwargs in llm.log] == [{"seed": 11}] * per_seed + [
        {"seed": 22}
    ] * per_seed


def test_answers_are_keyed_by_the_seed_string_from_the_configuration(config_dict):
    experiment = make_experiment(
        config_dict, {"m": RecordingLLM(responses=["x"])}, seeds=["7", "8"]
    )
    experiment.run()

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
    assert experiment._get_seed_values() == [DEFAULT_SEED] == [42]

    experiment.run()

    assert {kwargs["seed"] for _, kwargs in llm.log} == {42}
    persona_id = next(iter(experiment.demographic_profiles))
    assert experiment.questionnaire.instruction_items[0].get_answer("m", persona_id, 42) == "x"


def test_seed_that_is_not_an_integer_stops_the_run(config_dict):
    """Seeds must be convertible to int (docs/configuration.md), checked only at run time."""
    experiment = make_experiment(config_dict, {"m": RecordingLLM(responses=["x"])}, seeds=["abc"])
    with pytest.raises(ValueError, match="invalid literal for int"):
        experiment.run()


# ------------------------------------------------------------------------------------------
# Parameter merging and lazy loading of model configurations
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

    experiment.run()

    assert loaded == ["org/model"]
    assert len(llm.log) == 8
    assert experiment.get_answers()[0] == {
        "lazy": {pid: {"7": "lazy-answer"} for pid in experiment.demographic_profiles}
    }
    assert experiment.runnable_models == {"lazy": None}  # released after use


def test_model_parameters_are_merged_over_experiment_parameters(lazy_config, monkeypatch):
    monkeypatch.setattr(
        LocalHuggingFaceModelConfig, "load_model", lambda self: RecordingLLM(responses=["x"])
    )
    seen = []
    original = ExperimentProcessingMixin._get_chain

    def spy(self, *args, **kwargs):
        seen.append(kwargs["params"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentProcessingMixin, "_get_chain", spy)
    lazy_config["parameters"]["seeds"] = ["1", "2"]
    experiment = rup.experiment_from_dict(lazy_config)

    experiment.run()

    # experiment-level extras are the defaults, the model's own parameters win, seeds are excluded
    expected = {"lazy_load_models": True, "temperature": 0.1, "top_k": 3, "max_new_tokens": 5}
    assert seen == [expected, expected]


def test_models_without_configured_parameters_get_the_experiment_defaults(config_dict, monkeypatch):
    seen = []
    original = ExperimentProcessingMixin._get_chain

    def spy(self, *args, **kwargs):
        seen.append(kwargs["params"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentProcessingMixin, "_get_chain", spy)
    llm = RecordingLLM(responses=["x"])
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["1"], top_p=0.5)

    experiment.run()

    assert seen == [{"lazy_load_models": True, "top_p": 0.5}]
    # docs: extra experiment parameters are stored but not passed to the models
    assert {tuple(kwargs) for _, kwargs in llm.log} == {("seed",)}


def test_eager_loading_loads_models_at_construction_and_skips_failures(
    lazy_config, monkeypatch, capsys
):
    lazy_config["parameters"]["lazy_load_models"] = False
    lazy_config["models"]["broken"] = {"type": "local_huggingface", "name_or_path": "org/broken"}
    llm = RecordingLLM(responses=["x"])

    def load_model(self):
        if self.name_or_path == "org/broken":
            raise ValueError("no such model")
        return llm

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)

    experiment = rup.experiment_from_dict(lazy_config)

    assert experiment.runnable_models == {"lazy": llm}
    assert "Failed to load model: no such model" in capsys.readouterr().out
    assert experiment.list_models() == ["lazy", "broken"]  # the configuration is kept

    experiment.run()
    assert len(llm.log) == 8


def test_run_stops_when_every_eagerly_loaded_model_failed(lazy_config, monkeypatch):
    lazy_config["parameters"]["lazy_load_models"] = False

    def load_model(self):
        raise ValueError("no such model")

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    experiment = rup.experiment_from_dict(lazy_config)

    with pytest.raises(ValueError, match="No models have been set in runnable_models."):
        experiment.run()


def test_error_while_loading_a_model_stops_the_run_but_keeps_earlier_answers(
    config_dict, monkeypatch
):
    """Only failing model *calls* are tolerated; a model that cannot be loaded is raised."""

    def load_model(self):
        raise ValueError("cannot load")

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    experiment = make_experiment(config_dict, {"good": RecordingLLM(responses=["ok"])})
    experiment.models["bad"] = LocalHuggingFaceModelConfig(name_or_path="org/bad")
    experiment.runnable_models["bad"] = experiment.models["bad"]

    with pytest.raises(ValueError, match="cannot load"):
        experiment.run()

    assert len(experiment.get_answers_as_dataframe()) == 8  # the first model's answers survive


def test_unloadable_langchain_definition_raises_model_not_set(config_dict):
    config = copy.deepcopy(config_dict)
    # dumpd() cannot serialise a fake LLM: the definition is a "not_implemented" stub
    config["models"] = {
        "broken": {"type": "langchain", "definition": dumpd(FakeListLLM(responses=["x"]))}
    }
    experiment = rup.experiment_from_dict(config)

    with pytest.warns(UserWarning, match="Failed to load LangChain model"):
        with pytest.raises(ValueError, match="Model not set."):
            experiment.run()


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

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        experiment.run(callbacks=[callback])

    n_items, persona_ids = grid_of(experiment)
    total = n_items * len(persona_ids)
    assert len(llm.log) == total  # every slot was tried: the run did not stop
    answers = [c.answer for c in callback.calls]
    # the failing slots (calls 1 and 4) are reported to the callbacks as None ...
    assert answers[1] is None and answers[4] is None
    # ... the fake only advances its response index on success, so the others are a0, a1, ...
    assert [a for a in answers if a is not None] == numbered(total - 2)
    # ... and nothing is stored on the item for them
    df = experiment.get_answers_as_dataframe()
    assert len(df) == total - 2
    missing = {(callback.calls[i].item_id, callback.calls[i].profile_id) for i in (1, 4)}
    assert missing == {(0, persona_ids[1]), (2, persona_ids[0])}
    stored = {(row["Instruction ID"], row["Persona ID"]) for _, row in df.iterrows()}
    assert stored.isdisjoint(missing)
    # the error is logged, once per failing call
    errors = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["Error invoking chain for run: boom"] * 2


def test_failing_calls_still_report_the_elapsed_time(config_dict):
    llm = RecordingLLM(responses=["x"], failing_calls=set(range(100)))
    experiment = make_experiment(config_dict, {"m": llm})
    callback = RecordingCallback()

    experiment.run(callbacks=[callback])

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

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        experiment.run(callbacks=[callback])

    assert [c.answer for c in callback.calls] == [None] * 8
    assert "cannot parse x" in caplog.text


def test_prompt_with_unknown_placeholder_fails_every_call(fake_experiment, caplog):
    fake_experiment.set_prompt(
        ChatPromptTemplate.from_messages([("user", "{question} {no_such_placeholder}")])
    )

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        fake_experiment.run()

    assert len(fake_experiment.get_answers_as_dataframe()) == 0
    assert caplog.text.count("Error invoking chain for run") == 8
    assert "no_such_placeholder" in caplog.text


# ------------------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------------------


def test_callbacks_receive_the_documented_arguments(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(config_dict, {"m": llm}, seeds=["3", "4"])
    callback = RecordingCallback()

    experiment.run(callbacks=[callback])

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
    fake_experiment.run(callbacks=[first, second])

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
        fake_experiment.run(callbacks=[ExplodingCallback(), after])

    messages = [str(w.message) for w in record if "Error while saving answer" in str(w.message)]
    assert messages == ["Error while saving answer: callback boom"] * 8
    assert len(after.calls) == 8
    assert len(fake_experiment.get_answers_as_dataframe()) == 8  # the run itself is unaffected


def test_run_without_callbacks_by_default(fake_experiment):
    fake_experiment.run()
    assert len(fake_experiment.get_answers_as_dataframe()) == 8


# ------------------------------------------------------------------------------------------
# Cumulative ("response memory") mode
# ------------------------------------------------------------------------------------------


def test_cumulative_prompts_carry_the_previous_answers_of_the_same_persona(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})
    questions = [item.question for item in experiment.questionnaire.instruction_items]
    n_personas = len(experiment.demographic_profiles)

    experiment.run(cumulative=True)

    assert len(llm.log) == len(questions) * n_personas
    for item_id in range(len(questions)):
        for persona_no in range(n_personas):
            prompt = llm.log[item_id * n_personas + persona_no][0]
            history = "".join(
                f"Question: {questions[earlier]}\nAnswer: a{earlier * n_personas + persona_no}\n"
                for earlier in range(item_id)
            )
            assert human_part(prompt) == history + f"Question: {questions[item_id]}\nAnswer:"


def test_cumulative_second_question_sees_the_first_answer_only_of_its_own_persona(config_dict):
    llm = RecordingLLM(responses=["first-p0", "first-p1", "second-p0", "second-p1"] + numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True)

    prompts = [p for p, _ in llm.log]
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
    experiment.run(cumulative=True)

    systems = [prompt.split("\nHuman: ", 1)[0] for prompt, _ in llm.log]
    n_personas = len(experiment.demographic_profiles)
    for persona_no in range(n_personas):
        assert len({systems[i] for i in range(persona_no, len(systems), n_personas)}) == 1
    assert systems[0] != systems[1]  # different persona descriptions
    assert "Ms Muller is 18 years old" in systems[0]
    assert "Mr Grueber is 65 years old" in systems[1]


def test_cumulative_memory_starts_again_for_every_seed(config_dict):
    llm = RecordingLLM(responses=numbered())
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm}, seeds=["1", "2"])

    experiment.run(cumulative=True)

    n_items, persona_ids = grid_of(experiment)
    per_seed = n_items * len(persona_ids)
    first_question = experiment.questionnaire.instruction_items[0].question
    for seed_no in range(2):
        for persona_no in range(len(persona_ids)):
            prompt, kwargs = llm.log[seed_no * per_seed + persona_no]
            assert kwargs == {"seed": seed_no + 1}
            assert human_part(prompt) == f"Question: {first_question}\nAnswer:"


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
    experiment.run(cumulative=True)

    assert len(seen) == 8  # the prompt changes with every answer


def test_cumulative_mode_reports_the_same_callbacks_as_the_normal_mode(config_dict):
    normal, cumulative = RecordingCallback(), RecordingCallback()
    make_experiment(cumulative_config(config_dict), {"m": RecordingLLM(responses=numbered())}).run(
        callbacks=[normal]
    )
    make_experiment(cumulative_config(config_dict), {"m": RecordingLLM(responses=numbered())}).run(
        cumulative=True, callbacks=[cumulative]
    )

    def key(c):
        return (c.item_id, c.model_id, c.profile_id, c.random_seed, c.answer)

    assert [key(c) for c in normal.calls] == [key(c) for c in cumulative.calls]


def test_cumulative_run_survives_a_failing_call(config_dict):
    llm = RecordingLLM(responses=numbered(), failing_calls={0})
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True)

    assert len(llm.log) == 8
    assert len(experiment.get_answers_as_dataframe()) == 7


@pytest.mark.xfail(
    strict=True,
    reason="cumulative mode: a failed call (answer None) is written into the response memory as "
    "the literal text 'None' (str(answer)); the original code raised TypeError instead",
)
def test_cumulative_memory_does_not_contain_the_text_none_for_a_failed_call(config_dict):
    llm = RecordingLLM(responses=numbered(), failing_calls={0})
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True)

    # call 0 (item 0, first persona) failed, call 2 asks item 1 of the same persona
    assert "None" not in human_part(llm.log[2][0])


@pytest.mark.xfail(
    strict=True,
    reason="cumulative mode: a model answer containing braces such as {answer: '3'} breaks the "
    "response-memory template (KeyError from str.format) and aborts the whole run",
)
def test_cumulative_mode_handles_answers_with_curly_braces(config_dict):
    answer = '{answer: "3"}'  # the answer format the BFI prompts ask for
    llm = RecordingLLM(responses=[answer])
    experiment = make_experiment(cumulative_config(config_dict), {"m": llm})

    experiment.run(cumulative=True)

    assert len(llm.log) == 8
    assert f"Answer: {answer}\nQuestion:" in human_part(llm.log[2][0])
    assert len(experiment.get_answers_as_dataframe()) == 8


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


def test_questionnaire_without_items_fails_with_the_documented_type_error(config_dict):
    """docs troubleshooting: a misspelt 'instruction_items' key leaves the questionnaire empty."""
    config = copy.deepcopy(config_dict)
    config["questionnaire"]["instruction_item"] = config["questionnaire"].pop("instruction_items")
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["x"]), identifier="m")

    assert experiment.questionnaire.instruction_items is None
    with pytest.raises(TypeError, match="object of type 'NoneType' has no len"):
        experiment.run()


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
        fake_experiment.run()
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


def test_models_that_are_all_none_still_count_as_set(fake_experiment):
    """After a run the dict {id: None} is non-empty, so only the load step fails."""
    fake_experiment.runnable_models = {"fake": None}
    assert fake_experiment._ensure_requirements_to_run() is True


# ------------------------------------------------------------------------------------------
# print_assembled_prompt
# ------------------------------------------------------------------------------------------


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

    experiment.run()
    n_personas = len(experiment.demographic_profiles)
    assert assembled == llm.log[1 * n_personas + 1][0]


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


def test_create_input_dict_without_any_answer_options_is_the_documented_attribute_error(
    fake_experiment,
):
    """docs/tutorials/running-experiments.md troubleshooting table."""
    fake_experiment.questionnaire.default_answer_options = None
    profile = fake_experiment.demographic_profiles["Optimistic Persona"]
    item = fake_experiment.questionnaire.instruction_items[0]

    with pytest.raises(AttributeError, match="join_options"):
        fake_experiment._create_input_dict(profile, item)


def test_build_input_grid_is_item_major_persona_minor(fake_experiment):
    questionnaire = fake_experiment.questionnaire

    grid = fake_experiment._build_input_grid(questionnaire, fake_experiment.demographic_profiles)

    assert [(item_id, profile_id) for item_id, _, profile_id, _ in grid] == [
        (item_id, profile_id)
        for item_id in range(4)
        for profile_id in ("Optimistic Persona", "Conservative Persona")
    ]
    for item_id, item, profile_id, values in grid:
        assert item is questionnaire.instruction_items[item_id]
        assert values["question"] == item.question
        assert values["persona_description"] == str(
            fake_experiment.demographic_profiles[profile_id]
        )


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
    [(["1", "2"], ["1", "2"]), ([], [42]), (None, [42])],
)
def test_get_seed_values(config_dict, seeds, expected):
    experiment = make_experiment(config_dict)
    experiment.parameters.seeds = seeds
    assert experiment._get_seed_values() == expected


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


def test_get_chain_binds_the_seed_and_names_the_run(fake_experiment):
    llm = FakeListLLM(responses=["x"])

    chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, llm, None, "my-run", seed=5)

    assert isinstance(chain, RunnableBinding)
    assert chain.config == {"run_name": "my-run"}
    steps = chain.bound.steps
    assert [type(step).__name__ for step in steps] == [
        "RunnablePassthrough",
        "ChatPromptTemplate",
        "RunnableBinding",
        "StrOutputParser",
    ]
    assert steps[2].bound is llm
    assert steps[2].kwargs == {"seed": 5}


def test_get_chain_uses_the_default_seed_and_the_given_parser(fake_experiment):
    llm = FakeListLLM(responses=["x"])
    parser = StrOutputParser()

    chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, llm, parser, "n")

    assert chain.bound.steps[2].kwargs == {"seed": DEFAULT_SEED}
    assert chain.bound.steps[3] is parser


@pytest.mark.parametrize("empty_parser", [None, {}], ids=["none", "empty-dict"])
def test_get_chain_falls_back_to_a_string_parser(fake_experiment, empty_parser):
    """The experiment's ``runnable_parser`` defaults to ``{}``, which must also mean 'no parser'."""
    chain = fake_experiment._get_chain(
        fake_experiment.runnable_prompt, FakeListLLM(responses=["x"]), empty_parser, "n"
    )
    assert isinstance(chain.bound.steps[3], StrOutputParser)


def test_get_chain_does_not_bind_a_seed_to_google_models(fake_experiment):
    google = ChatGoogleGenerativeAI.model_construct()  # no client, no network

    chain = fake_experiment._get_chain(fake_experiment.runnable_prompt, google, None, "n", seed=9)

    assert chain.bound.steps[2] is google


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
    prompt, kwargs = llm.log[0]
    assert "I see myself as someone who..." in prompt
    assert kwargs == {"seed": 3}


def test_default_parser_is_empty_and_answers_are_plain_strings(fake_experiment):
    assert fake_experiment.runnable_parser == {}
    fake_experiment.run()
    assert {type(a) for a in fake_experiment.get_answers_as_dataframe()["Answer"]} == {str}


def test_set_parser_transforms_every_answer(fake_experiment):
    fake_experiment.set_parser(RunnableLambda(lambda text: f"<{text}>"))
    fake_experiment.run()
    assert set(fake_experiment.get_answers_as_dataframe()["Answer"]) == {"<3>"}


# ------------------------------------------------------------------------------------------
# Memory management
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cuda_available", [True, False])
def test_cleanup_memory_only_empties_the_cuda_cache_when_cuda_is_available(
    fake_experiment, monkeypatch, cuda_available
):
    emptied = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))

    fake_experiment._cleanup_memory()

    assert emptied == ([True] if cuda_available else [])


def test_memory_is_cleaned_after_every_model(config_dict, monkeypatch):
    cleaned = []
    monkeypatch.setattr(
        ExperimentProcessingMixin, "_cleanup_memory", lambda self: cleaned.append(1)
    )
    models = {mid: RecordingLLM(responses=["x"]) for mid in ("m1", "m2", "m3")}
    experiment = make_experiment(config_dict, models)

    experiment.run()

    assert cleaned == [1, 1, 1]


# ------------------------------------------------------------------------------------------
# Default model
# ------------------------------------------------------------------------------------------


def test_get_default_model_builds_the_small_flan_t5_pipeline(monkeypatch):
    seen = {}
    tokenizer, model = object(), object()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            seen["tokenizer"] = model_id
            return tokenizer

    class FakeSeq2Seq:
        @staticmethod
        def from_pretrained(model_id):
            seen["model"] = model_id
            return model

    def fake_pipeline(task, **kwargs):
        seen["pipeline"] = (task, kwargs)
        return "PIPELINE"

    def fake_wrapper(pipeline):
        seen["wrapped"] = pipeline
        return "WRAPPED"

    monkeypatch.setattr(experiment_processing, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(experiment_processing, "AutoModelForSeq2SeqLM", FakeSeq2Seq)
    monkeypatch.setattr(experiment_processing, "pipeline", fake_pipeline)
    monkeypatch.setattr(experiment_processing, "HuggingFacePipeline", fake_wrapper)

    assert experiment_processing.get_default_model() == "WRAPPED"

    assert seen["tokenizer"] == seen["model"] == "google/flan-t5-small"
    assert seen["pipeline"] == (
        "text2text-generation",
        {
            "model": model,
            "tokenizer": tokenizer,
            "min_new_tokens": 1,
            "max_new_tokens": 64,
            "temperature": 0.6,
            "do_sample": True,
        },
    )
    assert seen["wrapped"] == "PIPELINE"
