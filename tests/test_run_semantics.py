"""Semantics of ExperimentDocument.run: summary, error policy, concurrency, re-runs, memory."""

import copy
import threading
import time
import warnings
from typing import Any

import pytest
from langchain_core.language_models.llms import LLM
from pydantic import Field, PrivateAttr

import rupsycho as rup
from rupsycho.callbacks import Callback


class ScriptedLLM(LLM):
    """Deterministic fake model that records prompts and can fail or be slow."""

    answer: str = "3"
    delay: float = 0.0
    fail_on: str | None = None
    calls: list[Any] = Field(default_factory=list)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        with self._lock:
            self.calls.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError(f"cannot answer: {self.fail_on}")
        return self.answer.replace("<len>", str(len(prompt)))


def make_experiment(config_dict, model=None, **model_kwargs):
    experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))
    model = model or ScriptedLLM(**model_kwargs)
    experiment.add_model(model, identifier="scripted")
    return experiment, model


class Recorder(Callback):
    def __init__(self):
        self.rows = []

    def save_answer(self, experiment, item_id, item, model_id, profile_id, seed, time, answer):
        self.rows.append((item_id, profile_id, seed, answer))


class TestSummary:
    def test_counts_every_call(self, config_dict):
        experiment, _ = make_experiment(config_dict)
        summary = experiment.run(show_progress=False)

        expected = (
            len(experiment.questionnaire.instruction_items)
            * len(experiment.demographic_profiles)
            * len(experiment.parameters.seeds)
        )
        assert (summary.n_calls, summary.n_failed, summary.n_succeeded) == (expected, 0, expected)
        assert summary.elapsed >= 0
        assert str(summary) == f"{expected} model calls in {summary.elapsed:.1f}s"

    def test_summaries_add_up(self):
        total = rup.RunSummary(2, 1, 1.0, ["a"]) + rup.RunSummary(3, 0, 2.0, ["a", "b"])
        assert (total.n_calls, total.n_failed, total.elapsed, total.errors) == (
            5,
            1,
            3.0,
            ["a", "b"],
        )

    def test_collection_returns_one_summary_per_experiment(self, config_dict):
        experiments = [make_experiment(config_dict)[0] for _ in range(2)]
        summaries = rup.ExperimentCollection(experiments).run_all(show_progress=False)
        assert [s.n_failed for s in summaries] == [0, 0]


class TestErrorPolicy:
    def test_warn_continues_and_reports_once_at_the_end(self, config_dict):
        experiment, _ = make_experiment(config_dict, fail_on="Muller")
        with pytest.warns(RuntimeWarning, match=r"4 of 8 model calls failed"):
            summary = experiment.run(show_progress=False)

        assert summary.n_failed == 4
        assert summary.errors == ["RuntimeError: cannot answer: Muller"]
        df = experiment.get_answers_as_dataframe()
        assert set(df["Persona ID"]) == {"Conservative Persona"}  # failed answers are absent

    def test_raise_stops_at_the_first_failure(self, config_dict):
        experiment, model = make_experiment(config_dict, fail_on="Muller")
        with pytest.raises(RuntimeError, match="cannot answer"):
            experiment.run(on_error="raise", show_progress=False)
        assert len(model.calls) == 1

    def test_ignore_is_silent(self, config_dict):
        experiment, _ = make_experiment(config_dict, fail_on="Muller")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = experiment.run(on_error="ignore", show_progress=False)
        assert summary.n_failed == 4

    def test_failed_calls_are_logged(self, config_dict, caplog):
        experiment, _ = make_experiment(config_dict, fail_on="Muller")
        with caplog.at_level("WARNING", logger="rupsycho.mixins.experiment_processing"):
            experiment.run(on_error="ignore", show_progress=False)
        assert caplog.text == ""  # "ignore" does not log
        with caplog.at_level("WARNING", logger="rupsycho.mixins.experiment_processing"):
            with pytest.warns(RuntimeWarning):
                experiment.run(show_progress=False)
        assert "Model call failed" in caplog.text and "Optimistic Persona" in caplog.text

    def test_callbacks_see_failures_as_none(self, config_dict):
        experiment, _ = make_experiment(config_dict, fail_on="Muller")
        recorder = Recorder()
        experiment.run(callbacks=[recorder], on_error="ignore", show_progress=False)
        failed = [row for row in recorder.rows if row[1] == "Optimistic Persona"]
        assert failed and all(row[3] is None for row in failed)

    @pytest.mark.parametrize("kwargs", [{"on_error": "explode"}, {"max_concurrency": 0}])
    def test_invalid_arguments_are_rejected(self, config_dict, kwargs):
        experiment, _ = make_experiment(config_dict)
        with pytest.raises(ValueError):
            experiment.run(show_progress=False, **kwargs)

    def test_a_failing_callback_never_stops_the_run(self, config_dict):
        class Broken(Callback):
            def save_answer(self, *args):
                raise OSError("disk full")

        experiment, _ = make_experiment(config_dict)
        with pytest.warns(UserWarning, match="disk full"):
            summary = experiment.run(callbacks=[Broken()], show_progress=False)
        assert summary.n_failed == 0


class TestConcurrency:
    def test_results_and_callback_order_equal_sequential_run(self, config_dict):
        def run(workers):
            experiment, _ = make_experiment(config_dict, answer="len=<len>", delay=0.01)
            recorder = Recorder()
            experiment.run(callbacks=[recorder], max_concurrency=workers, show_progress=False)
            return recorder.rows, experiment.get_answers_as_dataframe()

        sequential_rows, sequential_df = run(1)
        parallel_rows, parallel_df = run(8)

        assert parallel_rows == sequential_rows
        assert parallel_df.equals(sequential_df)

    def test_calls_run_in_parallel(self, config_dict):
        def duration(workers):
            experiment, _ = make_experiment(config_dict, delay=0.1)
            start = time.perf_counter()
            experiment.run(max_concurrency=workers, show_progress=False)
            return time.perf_counter() - start

        sequential, parallel = duration(1), duration(8)
        assert sequential >= 0.8  # 8 calls x 0.1 s
        assert (
            parallel < sequential * 0.9
        )  # ideal is 8x; the bound only guards against serial execution

    def test_a_failure_does_not_affect_other_calls(self, config_dict):
        experiment, _ = make_experiment(config_dict, fail_on="Muller", delay=0.01)
        summary = experiment.run(max_concurrency=4, on_error="ignore", show_progress=False)
        assert (summary.n_calls, summary.n_failed) == (8, 4)

    def test_local_models_fall_back_to_sequential(self, config_dict, tiny_hf_config):
        config = copy.deepcopy(config_dict)
        config["models"] = {"tiny": tiny_hf_config}
        experiment = rup.experiment_from_dict(config)
        with pytest.warns(UserWarning, match="sequentially"):
            summary = experiment.run(max_concurrency=4, show_progress=False)
        assert summary.n_failed == 0


class TestRepeatedRuns:
    def test_an_experiment_can_be_run_twice(self, config_dict):
        experiment, _ = make_experiment(config_dict)
        first = experiment.run(show_progress=False)
        second = experiment.run(show_progress=False)
        assert first.n_calls == second.n_calls
        assert experiment.has_model("scripted")

    def test_lazily_loaded_models_are_released_but_stay_configured(
        self, config_dict, tiny_hf_config
    ):
        config = copy.deepcopy(config_dict)
        config["models"] = {"tiny": tiny_hf_config}
        experiment = rup.experiment_from_dict(config)
        definition = experiment.runnable_models["tiny"]

        experiment.run(show_progress=False)
        assert experiment.runnable_models["tiny"] is definition  # not the loaded model

        assert experiment.run(show_progress=False).n_failed == 0  # reloads from the definition

    def test_the_progress_bar_can_be_disabled(self, config_dict, capsys):
        experiment, _ = make_experiment(config_dict)
        experiment.run(show_progress=False)
        assert "prompts" not in capsys.readouterr().err


class TestCumulativeMode:
    @pytest.fixture
    def memory_config(self, config_dict):
        """A prompt whose user message is asked once per item, as cumulative mode expects."""
        config = copy.deepcopy(config_dict)
        config["prompt_template"] = {
            "type": "chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are {persona_description}. {general_instruction}",
                },
                {"role": "user", "content": "Q: {question} Options: {answer_options} A:"},
            ],
        }
        return config

    def test_later_prompts_contain_the_earlier_questions_and_answers(self, memory_config):
        experiment, model = make_experiment(memory_config, answer="ans")
        experiment.run(cumulative=True, show_progress=False)

        first = experiment.questionnaire.instruction_items[0].question
        second = experiment.questionnaire.instruction_items[1].question
        # prompts whose *live* question (the last one) is the second item
        asking_second = [p for p in model.calls if p.rsplit("Q: ", 1)[-1].startswith(second)]

        assert len(asking_second) == 2  # one per persona
        for prompt in asking_second:
            memory = prompt.rsplit("Q: ", 1)[0]
            assert f"Q: {first}" in memory and memory.rstrip().endswith("A: ans")

    def test_memory_is_per_persona(self, memory_config):
        experiment, model = make_experiment(memory_config, answer="ans")
        experiment.run(cumulative=True, show_progress=False)

        muller = [p for p in model.calls if "Muller" in p]
        assert muller and not any("Grueber" in p for p in muller)

    def test_braces_in_answers_do_not_break_the_next_prompt(self, memory_config):
        experiment, model = make_experiment(memory_config, answer='{"answer": "3"} {x} }')
        summary = experiment.run(cumulative=True, show_progress=False)

        assert summary.n_failed == 0
        assert any('A: {"answer": "3"} {x} }' in p for p in model.calls)

    def test_failed_calls_are_not_remembered(self, memory_config):
        experiment, model = make_experiment(memory_config, fail_on="I see myself")
        experiment.run(cumulative=True, on_error="ignore", show_progress=False)

        first_item_prompts = [p for p in model.calls if "I see myself" in p]
        assert len(first_item_prompts) == 2  # only the first item's calls; nothing was remembered
        assert not any("A: None" in p for p in model.calls)

    def test_first_prompt_equals_the_plain_prompt(self, memory_config):
        plain, plain_model = make_experiment(memory_config)
        plain.run(show_progress=False)
        memory, memory_model = make_experiment(memory_config)
        memory.run(cumulative=True, show_progress=False)
        assert memory_model.calls[0] == plain_model.calls[0]

    def test_requires_a_chat_prompt(self, config_dict):
        config = copy.deepcopy(config_dict)
        config["prompt_template"] = {"type": "normal", "template": "{question}"}
        experiment, _ = make_experiment(config)
        with pytest.raises(ValueError, match="chat prompt"):
            experiment.run(cumulative=True, show_progress=False)
