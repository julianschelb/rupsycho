# experiment_processing.py
"""Run loop of an experiment.

``ExperimentProcessingMixin`` implements ``ExperimentDocument.run``: for every model and
every seed it asks every persona every question, records the answers on the questionnaire
items and forwards them to callbacks.
"""

from __future__ import annotations

import gc
import logging
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from rupsycho.seeding import is_thread_safe, seed_model
from rupsycho.utils import import_tqdm

__all__ = ["DEFAULT_SEED", "ErrorPolicy", "ExperimentProcessingMixin", "RunSummary"]

logger = logging.getLogger(__name__)
tqdm = import_tqdm()  # Import tqdm based on the environment

DEFAULT_SEED = 42
"""Seed used when an experiment defines none."""

ErrorPolicy = Literal["warn", "raise", "ignore"]
"""What to do when a model call fails: warn and continue (default), raise, or stay silent."""

_MAX_RECORDED_ERRORS = 5


@dataclass
class RunSummary:
    """Outcome of :meth:`ExperimentProcessingMixin.run`.

    Attributes:
        n_calls: Number of model calls made.
        n_failed: Number of calls that raised; their answers are missing from the results.
        elapsed: Wall-clock seconds of the whole run.
        errors: The first distinct error messages (at most five).

    Example:
        ```python
        summary = experiment.run()
        if summary.n_failed:
            print(summary.errors)
        ```
    """

    n_calls: int = 0
    n_failed: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def n_succeeded(self) -> int:
        """Number of calls that returned an answer."""
        return self.n_calls - self.n_failed

    def __add__(self, other: RunSummary) -> RunSummary:
        merged = self.errors + [e for e in other.errors if e not in self.errors]
        return RunSummary(
            self.n_calls + other.n_calls,
            self.n_failed + other.n_failed,
            self.elapsed + other.elapsed,
            merged[:_MAX_RECORDED_ERRORS],
        )

    def __str__(self) -> str:
        text = f"{self.n_calls} model calls in {self.elapsed:.1f}s"
        return text + (f", {self.n_failed} failed" if self.n_failed else "")


@dataclass
class _Call:
    """One model call of the grid: an instruction item asked to one persona."""

    item_id: int
    item: Any
    profile_id: str
    inputs: dict[str, Any]


_CallResult = tuple[Any, float, "Exception | None"]


def _escape(text: str) -> str:
    """Escape braces so that ``text`` is taken literally inside a prompt template."""
    return text.replace("{", "{{").replace("}", "}}")


class _SafeFormat(dict):
    """``str.format_map`` helper that leaves unknown placeholders untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class ExperimentProcessingMixin:
    """Runs an experiment: models x seeds x items x personas.

    The mixin relies on the attributes of ``ExperimentDocument`` (``questionnaire``,
    ``demographic_profiles``, ``runnable_models``, ``runnable_prompt``, ``runnable_parser``,
    ``parameters``, ``models`` and ``name``).
    """

    if TYPE_CHECKING:
        # Provided by ExperimentDocument, which mixes this class in.
        name: str | None
        parameters: Any
        models: dict[str, Any]
        runnable_models: dict[str, Any]
        runnable_prompt: Any
        runnable_parser: Any
        questionnaire: Any
        demographic_profiles: dict[str, Any]

        def get_prompt(self) -> Any: ...

    # ------------------- Prompt inputs -------------------

    def _create_input_dict(self, profile: Any, instruction_item: Any) -> dict[str, Any]:
        """Build the template variables for asking ``instruction_item`` to ``profile``."""

        # Use the item's own answer options, or fall back to the questionnaire defaults
        if instruction_item.answer_options and instruction_item.answer_options.options:
            answer_options = instruction_item.answer_options.join_options()
        else:
            answer_options = self.questionnaire.default_answer_options.join_options()

        try:
            persona_description = profile.get_profile_desc()
        except KeyError as e:
            raise ValueError(
                f"The persona template uses the attribute {e}, which the persona does not define"
            ) from e

        return {
            "general_instruction": self.questionnaire.general_instruction,
            "persona_description": persona_description,
            "question": instruction_item.question,
            "answer_options": answer_options,
        }

    def _build_call_grid(
        self, questionnaire: Any, demographic_profiles: dict[str, Any]
    ) -> list[_Call]:
        """Precompute the prompt inputs of every (item, persona) pair.

        The inputs depend neither on the model nor on the seed, so they are built once
        per run instead of once per call.
        """
        return [
            _Call(item_id, item, profile_id, self._create_input_dict(profile, item))
            for item_id, item in enumerate(questionnaire.instruction_items)
            for profile_id, profile in demographic_profiles.items()
        ]

    # ------------------- Chain handling -------------------

    def _get_chain(
        self,
        prompt_template: Runnable,
        model: Runnable,
        parser: Runnable | None,
        name: str,
        seed: int = DEFAULT_SEED,
    ) -> Any:
        """Create the chain ``prompt | seeded model | parser`` for one model and seed."""

        if not prompt_template:
            raise ValueError("Prompt template not set.")
        if not model:
            raise ValueError("Model not set.")
        if not parser:
            parser = StrOutputParser()

        return (
            RunnablePassthrough() | prompt_template | seed_model(model, seed) | parser
        ).with_config(run_name=name)

    def _invoke(self, chain: Runnable, input_values: dict[str, Any]) -> _CallResult:
        """Invoke the chain once; never raises. Returns ``(answer, seconds, error)``."""
        start = perf_counter()
        try:
            answer, error = chain.invoke(input_values), None
        except Exception as e:
            answer, error = None, e
        return answer, round(perf_counter() - start, 3), error

    def _generate_answer(self, chain: Runnable, input_values: dict[str, Any]) -> tuple[Any, float]:
        """Invoke the chain and return ``(answer, elapsed_seconds)``.

        A failing invocation does not abort the experiment: the error is logged and the
        answer is ``None``.
        """
        answer, elapsed, error = self._invoke(chain, input_values)
        if error is not None:
            logger.error("Error invoking chain for run: %s", error)
        return answer, elapsed

    @staticmethod
    def _run_calls(
        thunks: Sequence[Callable[[], _CallResult]], max_workers: int
    ) -> Iterator[_CallResult]:
        """Run the calls and yield their results **in submission order**.

        With ``max_workers > 1`` the calls run in a thread pool (for I/O-bound API models);
        order and per-call error isolation are preserved either way.
        """
        if max_workers <= 1 or len(thunks) <= 1:
            for thunk in thunks:
                yield thunk()
            return

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(thunk) for thunk in thunks]
            try:
                for future in futures:
                    yield future.result()
            finally:
                for future in futures:
                    future.cancel()

    # ------------------- Validation helpers -------------------

    def _ensure_requirements_to_run(self) -> bool:
        """Ensure that models, a prompt, a questionnaire and personas are set."""

        if not self.runnable_models:
            raise ValueError("No models have been set in runnable_models.")
        if self.runnable_prompt is None:
            raise ValueError("runnable_prompt has not been set.")
        if self.questionnaire is None:
            raise ValueError("questionnaire has not been set.")
        if self.demographic_profiles is None:
            raise ValueError("demographic_profiles has not been set.")
        if not self.questionnaire.instruction_items:
            raise ValueError("The questionnaire has no instruction_items to ask.")
        if not self.questionnaire.default_answer_options:
            missing = [
                position
                for position, item in enumerate(self.questionnaire.instruction_items)
                if not (item.answer_options and item.answer_options.options)
            ]
            if missing:
                raise ValueError(
                    f"Item(s) {missing} have no answer options and the questionnaire defines no "
                    "default_answer_options."
                )

        return True

    def _get_seed_values(self) -> list:
        """Return the seed values to be used in the experiment."""
        return self.parameters.seeds if self.parameters.seeds else [str(DEFAULT_SEED)]

    def _calculate_total_iterations(self) -> int:
        """Calculate the total number of model calls for the progress bar."""
        return (
            len(self.questionnaire.instruction_items)
            * len(self.runnable_models)
            * len(self.demographic_profiles)
            * len(self._get_seed_values())
        )

    def _is_runnable(self, model: Any) -> bool:
        """Check if the model is a runnable LangChain model."""
        return isinstance(model, Runnable)

    def assemble_prompt(self, item_idx: int = 0, persona_idx: int = 0) -> str:
        """Return the fully assembled prompt for one item and persona, exactly as sent.

        Args:
            item_idx: Index of the instruction item.
            persona_idx: Index of the demographic profile (in configuration order).

        Returns:
            The prompt text; for chat prompts the messages are joined as ``"System: ..."`` /
            ``"Human: ..."`` blocks.

        Raises:
            IndexError: If ``item_idx`` or ``persona_idx`` is out of range.
            ValueError: If the experiment has no questionnaire or prompt.

        Note:
            Does not include the accumulated memory of ``cumulative`` runs.

        Example:
            ```python
            print(experiment.assemble_prompt(item_idx=0, persona_idx=1))
            ```
        """
        if self.questionnaire is None or self.runnable_prompt is None:
            raise ValueError("The experiment needs a questionnaire and a prompt template.")
        profile = list(self.demographic_profiles.values())[persona_idx]
        item = self.questionnaire.instruction_items[item_idx]
        return str(self.runnable_prompt.format(**self._create_input_dict(profile, item)))

    def print_assembled_prompt(self, item_idx: int = 0, persona_idx: int = 0) -> None:
        """Print the assembled prompt of one item and persona (see ``assemble_prompt``).

        Problems (for example an out-of-range index) are reported as a warning instead of
        raising, which is convenient in notebooks.

        Args:
            item_idx: Index of the instruction item.
            persona_idx: Index of the demographic profile.
        """
        try:
            assembled_prompt = self.assemble_prompt(item_idx, persona_idx)
        except Exception as e:
            warnings.warn(f"Failed to assemble prompt: {e}", UserWarning, stacklevel=2)
            return
        print(
            f"\n++++++++++++++++++ assembled prompt (item {item_idx}, persona {persona_idx}) "
            f"++++++++++++++++++\n{assembled_prompt}\n"
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

    # ------------------- Memory management -------------------

    def _cleanup_memory(self) -> None:
        """Run garbage collection and release cached GPU memory if PyTorch is in use."""
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model: Any) -> Any:
        """Lazy load the model if it's not runnable."""
        if not self._is_runnable(model):
            return model.load_model()
        return model

    # ------------------- Answer generation -------------------

    def _record_answer(
        self,
        callbacks: Sequence,
        call: _Call,
        model_id: str,
        random_seed: Any,
        elapsed: float,
        answer: Any,
    ) -> None:
        """Store an answer on the instruction item and forward it to all callbacks."""
        call.item.update_answer(model_id, call.profile_id, random_seed, answer)
        self._trigger_callbacks(
            callbacks,
            call.item_id,
            call.item,
            model_id,
            call.profile_id,
            random_seed,
            elapsed,
            answer,
        )

    def _handle_result(
        self,
        result: _CallResult,
        call: _Call,
        model_id: str,
        random_seed: Any,
        callbacks: Sequence,
        pbar: Any,
        summary: RunSummary,
        on_error: ErrorPolicy,
    ) -> Any:
        """Count a finished call, apply the error policy, record the answer."""
        answer, elapsed, error = result
        summary.n_calls += 1
        if error is not None:
            summary.n_failed += 1
            message = f"{type(error).__name__}: {error}"
            if message not in summary.errors and len(summary.errors) < _MAX_RECORDED_ERRORS:
                summary.errors.append(message)
            if on_error == "raise":
                raise error
            if on_error == "warn":
                logger.warning(
                    "Model call failed (model=%s, persona=%s, item=%s, seed=%s): %s",
                    model_id,
                    call.profile_id,
                    call.item_id,
                    random_seed,
                    message,
                )
        self._record_answer(callbacks, call, model_id, random_seed, elapsed, answer)
        pbar.update(1)
        return answer

    def _generate_and_process_answers(
        self,
        model: Any,
        model_id: str,
        seed_values: list,
        grid: list[_Call],
        callbacks: Sequence,
        pbar: Any,
        summary: RunSummary,
        on_error: ErrorPolicy = "warn",
        max_workers: int = 1,
    ) -> None:
        """Ask every persona every question once per seed."""

        for random_seed in seed_values:
            # The chain only depends on the model and the seed
            chain = self._get_chain(
                self.runnable_prompt, model, self.runnable_parser, model_id, seed=int(random_seed)
            )
            thunks = [partial(self._invoke, chain, call.inputs) for call in grid]
            for call, result in zip(grid, self._run_calls(thunks, max_workers)):
                self._handle_result(
                    result, call, model_id, random_seed, callbacks, pbar, summary, on_error
                )

    def _memory_prompt(
        self,
        history: list[tuple[str, str]],
        system_template: Any,
        user_template: Any,
        rest: Sequence[Any] = (),
    ) -> Runnable:
        """Build the chat prompt of one persona that remembers its earlier answers.

        ``history`` holds the already rendered earlier user messages with the answers given.
        They are inserted literally (braces escaped), followed by the live user template. Any
        messages after the user message (``rest``) are kept in place.
        """
        memory = "".join(f"{_escape(question)} {_escape(answer)}\n" for question, answer in history)
        user_text = memory + user_template.prompt.template
        return ChatPromptTemplate.from_messages(
            [system_template, type(user_template).from_template(user_text), *rest]
        )

    def _generate_and_process_answers_cumulative(
        self,
        model: Any,
        model_id: str,
        seed_values: list,
        grid: list[_Call],
        callbacks: Sequence,
        pbar: Any,
        summary: RunSummary,
        on_error: ErrorPolicy = "warn",
        max_workers: int = 1,
    ) -> None:
        """Ask the questions in order, letting each persona remember its earlier answers.

        Personas are independent of each other, so the calls of one item run concurrently
        when ``max_workers > 1``; the items themselves are asked in order.
        """
        messages = getattr(self.runnable_prompt, "messages", None)
        if (
            not isinstance(self.runnable_prompt, ChatPromptTemplate)
            or messages is None
            or len(messages) < 2
        ):
            raise ValueError(
                "Cumulative mode needs a chat prompt template with a system message followed "
                "by a user message."
            )
        system_template, user_template, rest = messages[0], messages[1], messages[2:]
        calls_by_item: dict[int, list[_Call]] = {}
        for call in grid:
            calls_by_item.setdefault(call.item_id, []).append(call)

        for random_seed in seed_values:
            history: dict[str, list[tuple[str, str]]] = {call.profile_id: [] for call in grid}

            for item_calls in calls_by_item.values():
                thunks = []
                for call in item_calls:
                    prompt = self._memory_prompt(
                        history[call.profile_id], system_template, user_template, rest
                    )
                    chain = self._get_chain(
                        prompt, model, self.runnable_parser, model_id, seed=int(random_seed)
                    )
                    thunks.append(partial(self._invoke, chain, call.inputs))

                for call, result in zip(item_calls, self._run_calls(thunks, max_workers)):
                    answer = self._handle_result(
                        result, call, model_id, random_seed, callbacks, pbar, summary, on_error
                    )
                    if answer is not None:
                        # Remember the question as the persona saw it, plus the answer given
                        rendered = user_template.prompt.template.format_map(
                            _SafeFormat(call.inputs)
                        )
                        history[call.profile_id].append((rendered, str(answer)))

    def _trigger_callbacks(
        self,
        callbacks: Sequence,
        instruction_item_id: int,
        instruction_item: Any,
        model_id: str,
        profile_id: str,
        random_seed: Any,
        time: float,
        answer: Any,
    ) -> None:
        """Trigger all the callbacks to save the generated answer."""
        for callback in callbacks:
            try:
                callback.save_answer(
                    self,
                    instruction_item_id,
                    instruction_item,
                    model_id,
                    profile_id,
                    random_seed,
                    time,
                    answer,
                )
            except Exception as e:
                warnings.warn(f"Error while saving answer: {e}", stacklevel=2)

    # ------------------- Entry points -------------------

    def process_single_experiment(
        self,
        cumulative: bool,
        pbar: Any = None,
        callbacks: Sequence = (),
        *,
        max_concurrency: int = 1,
        on_error: ErrorPolicy = "warn",
    ) -> RunSummary:
        """Process every model of the experiment and return a :class:`RunSummary`.

        Args:
            cumulative: Let each persona remember its earlier answers.
            pbar: Progress bar to update; one is created if omitted.
            callbacks: Callbacks that receive every answer.
            max_concurrency: Calls to run at the same time (API models only; local
                Hugging Face models always run sequentially).
            on_error: ``"warn"`` (default) logs failed calls and continues, ``"raise"``
                stops at the first failure, ``"ignore"`` continues silently.

        Returns:
            A summary of the calls made.

        Raises:
            ValueError: If ``on_error`` or ``max_concurrency`` is invalid or the experiment
                is incomplete.
        """
        if on_error not in ("warn", "raise", "ignore"):
            raise ValueError(f"on_error must be 'warn', 'raise' or 'ignore', got {on_error!r}")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._ensure_requirements_to_run()

        own_pbar = pbar is None
        if own_pbar:
            pbar = tqdm(
                total=self._calculate_total_iterations(),
                desc=self.name or "Experiment",
                unit=" prompts",
            )

        seed_values = self._get_seed_values()
        generate = (
            self._generate_and_process_answers_cumulative
            if cumulative
            else self._generate_and_process_answers
        )
        # The prompt inputs depend neither on the model nor on the seed: build them once
        grid = self._build_call_grid(self.questionnaire, self.demographic_profiles)
        summary = RunSummary()
        start = perf_counter()

        for model_id, original in list(self.runnable_models.items()):
            # Load the model (lazy loading if required)
            model = self._load_model(original)

            workers = max_concurrency
            if workers > 1 and not is_thread_safe(model):
                warnings.warn(
                    f"Model '{model_id}' runs in this process and cannot be called concurrently; "
                    "running it sequentially.",
                    UserWarning,
                    stacklevel=3,
                )
                workers = 1

            try:
                generate(
                    model,
                    model_id,
                    seed_values,
                    grid,
                    callbacks,
                    pbar,
                    summary,
                    on_error,
                    workers,
                )
            finally:
                # Release a model that was loaded for this run but keep its definition, so the
                # experiment can be run again. Ready models handed in by the user stay alive.
                loaded_here = model is not original
                del model
                self.runnable_models[model_id] = original
                if loaded_here:
                    self._cleanup_memory()

        summary.elapsed = round(perf_counter() - start, 3)
        if own_pbar:
            pbar.close()
        if summary.n_failed and on_error == "warn":
            warnings.warn(
                f"{summary.n_failed} of {summary.n_calls} model calls failed and are missing "
                f"from the results (first error: {summary.errors[0]}).",
                RuntimeWarning,
                stacklevel=3,
            )
        return summary

    def run(
        self,
        callbacks: Sequence = (),
        cumulative: bool = False,
        *,
        max_concurrency: int = 1,
        on_error: ErrorPolicy = "warn",
        show_progress: bool = True,
    ) -> RunSummary:
        """Run the experiment.

        Every model is asked every question as every persona, once per seed. Answers are
        stored on the questionnaire items (see ``get_answers`` / ``get_answers_as_dataframe``)
        and passed to the callbacks as soon as they are generated.

        Args:
            callbacks: Callbacks such as ``CSVCallback`` that receive every answer.
            cumulative: Let each persona remember its earlier answers ("response memory").
                Needs a chat prompt whose user message is asked once per item.
            max_concurrency: Number of calls to run in parallel. Useful for API models where
                the time is spent waiting; models running in this process (local Hugging
                Face) are always called sequentially. Results and callbacks keep their order.
            on_error: What to do when a model call fails. ``"warn"`` (default) logs the
                failure, continues and warns once at the end; ``"raise"`` stops at the first
                failure; ``"ignore"`` continues silently. Failed calls have no answer.
            show_progress: Show a progress bar.

        Returns:
            A [`RunSummary`][rupsycho.mixins.experiment_processing.RunSummary] with the number
            of calls, failures and the elapsed time.

        Raises:
            ValueError: If the experiment is incomplete (no model, prompt, questionnaire).

        Example:
            ```python
            summary = experiment.run(callbacks=[CSVCallback("answers.csv")], max_concurrency=8)
            print(summary)  # "160 model calls in 12.3s"
            ```
        """
        self._ensure_requirements_to_run()

        with tqdm(
            total=self._calculate_total_iterations(),
            desc=self.name or "Experiment",
            unit=" prompts",
            disable=not show_progress,
        ) as pbar:
            return self.process_single_experiment(
                cumulative,
                pbar,
                callbacks=callbacks,
                max_concurrency=max_concurrency,
                on_error=on_error,
            )
