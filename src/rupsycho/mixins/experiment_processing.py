# ===========================================================================
#                      ExperimentProcessingMixin Implementation
# ===========================================================================
#  This mixin provides methods for processing experiments by creating and
#  executing chains with different models, profiles, and instructions. It
#  includes error handling, progress tracking, and result storage.

from __future__ import annotations

import gc
import logging
import warnings
from collections.abc import Sequence
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from rupsycho.utils import import_tqdm

logger = logging.getLogger(__name__)
tqdm = import_tqdm()  # Import tqdm based on the environment

DEFAULT_SEED = 42

# ------------------- DEFAULT MODEL -------------------


def get_default_model() -> HuggingFacePipeline:
    """Load the small default model (``google/flan-t5-small``) as a LangChain LLM."""
    params = {
        "min_new_tokens": 1,
        "max_new_tokens": 64,
        "temperature": 0.6,
        "do_sample": True,
    }

    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, **params)  # type: ignore[call-overload]
    return HuggingFacePipeline(pipeline=pipe)


# ------------------------------------------------


class ExperimentProcessingMixin:
    """
    A mixin that provides methods for processing experiments with various models,
    profiles, and instructions. This class is designed to facilitate the creation
    and execution of experiment chains, manage errors, track progress, and store results.
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
        def _convert_prompt(self, prompt: Any) -> Any: ...

    # ------------------- Prompt inputs -------------------

    def _create_input_dict(self, profile: Any, instruction_item: Any) -> dict[str, Any]:
        """Create the input dictionary required for invoking the chain."""

        # Use the item's own answer options, or fall back to the questionnaire defaults
        if instruction_item.answer_options:
            answer_options = instruction_item.answer_options.join_options()
        else:
            answer_options = self.questionnaire.default_answer_options.join_options()

        return {
            "general_instruction": self.questionnaire.general_instruction,
            "persona_description": profile.get_profile_desc(),
            "question": instruction_item.question,
            "answer_options": answer_options,
        }

    def _build_input_grid(self, questionnaire: Any, demographic_profiles: dict[str, Any]) -> list:
        """Precompute the prompt inputs for every (item, profile) pair.

        The inputs do not depend on the model or the seed, so they are built once
        and reused for every seed instead of being rebuilt in the innermost loop.
        """
        return [
            (item_id, item, profile_id, self._create_input_dict(profile, item))
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
        params: dict | None = None,
    ) -> Any:
        """Create a chain ``prompt | model | parser`` for running the experiment."""

        if not prompt_template:
            raise ValueError("Prompt template not set.")
        if not model:
            raise ValueError("Model not set.")
        if not parser:
            parser = StrOutputParser()

        # Google models do not accept a seed parameter
        bound_model = (
            model if isinstance(model, ChatGoogleGenerativeAI) else model.bind(seed=int(seed))
        )

        return (RunnablePassthrough() | prompt_template | bound_model | parser).with_config(
            run_name=name
        )

    def _generate_answer(self, chain: Runnable, input_values: dict[str, Any]) -> tuple[Any, float]:
        """Invoke the chain and return ``(answer, elapsed_seconds)``.

        A failing invocation does not abort the experiment: the error is logged and
        the answer is ``None``.
        """
        start = perf_counter()
        try:
            answer = chain.invoke(input_values)
        except Exception as e:
            logger.error("Error invoking chain for run: %s", e)
            answer = None
        return answer, round(perf_counter() - start, 3)

    # ------------------- Validation helpers -------------------

    def _ensure_requirements_to_run(self) -> bool:
        """Ensure that at least one model, a prompt, a questionnaire and personas are set."""

        if not self.runnable_models:
            raise ValueError("No models have been set in runnable_models.")
        if self.runnable_prompt is None:
            raise ValueError("runnable_prompt has not been set.")
        if self.questionnaire is None:
            raise ValueError("questionnaire has not been set.")
        if self.demographic_profiles is None:
            raise ValueError("demographic_profiles has not been set.")

        return True

    def _get_seed_values(self) -> list:
        """Return the seed values to be used in the experiment."""
        return self.parameters.seeds if self.parameters.seeds else [DEFAULT_SEED]

    def _calculate_total_iterations(self) -> int:
        """Calculate the total number of iterations for the progress bar."""
        return (
            len(self.questionnaire.instruction_items)
            * len(self.runnable_models)
            * len(self.demographic_profiles)
            * len(self._get_seed_values())
        )

    def _is_runnable(self, model: Any) -> bool:
        """Check if the model is a runnable LangChain model."""
        return isinstance(model, Runnable)

    def print_assembled_prompt(self, item_idx: int = 0, persona_idx: int = 0) -> None:
        """
        Print the fully assembled prompt with the specified instruction item and persona.

        Note: does not support assembly of a response memory prompt.
        """
        try:
            profile = list(self.demographic_profiles.values())[persona_idx]
            item = self.questionnaire.instruction_items[item_idx]
            input_values = self._create_input_dict(profile, item)
            assembled_prompt = self.get_prompt().format(**input_values)
            print(
                f"\n++++++++++++++++++ assembled prompt (item {item_idx}, persona {persona_idx}) ++++++++++++++++++\n"
                + assembled_prompt
                + "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )

        except Exception as e:
            warnings.warn(f"Failed to assemble prompt: {e}", UserWarning, stacklevel=2)

    # ------------------- Memory management -------------------

    def _cleanup_memory(self) -> None:
        """Cleanup memory by running garbage collection and emptying the CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
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
        instruction_item_id: int,
        instruction_item: Any,
        model_id: str,
        profile_id: str,
        random_seed: Any,
        elapsed: float,
        answer: Any,
    ) -> None:
        """Store an answer on the instruction item and forward it to all callbacks."""
        instruction_item.update_answer(model_id, profile_id, random_seed, answer)
        self._trigger_callbacks(
            callbacks,
            instruction_item_id,
            instruction_item,
            model_id,
            profile_id,
            random_seed,
            elapsed,
            answer,
        )

    def _generate_and_process_answers(
        self,
        model: Any,
        model_id: str,
        seed_values: list,
        params: dict,
        questionnaire: Any,
        demographic_profiles: dict[str, Any],
        callbacks: Sequence,
        pbar: Any,
    ) -> None:
        """Generate answers for each random seed, instruction item and demographic profile."""
        input_grid = self._build_input_grid(questionnaire, demographic_profiles)

        for random_seed in seed_values:
            # The chain only depends on the model and the seed
            chain = self._get_chain(
                self.runnable_prompt,
                model,
                self.runnable_parser,
                model_id,
                seed=int(random_seed),
                params=params,
            )

            for item_id, item, profile_id, input_values in input_grid:
                answer, elapsed = self._generate_answer(chain, input_values)
                self._record_answer(
                    callbacks, item_id, item, model_id, profile_id, random_seed, elapsed, answer
                )
                pbar.update(1)

    def _generate_and_process_answers_cumulative(
        self,
        model: Any,
        model_id: str,
        seed_values: list,
        params: dict,
        questionnaire: Any,
        demographic_profiles: dict[str, Any],
        callbacks: Sequence,
        pbar: Any,
    ) -> None:
        """Generate answers with a 'response memory' prompt that accumulates prior items and answers."""
        input_grid = self._build_input_grid(questionnaire, demographic_profiles)

        base_messages = {
            "type": "chat",
            "messages": [
                {
                    "role": "system",
                    "content": self.runnable_prompt.messages[0].prompt.template,
                },
                {
                    "role": "user",
                    "content": self.runnable_prompt.messages[1].prompt.template,
                },
            ],
        }
        base_message_str_user = base_messages["messages"][1]["content"]  # type: ignore[index]

        for random_seed in seed_values:
            # Each persona gets its own prompt with its individual prior answers
            persona_messages = {pid: deepcopy(base_messages) for pid in demographic_profiles}

            for item_id, item, profile_id, input_values in input_grid:
                current_prompt = self._convert_prompt(
                    persona_messages[profile_id]
                ).load_prompt_template()

                # The prompt changes with every answer, so the chain is rebuilt per call
                chain = self._get_chain(
                    current_prompt,
                    model,
                    self.runnable_parser,
                    model_id,
                    seed=int(random_seed),
                    params=params,
                )
                answer, elapsed = self._generate_answer(chain, input_values)

                # Extend this persona's prompt with the question and the answer just given
                user_message = persona_messages[profile_id]["messages"][1]
                user_message["content"] = (  # type: ignore[index]
                    user_message["content"].format(question=item.question)  # type: ignore[index]
                    + " "
                    + str(answer)
                    + "\n"
                    + base_message_str_user
                )

                self._record_answer(
                    callbacks, item_id, item, model_id, profile_id, random_seed, elapsed, answer
                )
                pbar.update(1)

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
        self, cumulative: bool, pbar: Any = None, callbacks: Sequence = ()
    ) -> None:
        """Process the experiment, iterating through models, profiles, and instruction items."""

        self._ensure_requirements_to_run()

        # Create a progress bar for tracking the experiment progress if not provided
        if pbar is None:
            pbar = tqdm(
                total=self._calculate_total_iterations(),
                desc=self.name or "Experiment",
                unit=" prompts",
            )

        # Precompute default parameters once as they do not change per model
        default_params = self.parameters.model_dump(exclude_none=True, exclude=["seeds"])

        seed_values = self._get_seed_values()
        generate = (
            self._generate_and_process_answers_cumulative
            if cumulative
            else self._generate_and_process_answers
        )

        for model_id, model in list(self.runnable_models.items()):
            # Load the model (lazy loading if required)
            model = self._load_model(model)
            self.runnable_models[model_id] = model

            # Merge model-specific parameters over the experiment defaults
            model_config = self.models.get(model_id)
            model_params = model_config.parameters if model_config else {}
            params = {**default_params, **model_params}

            generate(
                model,
                model_id,
                seed_values,
                params,
                self.questionnaire,
                self.demographic_profiles,
                callbacks,
                pbar,
            )

            # Release the model before the next one is loaded
            del model
            self.runnable_models[model_id] = None
            self._cleanup_memory()

    def run(self, callbacks: Sequence = (), cumulative: bool = False) -> None:
        """Run the experiment processing with a progress bar and callbacks."""

        self._ensure_requirements_to_run()

        with tqdm(
            total=self._calculate_total_iterations(),
            desc=self.name or "Experiment",
            unit=" prompts",
        ) as pbar:
            self.process_single_experiment(cumulative, pbar, callbacks=callbacks)
