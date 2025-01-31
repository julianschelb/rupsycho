# ===========================================================================
#                      ExperimentProcessingMixin Implementation
# ===========================================================================
#  This mixin provides methods for processing experiments by creating and
#  executing chains with different models, profiles, and instructions. It
#  includes error handling, progress tracking, and result storage.


import gc
import torch
import warnings
from collections import defaultdict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface.llms import HuggingFacePipeline
from rupsycho.utils import import_tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Any, Dict, Optional
tqdm = import_tqdm()  # Import tqdm based on the environment

# ------------------- DEFAULT MODEL -------------------


def get_default_model() -> HuggingFacePipeline:
    params = {
        "min_new_tokens": 1,
        "max_new_tokens": 64,
        "temperature": 0.6,
        "do_sample": True,
    }

    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model,
                    tokenizer=tokenizer, **params)
    return HuggingFacePipeline(pipeline=pipe)

# ------------------------------------------------


class ExperimentProcessingMixin:
    """
    A mixin that provides methods for processing experiments with various models,
    profiles, and instructions. This class is designed to facilitate the creation
    and execution of experiment chains, manage errors, track progress, and store results.
    """

    def _create_input_dict(self, profile: Any, instruction_item: Any) -> Dict[str, Any]:
        """Create the input dictionary required for invoking the chain."""

        # Get the answer options for the instruction item, or use the default options if none are provided
        if instruction_item.answer_options:
            answer_options = instruction_item.answer_options.join_options()
        else:
            answer_options = self.questionnaire.default_answer_options.join_options()

        # Return the input dictionary
        return {
            "general_instruction": self.questionnaire.general_instruction,
            "persona_description": profile.get_profile_desc(),
            "question": instruction_item.question,
            "answer_options": answer_options  # Joined options with the specified delimiter
        }

    def _get_chain(self, prompt_template: Runnable, model: Runnable, parser: Runnable, name: str, seed: int = 42, params: dict = {}) -> Any:
        """Create a chain for running the experiment using the model identified by model_key."""

        # Check if the prompt template, model, and parser are set
        if not prompt_template:
            raise ValueError("Prompt template not set.")
        if not model:
            raise ValueError("Model not set.")
        if not parser:
            parser = StrOutputParser()

        # Create the chain
        return (
            RunnablePassthrough()
            | prompt_template
            | model.bind(seed=int(seed))  # TODO: Add params
            | parser
        ).with_config(run_name=name)

    def _generate_answer(self, chain, input_values):
        """Invoke the chain to generate an answer."""
        try:
            answer = chain.invoke(input_values)
        except Exception as e:
            print(f"Error invoking chain for run: {e}")
            answer = None
        return answer

    def _ensure_requirements_to_run(self) -> None:
        """Ensure that at least one model and a prompt are set."""

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
        return self.parameters.seeds if self.parameters.seeds else [42]

    def _calculate_total_iterations(self) -> int:
        """Calculate the total number of iterations for the progress bar."""
        return (
            len(self.questionnaire.instruction_items)
            * len(self.runnable_models)
            * len(self.demographic_profiles)
            * len(self._get_seed_values())
        )

    def _is_runnable(self, model):
        """Check if the model is a runnable LangChain model."""
        return isinstance(model, Runnable)

    def _cleanup_memory(self):
        """Cleanup memory by running garbage collection and emptying the cache."""
        gc.collect()
        torch.cuda.empty_cache()

    def _load_model(self, model):
        """Lazy load the model if it's not runnable."""
        if not self._is_runnable(model):
            return model.load_model()
        return model

    def _generate_and_process_answers(self, model, model_id, seed_values, params, questionnaire, demographic_profiles, callbacks, pbar):
        """Generate answers for each instruction item, random seed, and demographic profile."""
        # Iterate over random seeds
        for random_seed in seed_values:

            # Generate the chain for the current model and seed combination.
            chain = self._get_chain(self.runnable_prompt, model, self.runnable_parser,
                                    model_id, seed=int(random_seed), params=params)

            # Iterate over each instruction item in the questionnaire
            for instruction_item_id, instruction_item in enumerate(questionnaire.instruction_items):

                # Iterate over each demographic profile
                for profile_id, profile in demographic_profiles.items():

                    # Create an input dictionary based on the profile and instruction item
                    input_values = self._create_input_dict(
                        profile, instruction_item)

                    # Generate an answer using the chain
                    answer = self._generate_answer(chain, input_values)

                    # Update the instruction item with the generated answer.
                    instruction_item.update_answer(
                        model_id, profile_id, random_seed, answer)

                    # Trigger all callbacks to save the answer
                    self._trigger_callbacks(
                        callbacks, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer)

                    # Update the progress bar
                    pbar.update(1)

    def _trigger_callbacks(self, callbacks, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        """Trigger all the callbacks to save the generated answer."""
        for callback in callbacks:
            try:
                callback.save_answer(
                    self, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer)
            except Exception as e:
                warnings.warn(f"Error while saving answer: {e}")

    def process_single_experiment(self, pbar=None, callbacks=[]) -> None:
        """Process the experiment, iterating through models, profiles, and instruction items."""

        # ------------------- Validation -------------------
        self._ensure_requirements_to_run()

        # Create a progress bar for tracking the experiment progress if not provided
        if pbar is None:
            pbar = tqdm(total=self._calculate_total_iterations(),
                        desc=self.name or "Experiment", unit=" prompts")

        # Precompute default parameters once as they do not change per model
        default_params = self.parameters.model_dump(
            exclude_none=True, exclude=['seeds'])

        # Pre-fetch reusable properties
        seed_values = self._get_seed_values()
        demographic_profiles = self.demographic_profiles
        runnable_models = self.runnable_models
        questionnaire = self.questionnaire

        # ------------------- Model Processing -------------------
        for model_id, model in runnable_models.items():

            # Load the model (lazy loading if required)
            model = self._load_model(model)
            # Update the model reference
            self.runnable_models[model_id] = model

            # Retrieve model-specific parameters if available, and merge them with the default parameters.
            model_config = self.models.get(model_id)
            model_params = model_config.parameters if model_config else {}
            params = {**default_params, **model_params}

            # Generate answers for the current model
            self._generate_and_process_answers(
                model, model_id, seed_values, params, questionnaire, demographic_profiles, callbacks, pbar)

            # Cleanup memory after processing the model
            del model
            self.runnable_models[model_id] = None  # Remove from the dictionary
            gc.collect()  # Cleanup CPU memory
            torch.cuda.empty_cache()  # Cleanup GPU memory

    def run(self, callbacks=[]) -> None:
        """Run the experiment processing with a progress bar and callbacks."""

        # Ensure that the required attributes are set
        self._ensure_requirements_to_run()

        # Create a progress bar for tracking the experiment progress
        with tqdm(total=self._calculate_total_iterations(), desc=self.name or "Experiment", unit=" prompts") as pbar:
            self.process_single_experiment(pbar, callbacks=callbacks)
