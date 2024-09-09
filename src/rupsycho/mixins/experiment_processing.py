# ===========================================================================
#                      ExperimentProcessingMixin Implementation
# ===========================================================================
#  This mixin provides methods for processing experiments by creating and
#  executing chains with different models, profiles, and instructions. It
#  includes error handling, progress tracking, and result storage.

import warnings
from collections import defaultdict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

    def create_chain(self, model_key: str, seed: int = 42, params: dict = {}) -> Any:
        """Create a chain for running the experiment using the model identified by model_key."""
        self._ensure_models_and_prompt()

        parser = self.runnable_parser or self._default_parser()

        return (
            RunnablePassthrough()
            | self.runnable_prompt
            | self.runnable_models[model_key]  # .bind(seed=seed, **params)
            | parser
        ).with_config(run_name=f"experiment_chain_{model_key}")

    def create_all_chains(self, seed: int = 42, params: dict = {}) -> Dict[str, Any]:
        """Create chains for running the experiment with all models."""
        return {model_key: self.create_chain(model_key, seed, params) for model_key in self.runnable_models}

    def _default_parser(self) -> Any:
        """Provide a default parser if none is set."""
        warnings.warn(
            "No parser set. Using StrOutputParser as default.", UserWarning)
        return StrOutputParser()

    def _ensure_models_and_prompt(self) -> None:
        """Ensure that at least one model and a prompt are set."""
        if not self.runnable_models:
            raise ValueError("No models have been set in runnable_models.")
        if self.runnable_prompt is None:
            raise ValueError("runnable_prompt has not been set.")

    def initialize_answers_if_needed(self, instruction_item: Any) -> None:
        """Initialize the 'answers' attribute as a nested defaultdict if it doesn't exist."""
        if not hasattr(instruction_item, "answers"):
            instruction_item.answers = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )

    def create_input_dict(self, profile: Any, instruction_item: Any) -> Dict[str, Any]:
        """Create the input dictionary required for invoking the chain."""

        if instruction_item.answer_options:
            answer_options = instruction_item.answer_options.values()
        else:
            answer_options = self.questionnaire.default_answer_options.values()

        return {
            "general_instruction": self.questionnaire.general_instruction,
            "persona_description": profile,
            "question": instruction_item.question,
            "answer_options": " ".join(
                option.text for option in answer_options
            )
        }

    def process_single_experiment(self, pbar) -> None:
        """Process the experiment, iterating through models, profiles, and instruction items."""

        # Ensure at least one model is available
        if not self.runnable_models:
            self.add_model(get_default_model(), "default_model")
            warnings.warn(
                "No models added to the experiment. Using default model.", UserWarning)

        # Iterate through instruction items, models, profiles, and seeds
        for instruction_item in self.questionnaire.instruction_items:
            self.initialize_answers_if_needed(instruction_item)

            # Iterate over the seed values for reproducibility
            for run_idx in self._get_seed_values():

                # Create chains for each model using the current seed

                all_chains = self.create_all_chains(
                    seed=run_idx, params=self.parameters.model_dump())
                for model_key, chain in all_chains.items():

                    # Iterate over all demographic profiles
                    for profile_key, profile in self.demographic_profiles.items():

                        # Create input dictionary for the model based on the profile and instruction item
                        input_dict = self.create_input_dict(
                            profile, instruction_item)

                        # Invoke the model chain and get the generated answer
                        answer = self._invoke_chain(chain, input_dict, run_idx)

                        # Store the answer in the appropriate location
                        self._store_answer(
                            instruction_item, model_key, profile_key, run_idx, answer)

                        # Update the progress bar after processing each answer
                        pbar.update(1)

    def _get_seed_values(self) -> list:
        """Return the seed values to be used in the experiment."""
        return self.parameters.seeds if self.parameters.seeds else [42]

    def _invoke_chain(self, chain: Any, input_dict: Dict[str, Any], run_idx: int) -> Optional[Any]:
        """Invoke the chain and handle any errors."""
        try:
            return chain.invoke(input_dict)
        except Exception as e:
            print(f"Error invoking chain for run {run_idx}: {e}")
            return None

    def _store_answer(self, instruction_item: Any, model_key: str, profile_key: str, run_idx: int, answer: Any) -> None:
        """Store the answer in the appropriate location."""
        if answer is not None:
            instruction_item.answers[model_key][profile_key][run_idx] = answer

    def run(self) -> None:
        """Run the experiment processing with a progress bar."""
        with tqdm(total=self._calculate_total_iterations(), desc=self.name or "Experiment", unit=" prompts") as pbar:
            self.process_single_experiment(pbar)

    def _calculate_total_iterations(self) -> int:
        """Calculate the total number of iterations for the progress bar."""
        return (
            len(self.questionnaire.instruction_items)
            * len(self.runnable_models)
            * len(self.demographic_profiles)
            * len(self._get_seed_values())
        )
