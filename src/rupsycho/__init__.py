"""
RUPsycho: A Python Package for Social Science Research Using Large Language Models (LLMs)
=========================================================================================

RUPsycho is a Python package designed to facilitate the application of large language models (LLMs) 
in social science research. It offers tools to explore human-like behaviors through LLMs, providing 
a novel approach in the field of natural language processing (NLP).

This module within RUPsycho provides functions to load and validate experimental data from JSON 
dictionaries and files. It supports both single and multiple experiment loading, making it easy 
to manage and analyze experiments in various formats.


Functions
---------
- `experiment_from_dict(json_data: dict) -> ExperimentDocument`:
    Load a single experiment from a JSON dictionary.

- `experiment_from_file(path: str) -> ExperimentDocument`:
    Load a single experiment from a JSON file.

- `experiments_from_dicts(json_data_list: List[dict]) -> Iterator[ExperimentDocument]`:
    Load multiple experiments from a list of JSON dictionaries.

- `experiments_from_files(paths: List[str]) -> Iterator[ExperimentDocument]`:
    Load multiple experiments from a list of JSON files.

Usage
-----
- **Single Experiment Loading**:
    - Use `experiment_from_dict` for loading an experiment from an in-memory JSON dictionary.
    - Use `experiment_from_file` for loading an experiment from a JSON file.

- **Multiple Experiments Loading**:
    - Use `experiments_from_dicts` to load multiple experiments from a list of JSON dictionaries.
    - Use `experiments_from_files` to load multiple experiments from a list of JSON files.

Example
-------
After importing RUPsycho, you can start using it in your Python script or Jupyter notebook:

    import rupsycho as rup

    # Load experiment data
    experiment = rup.experiment_from_file("bfi.json")

    # Run the experiment
    experiment.run()

    # Save the results
    experiment.export_to_file("results.json")

For more detailed examples and usage, refer to the official RUPsycho documentation.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from rupsycho.experiment_collection import ExperimentCollection
from rupsycho.reader import ExperimentLoader, ExperimentDocument
from transformers import pipeline
from typing import List, Iterator
from typing import Optional


# ================================= Create Single Experiment ================================


def experiment_from_dict(json_data: dict) -> ExperimentDocument:
    """Create a new experiment from a JSON dictionary."""
    try:
        loader = ExperimentLoader()
        # Use the loader to validate and return the ExperimentDocument
        return next(loader.lazy_load_from_dicts([json_data]))
    except Exception as e:
        raise RuntimeError(
            f"Failed to create experiment from dict: {e}") from e


def experiment_from_file(path: str) -> ExperimentDocument:
    """Create a new experiment from a JSON file."""
    try:
        experiments = ExperimentLoader(path_pattern=path).load()
        return experiments[0] if len(experiments) > 0 else None
    except Exception as e:
        raise RuntimeError(
            f"Failed to create experiment from file {path}: {e}") from e

# ================================= Create Multiple Experiments ================================


def experiments_from_dicts(json_data_list: List[dict]) -> Iterator[ExperimentDocument]:
    """Create multiple experiments from a list of JSON dictionaries."""
    try:
        loader = ExperimentLoader()
        # Use the loader to validate and return ExperimentDocuments
        return loader.lazy_load_from_dicts(json_data_list)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create experiments from list of dicts: {e}") from e


def experiments_from_files(path_pattern: str) -> Iterator[ExperimentDocument]:
    """Create multiple experiments from a list of JSON files."""
    try:
        experiments = ExperimentLoader(path_pattern=path_pattern).load()
        return ExperimentCollection(experiments)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create experiments from files matching {path_pattern}: {e}") from e

# ================================= Create Example Experiments ================================


def example_experiment_bfi(
    model_name: str = "google/flan-t5-small",
    pipeline_type: str = "text2text-generation",
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    api_key: Optional[str] = None
) -> ExperimentDocument:
    """
    Create an example experiment with a specified model, pipeline type, and generative parameters.

    :param model_name: Name of the Hugging Face model to use.
    :param pipeline_type: Type of pipeline to use (e.g., 'text2text-generation').
    :param temperature: Temperature setting for the generative model.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param api_key: Optional Hugging Face API key for accessing gated models.
    :return: Configured ExperimentDocument.
    """

    # Set the Hugging Face API key if provided
    if api_key:
        from huggingface_hub import login
        login(api_key)

    # Example experiment data
    experiment_data = {
        "name": "Generative Models for Big Five Inventory",
        "description": "Description: Testing Generative Models for BFI questionnaire using Rupsycho.",
        "demographic_profiles": {
            "Profile 1": {
                "attributes": {
                    "title": "Mr",
                    "name": "Grueber"
                },
                "template": "{title} {name}"
            }
        },
        "parameters": {
            "seeds": ["7"],
        },
        "questionnaire":  {
            "name": "BIG FIVE INVENTORY RESPONSE FORM AND INSTRUCTIONS TO PARTICIPANTS",
            "general_instruction": "Here are a number of characteristics that may or may not apply to you. For example, do you agree that you are someone who likes to spend time with others? Please return the number corresponding to the answer options to indicate the extent to which you agree or disagree with that statement.",
            "attributes": {
                "dimension": {
                    "1": "Extraversion",
                    "2": "Agreeableness",
                    "3": "Conscientiousness",
                    "4": "Neuroticism",
                    "5": "Openness"
                }
            },
            "default_answer_options": {
                "1": {
                    "text": "1. Disagree strongly",
                    "ignored_for_scale": False,
                    "weight": 1
                },
                "2": {
                    "text": "2. Disagree a little",
                    "ignored_for_scale": False,
                    "weight": 2
                },
                "3": {
                    "text": "3. Neither agree nor disagree",
                    "ignored_for_scale": False,
                    "weight": 3
                },
                "4": {
                    "text": "4. Agree a little",
                    "ignored_for_scale": False,
                    "weight": 4
                },
                "5": {
                    "text": "5. Agree strongly",
                    "ignored_for_scale": False,
                    "weight": 5
                }
            },
            "instruction_items": [
                {
                    "question": "I see myself as someone who...",
                    "reversed": False,
                    "attributes": {
                        "dimension": "1"
                    }
                },
                {
                    "question": "Tends to find fault with others",
                    "reversed": False,
                    "attributes": {
                        "dimension": "1"
                    }
                },
                {
                    "question": "Does a thorough job",
                    "reversed": False,
                    "attributes": {
                        "dimension": "1"
                    }
                },
                {
                    "question": "Is depressed, blue",
                    "reversed": False,
                    "attributes": {
                        "dimension": "1"
                    }
                }
            ]
        }
    }

    # Load a single experiment from the example data
    experiment = experiment_from_dict(experiment_data)

    # Load the specified Hugging Face generative model
    pipe = pipeline(
        pipeline_type,
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    model = HuggingFacePipeline(pipeline=pipe)

    # Add the model to the experiment
    experiment.add_model(model, identifier="hf_generative_model")

    # Create the provided ChatPromptTemplate for the experiment
    system_message_template = """
        Objective: "{general_instruction}"
        Answer with respect to the following persona description and question.
    """
    # Create the user message template
    user_message_template = """ 
        Question:
        {persona_description} was asked the following question. {question}

        Answer Options: 
        {answer_options}

        Instructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct.
        
        Answer:
    """

    # Create the chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message_template),
        ("user", user_message_template)
    ])

    # Set the chat prompt for the experiment
    experiment.set_prompt(chat_prompt)

    # Set a simple string output parser
    parser = StrOutputParser()
    experiment.set_parser(parser)

    # Return the configured experiment
    return experiment
