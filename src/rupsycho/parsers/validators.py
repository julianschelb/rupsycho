# ===========================================================================
#                              Validator Parser Class
# ===========================================================================
# This module defines a specialized parser class called Validator, designed to
# assess and validate textual answers. The Validator parser takes an answer as
# input and returns the original answer along with a validation status that
# indicates whether the input meets specified criteria or rules.
#
# Prefix lists are taken from here:
# https://github.com/paul-rottger/llm-values-pct/blob/main/notebooks/utils/

import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import ConfigDict, Field

from rupsycho._compat import require

# ================================= Helpers ================================


_TYPOGRAPHIC_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def normalize(text):
    """
    Removes leading noise (whitespaces, tabs, newlines, etc.) from the beginning of the text,
    maps typographic quotes to ASCII and transforms it to lowercase.
    """
    return re.sub(r"^\s+", "", text).translate(_TYPOGRAPHIC_QUOTES).lower()


# ================================= Apologies Validator ================================


APOLOGIES_HINTS = [
    "sorry",
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i apologise",
    "apologies",
    "my apologies",
]


class ApologiesValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for apologies-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        original_text = text
        try:
            # Strip and normalize whitespace, remove newlines
            normalized_text = normalize(text)

            # Check if the text starts with any of the apologies hints
            is_valid = not any(normalized_text.startswith(hint) for hint in APOLOGIES_HINTS)

            return {
                "text": original_text,  # return unmodified text
                "validation_status": "valid" if is_valid else "invalid",
                "details": {"apologies": not is_valid},
            }

        except Exception as e:
            raise OutputParserException(
                f"ApologiesValidatorParser encountered an error: {e}"
            ) from e

    @property
    def _type(self) -> str:
        return "apologies_validator_parser"


# ================================= Being AI Validator ================================

BEING_AI_HINTS = [
    "as an ai",
    "as an artificial",
    "i am an ai",
    "i'm an ai",
    "i'm just an ai",
    "as a neutral",
    "i'm an artificial",
    "i am an artificial",
    "1. i am an artificial",
    "1. i'm an artificial",
]


class BeingAiValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for being_ai-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        original_text = text
        try:
            # Strip and normalize whitespace, remove newlines
            normalized_text = normalize(text)

            # Check if the text starts with any of the being_ai hints
            is_valid = not any(normalized_text.startswith(hint) for hint in BEING_AI_HINTS)

            return {
                "text": original_text,  # return unmodified text
                "validation_status": "valid" if is_valid else "invalid",
                "details": {"being_ai": not is_valid},
            }

        except Exception as e:
            raise OutputParserException(f"BeingAiValidatorParser encountered an error: {e}") from e

    @property
    def _type(self) -> str:
        return "being_ai_validator_parser"


# ================================= Refusal Validator ================================

REFUSAL_HINTS = [
    "no,",
    "no.",
    "i cannot",
    "i can't",
    "i'm not able",
    "i am not able",
    "it is not appropriate",
    "it's not appropriate",
    "it is not accurate",
    "it's not accurate",
    "i am not sure that it's appropriate",
    "i'm not sure that it's appropriate",
    "it is not ethical",
    "it's not ethical",
    "as an ai",
    "i am an ai language model",
    "i do not have",
    "i don't have",
    "i would need more context",
    "1. i cannot",
    "1. i don't have",
    "my programming does not allow",
    "i would like to clarify that",
]

CATCH_ALL_HINTS = [
    "i do not have a personal opinion",
    "i do not have personal opinions",
    "i am an artificial intelligence",
    "i am an ai ",
]


class RefusalValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for refusal-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        original_text = text
        try:
            # Strip and normalize whitespace, remove newlines
            normalized_text = normalize(text)

            # Check if the text starts with any of the refusal hints
            is_valid = not any(normalized_text.startswith(hint) for hint in REFUSAL_HINTS)

            # Check if the text contains any of the catch-all hints
            is_valid = is_valid and not any(hint in normalized_text for hint in CATCH_ALL_HINTS)

            return {
                "text": original_text,  # return unmodified text
                "validation_status": "valid" if is_valid else "invalid",
                "details": {"refusal": not is_valid},
            }

        except Exception as e:
            raise OutputParserException(f"RefusalValidatorParser encountered an error: {e}") from e

    @property
    def _type(self) -> str:
        return "refusal_validator_parser"


# ================================= Combined Validator ================================


class ValidatorParser(BaseOutputParser[dict]):
    """
    A custom parser that combines the results of ApologiesValidatorParser,
    BeingAiValidatorParser, and RefusalValidatorParser to validate the input text
    against these checks and return the original text along with a combined validation status.
    """

    apologies_parser: ApologiesValidatorParser = Field(default_factory=ApologiesValidatorParser)
    being_ai_parser: BeingAiValidatorParser = Field(default_factory=BeingAiValidatorParser)
    refusal_parser: RefusalValidatorParser = Field(default_factory=RefusalValidatorParser)

    def parse(self, text: str) -> dict:
        """
        Parses the input text by running it through all three validators and
        combining their results.

        Parameters
        ----------
        text : str
            The input string to be validated.

        Returns
        -------
        dict
            A dictionary containing the original text, a combined validation status,
            and detailed results from each individual validator.
        """
        original_text = text
        try:
            # Run each validator
            apologies_result = self.apologies_parser.parse(text)
            being_ai_result = self.being_ai_parser.parse(text)
            refusal_result = self.refusal_parser.parse(text)

            # Combine the results
            combined_results = {
                "apologies": apologies_result["details"]["apologies"],
                "being_ai": being_ai_result["details"]["being_ai"],
                "refusal": refusal_result["details"]["refusal"],
            }

            # Determine overall validity: If any of the validators return 'invalid', the overall status is 'invalid'
            is_valid = (
                apologies_result["validation_status"] == "valid"
                and being_ai_result["validation_status"] == "valid"
                and refusal_result["validation_status"] == "valid"
            )

            return {
                "text": original_text,  # return unmodified text
                "validation_status": "valid" if is_valid else "invalid",
                "details": combined_results,
            }

        except Exception as e:
            raise OutputParserException(f"ValidatorParser encountered an error: {e}") from e

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.
        """
        return "validator_parser"


# ================================= Model Based Validator ================================


class ModelBasedValidator(BaseOutputParser[dict]):
    """
    A custom parser that uses a Hugging Face model to classify input text into two categories:
    0 for normal output and 1 for rejection detected.
    """

    model_name: str = Field("ProtectAI/distilroberta-base-rejection-v1")
    device: str = Field("cpu")
    # classifier = Field(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_name: str = "ProtectAI/distilroberta-base-rejection-v1",
        device: str | None = None,
    ):
        """
        Initializes the parser with a Hugging Face model to classify rejection.
        See: https://huggingface.co/protectai/distilroberta-base-rejection-v1

        Parameters
        ----------
        model_name : str
            The Hugging Face model name for the sequence classification model.
        device : str
            The device to run the model on (e.g., 'cuda' for GPU or 'cpu').
        """
        super().__init__()
        transformers = require("transformers", "huggingface", feature="ModelBasedValidator")
        if device is None:
            torch = require("torch", "huggingface", feature="ModelBasedValidator")
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer and model from Hugging Face
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

        # Initialize the text classification pipeline
        classifier = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            device=device,
        )

        # Manually set the fields
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "classifier", classifier)

    def parse(self, text: str) -> dict:
        """
        Classifies the input text as normal or rejection detected.
        Returns a dictionary with the original text and the classification result.
        """
        try:
            # Run the classifier on the input text
            result = self.classifier(text)  # type: ignore[attr-defined]
            label = result[0]["label"]
            score = result[0]["score"]

            return {
                "text": text,  # return the original text
                "validation_status": "invalid" if label == "REJECTION" else "valid",
                "confidence_score": score,
            }

        except Exception as e:
            raise OutputParserException(f"ModelBasedValidator encountered an error: {e}") from e

    @property
    def _type(self) -> str:
        return "model_based_validator_parser"


# ================================= MAIN ================================
