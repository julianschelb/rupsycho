# ===========================================================================
#                              Validator Parser Class
# ===========================================================================
# This module defines a specialized parser class called Validator, designed to
# assess and validate textual answers. The Validator parser takes an answer as
# input and returns the original answer along with a validation status that
# indicates whether the input meets specified criteria or rules.

from pydantic import Field
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from rupsycho.parsers.parser_utils import check_span


# ================================= Apologies Validator ================================

APOLOGIES_HINTS = {
    "apologies": [
        "sorry",
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "i apologise",
        "apologies",
        "my apologies"
    ]
}


class ApologiesValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for apologies-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        try:
            # Perform the validation check using check_span
            apologies_check = check_span(text, APOLOGIES_HINTS)

            # Determine if the text is valid
            is_valid = not any(apologies_check.values())

            return {
                'text': text,
                'validation_status': 'valid' if is_valid else 'invalid',
                'details': apologies_check
            }

        except Exception as e:
            raise OutputParserException(
                f"ApologiesValidatorParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        return "apologies_validator_parser"

# ================================= Being AI Validator ================================


BEING_AI_HINTS = {
    "being_ai": [
        "as an ai",
        "as an artificial",
        "i am an ai",
        "i'm an ai",
        "i'm just an ai",
        "as a neutral",
        "i'm an artificial",
        "i am an artificial",
        "I was created",
        "my programming does not allow",
        "i would like to clarify that",
        "i do not have a personal opinion",
        "i do not have personal opinions",
        "i am an artificial intelligence",
        "i am an ai "
    ]
}


class BeingAiValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for being_ai-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        try:
            # Perform the validation check using check_span
            being_ai_check = check_span(text, BEING_AI_HINTS)

            # Determine if the text is valid
            is_valid = not any(being_ai_check.values())

            return {
                'text': text,
                'validation_status': 'valid' if is_valid else 'invalid',
                'details': being_ai_check
            }

        except Exception as e:
            raise OutputParserException(
                f"BeingAiValidatorParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        return "being_ai_validator_parser"


# ================================= Refusal Validator ================================

REFUSAL_HINTS = {
    "refusal": [
        "no",
        "not",
        "cannot",
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
        "i do not have",
        "i don't have",
        "i would need more context",
        "1. i cannot",
        "1. i don't have"
    ]
}


class RefusalValidatorParser(BaseOutputParser[dict]):
    """
    A parser that validates the input text for refusal-related content.
    Returns the original text along with a validation status.
    """

    def parse(self, text: str) -> dict:
        try:
            # Perform the validation check using check_span
            refusal_check = check_span(text, REFUSAL_HINTS)

            # Determine if the text is valid
            is_valid = not any(refusal_check.values())

            return {
                'text': text,
                'validation_status': 'valid' if is_valid else 'invalid',
                'details': refusal_check
            }

        except Exception as e:
            raise OutputParserException(
                f"RefusalValidatorParser encountered an error: {e}")

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

    apologies_parser: ApologiesValidatorParser = Field(...)
    being_ai_parser: BeingAiValidatorParser = Field(...)
    refusal_parser: RefusalValidatorParser = Field(...)

    def __init__(self):
        """
        Initializes the combined validator parser by creating instances of the
        Apologies, Being AI, and Refusal validators.
        """
        super().__init__()
        self.apologies_parser = ApologiesValidatorParser()
        self.being_ai_parser = BeingAiValidatorParser()
        self.refusal_parser = RefusalValidatorParser()

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
        try:
            # Run each validator
            apologies_result = self.apologies_parser.parse(text)
            being_ai_result = self.being_ai_parser.parse(text)
            refusal_result = self.refusal_parser.parse(text)

            # Combine the results
            combined_results = {
                'apologies': list(apologies_result['details'].values())[0],
                'being_ai': list(being_ai_result['details'].values())[0],
                'refusal': list(refusal_result['details'].values())[0]
            }

            # Determine overall validity: If any of the validators return 'invalid', the overall status is 'invalid'
            is_valid = (
                apologies_result['validation_status'] == 'valid' and
                being_ai_result['validation_status'] == 'valid' and
                refusal_result['validation_status'] == 'valid'
            )

            return {
                'text': text,
                'validation_status': 'valid' if is_valid else 'invalid',
                'details': combined_results
            }

        except Exception as e:
            raise OutputParserException(
                f"ValidatorParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.
        """
        return "validator_parser"


# ================================= MAIN ================================


if __name__ == "__main__":
    # Example texts to validate
    example_texts = {
        "general": "I think I would choose option 1 because it seems the best. Also, I somewhat disagree with option 2.",
        "apologies": "I apologize for the inconvenience.",
        "being_ai": "As an AI, I do not have personal opinions.",
        "refusal": "I cannot provide that information.",
        "combined": "I apologize for any confusion, but as an AI, I cannot provide that information."
    }

    # Instantiate the individual validators
    apologies_parser = ApologiesValidatorParser()
    being_ai_parser = BeingAiValidatorParser()
    refusal_parser = RefusalValidatorParser()

    # Function to validate text with a specific parser
    def validate_text(parser, text_key):
        text = example_texts[text_key]
        result = parser.parse(text)
        print(f'{parser._type.capitalize()} validation result: {result}\n')

    # Validate using individual parsers
    print("Running individual validators:")
    validate_text(apologies_parser, "apologies")
    validate_text(being_ai_parser, "being_ai")
    validate_text(refusal_parser, "refusal")

    # Instantiate the combined ValidatorParser
    validator_parser = ValidatorParser()

    # Validate using the combined parser
    print("Running combined validator:")
    combined_result = validator_parser.parse(example_texts["combined"])
    print(f'Combined validation result: {combined_result}')
