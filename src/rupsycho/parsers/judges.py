# ===========================================================================
#                              Judges Parser Classes
# ===========================================================================
# This module defines a specialized parser class called Judges, designed to
# evaluate and process input answers. The Judges parser takes in a textual
# answer and applies a series of checks and transformations to determine a
# final prediction or verdict.

from pydantic import Field
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from rupsycho.parsers.parser_utils import check_multiple_choice_answers


# ========================== Multiple Choice Parser ========================


class MultipleChoiceParser(BaseOutputParser[str]):
    """
    A custom parser that processes a text input and returns the most likely
    answer from a set of possible multiple-choice answers using the
    check_multiple_choice_answers function.

    This parser is designed to work with LangChain and can be integrated
    into various chains or agents that require decision-making based on
    textual analysis.
    """

    possible_answers: list[str] = Field(...)

    def __init__(self, possible_answers: list[str]):
        """
        Initializes the parser with a list of possible answers.

        Parameters
        ----------
        possible_answers : list[str]
            A list of possible answers to be considered in the parsing process.
        """
        super().__init__()
        # Manually set the field
        object.__setattr__(self, 'possible_answers', possible_answers)

    def parse(self, text: str) -> str:
        """
        Parses the input text to determine the most likely answer from the
        possible answers.
        """
        try:
            # Use the check_multiple_choice_answers function to analyze the text
            results = check_multiple_choice_answers(
                text, self.possible_answers)

            # Find the answer with the highest count
            most_likely_answer = max(results, key=results.get)

            return most_likely_answer

        except Exception as e:
            raise OutputParserException(
                f"MultipleChoiceParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.
        """
        return "multiple_choice_parser"


if __name__ == "__main__":

    # Define possible answers
    possible_answers_1 = ["1. strongly disagree",
                          "2. somewhat agree", "3. agree"]
    possible_answers_2 = ["A. strongly disagree",
                          "B. somewhat agree", "C. agree"]

    # Instantiate the custom parser
    parser_1 = MultipleChoiceParser(possible_answers_1)
    parser_2 = MultipleChoiceParser(possible_answers_2)

    # Example text to parse
    text = "I think I would choose option 1 because it seems the best. Also, I somewhat disagree with option 2."

    # Parse the output using both sets of possible answers
    result_1 = parser_1.invoke(text)
    print(f'Most likely answer with possible answers 1: {result_1}')

    result_2 = parser_2.invoke(text)
    print(f'Most likely answer with possible answers 2: {result_2}')
