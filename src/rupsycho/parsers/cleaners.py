# ===========================================================================
#                              Cleaner Parser Classes
# ===========================================================================
# This module defines a specialized parser class called Cleaner, designed to
# process and clean textual answers by removing artifacts that may have been
# introduced during the text generation process.


import re
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import Field
from rupsycho.parsers.parser_utils import prompt_cleaner


class BasicCleaner(BaseOutputParser[str]):
    """
    A custom parser that processes and cleans text by removing line breaks
    and non-ASCII Unicode characters.

    This parser is designed to work with LangChain and can be integrated
    into various chains or agents that require cleaned text output.
    """

    def parse(self, text: str) -> str:
        """
        Parses the input text to remove line breaks and non-ASCII Unicode characters.

        Parameters
        ----------
        text : str
            The input string to be cleaned.

        Returns
        -------
        str
            The cleaned string with line breaks and non-ASCII Unicode characters removed.

        Raises
        ------
        OutputParserException
            If an error occurs during parsing, an OutputParserException is raised with a
            descriptive error message.
        """
        try:
            # Remove line breaks and replace with a space
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ')

            # Remove non-ASCII Unicode characters
            cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)

            # Trim extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            return cleaned_text

        except Exception as e:
            raise OutputParserException(
                f"BasicParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.

        Returns
        -------
        str
            The string "basic_parser", identifying the type of this parser.
        """
        return "basic_parser"


# ================================= Prompt Removal ================================


class PromptRemovalCleaner(BaseOutputParser[str]):
    """
    A custom parser that removes prompt-related text from a completion.
    It uses the prompt_cleaner function to process and clean the input text.
    """

    prompt: str = Field(...)
    similarity_threshold: float = Field(0.7)
    fast: bool = Field(True)

    def __init__(self, prompt: str, similarity_threshold: float = 0.7, fast: bool = True):
        """
        Initializes the PromptRemovalCleanerParser with the specified prompt
        and parameters for cleaning the completion.

        Parameters
        ----------
        prompt : str
            The prompt text that should be removed or considered for cleaning from the completion.
        similarity_threshold : float, optional
            The similarity threshold for the prompt_cleaner function (default is 0.7).
        fast : bool, optional
            A flag to indicate whether the cleaning process should be fast (default is True).
        """
        super().__init__()
        self.prompt = prompt
        self.similarity_threshold = similarity_threshold
        self.fast = fast

    def parse(self, completion: str) -> str:
        """
        Cleans the input completion text by removing prompt-related content.

        Parameters
        ----------
        completion : str
            The input text (completion) that needs to be cleaned.

        Returns
        -------
        str
            The cleaned completion text.

        Raises
        ------
        OutputParserException
            If an error occurs during parsing, an OutputParserException is raised with a
            descriptive error message.
        """
        try:
            # Use the prompt_cleaner function to clean the completion text
            cleaned_completion = prompt_cleaner(
                prompt=self.prompt,
                completion=completion,
                similarity_threshold=self.similarity_threshold,
                fast=self.fast
            )
            return cleaned_completion

        except Exception as e:
            raise OutputParserException(
                f"PromptRemovalCleanerParser encountered an error: {e}")

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.

        Returns
        -------
        str
            The string "prompt_removal_cleaner_parser", identifying the type of this parser.
        """
        return "prompt_removal_cleaner_parser"


if __name__ == "__main__":
    # Example usage of the BasicParser class.

    # Instantiate the custom parser
    basic_parser = BasicCleaner()

    # Example text to parse
    raw_output = "Hello, world!\nThis is a test text with some emojis 😊 and line breaks.\n"

    # Parse the output
    parsed_output = basic_parser.parse(raw_output)

    # Print the cleaned output
    print("Cleaned output:", parsed_output)

    # Example prompt and completion
    prompt_example = "Once upon a time, there was a brave knight."
    completion_example = "Once upon a time, there was a brave knight. He fought valiantly against the dragon."

    # Instantiate the PromptRemovalCleanerParser
    prompt_cleaner_parser = PromptRemovalCleaner(prompt=prompt_example)

    # Parse the completion to clean it from the prompt
    cleaned_completion = prompt_cleaner_parser.parse(completion_example)
    print(f'Cleaned completion: {cleaned_completion}')
