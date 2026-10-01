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
            cleaned_text = text.replace("\n", " ").replace("\r", " ")

            # Map typographic quotes/apostrophes and no-break spaces to ASCII before dropping the rest
            cleaned_text = cleaned_text.translate(
                {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x00A0: " "}
            )

            # Remove non-ASCII Unicode characters
            cleaned_text = re.sub(r"[^\x00-\x7F]+", "", cleaned_text)

            # Trim extra whitespace
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            return cleaned_text

        except Exception as e:
            raise OutputParserException(f"BasicParser encountered an error: {e}") from e

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
        super().__init__(prompt=prompt, similarity_threshold=similarity_threshold, fast=fast)
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
                fast=self.fast,
            )
            return cleaned_completion["completion"]

        except Exception as e:
            raise OutputParserException(
                f"PromptRemovalCleanerParser encountered an error: {e}"
            ) from e

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


class RegexExtractorCleaner(BaseOutputParser[str]):
    """
    A custom parser that extracts values from the input text based on a given regex pattern.
    If the extraction fails, it simply returns the original input text.
    """

    pattern: str = Field(...)

    def __init__(self, pattern: str):
        """
        Initializes the RegexExtractorCleaner with the specified regex pattern.

        Parameters
        ----------
        pattern : str
            The regex pattern used to extract values from the input text.
        """
        super().__init__(pattern=pattern)
        self.pattern = pattern

    def parse(self, text: str) -> str:
        """
        Parses the input text to extract values based on the regex pattern.

        If the regex match fails, returns the original input text.

        Parameters
        ----------
        text : str
            The input string to be processed.

        Returns
        -------
        str
            The extracted value based on the regex pattern, or the original input text
            if extraction fails.

        Raises
        ------
        OutputParserException
            If an error occurs during parsing, an OutputParserException is raised with a
            descriptive error message.
        """
        try:
            # Search for the regex pattern in the input text
            match = re.search(self.pattern, text)

            # If a match is found, return the extracted value
            if match:
                return match.group(1)
            else:
                # If no match, return the original input text
                return text

        except Exception as e:
            raise OutputParserException(f"RegexExtractorCleaner encountered an error: {e}") from e

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.

        Returns
        -------
        str
            The string "regex_extractor_cleaner", identifying the type of this parser.
        """
        return "regex_extractor_cleaner"
