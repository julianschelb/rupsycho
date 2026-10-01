# ===========================================================================
#                            Output Parser Class
# ===========================================================================
# This module defines a custom output parser class for processing text.
# The parser is designed to remove line breaks and non-ASCII Unicode characters
# from a given string, returning a cleaned version of the input.

import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class BasicParser(BaseOutputParser[str]):
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
