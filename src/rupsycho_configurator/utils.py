"""
This module defines utility methods that are used across the project.
"""

import re
import ast
import json
import typing
from pypdf import PdfReader
import traceback
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import SystemMessage, HumanMessage


def extract_quest_text(file: str | typing.BinaryIO, page_numbers: list[int] = None) -> str:
    """Return the text from the specified pages of the given PDF as a single string.

    Args:
        file (str | typing.BinaryIO): The PDF as either a File object or the path to the PDF file as a string.
        page_numbers (list[int]): List of the pages from which to extract the text from. Page numbers are counted starting at 1. By default all pages are used. 

    Returns:
        str: The extracted text as a string.
    """
    reader = PdfReader(file)
    pages = [page.extract_text() for page in reader.pages]
    number_of_pages = len(pages)
    page_numbers = (range(1, number_of_pages+1)) if not page_numbers else page_numbers
    pdf_text = ''
    
    for p in page_numbers:
        try:    
            pdf_text += pages[p - 1] + '\n'
        except Exception:
            print(traceback.format_exc())
            print(f'{p} / index {p-1} is an invalid page number')
    
    pdf_text = rmv_special_chars(pdf_text) # to avoid conflicts with JSON formatting
    return pdf_text


def extract_quest_pages(file: str | typing.BinaryIO) -> list[str]:
    """Return the text of the given PDF as a list that contains the text of each page as an individual element.
    Args:
        file (str | typing.BinaryIO): The PDF as either a File like object or the path to the PDF file as a string.
    Returns:
        list[str]: A list containing the text of each page as an individual element. The order of the strings in the list is the same as the order of the pages they represent.
    """
    reader = PdfReader(file)
    pages = [page.extract_text() for page in reader.pages]
    pages = [rmv_special_chars(page) for page in pages]
    return pages


def rmv_special_chars(s: str):
    """Remove all occurrences of double quotes, backslash and slash from the given string and return the cleaned string."""
    s = re.sub('["/\\\]', '', s) # 
    return s


def list_pretty_print(lst: list[str], name_of_list='list'):
    """Print the given list in an easy to read format."""
    print(f'\n{name_of_list} = [')
    for i in range(len(lst)):
        line = f'    {i+1} - {lst[i]}'
        # line = line + ',' if i != len(lst)-1 else line # adds commas
        print(line)
    print(']\n')


def extract_json(output: AIMessage | str) -> any:
    """Extract the first JSON instance that is contained in the given language model output and return it as an object.

    Args:
        output (AIMessage): Output string in JSON format that contains an array on the outer most layer. 
            The JSON is expected to be wrapped in "\`\`\`json" and "\`\`\`" tags.
    
    Returns:
        any: A lists containing the deserialized contents of the given JSON array. 
            May return None or a dictionary if the input does not follow the specified format.

    Raises:
        ValueError: If the found JSON instance is not valid JSON or if there is no JSON wrapper.
    """
    if not isinstance(output, str):
        output = output.content
    # print(f"------------------ raw model output ---------------\n{output}")

    json_pattern = r"\`\`\`json(.*?)\`\`\`"

    match = re.search(json_pattern, output, re.DOTALL) # get leftmost match of pattern
    if match:
        try:
            json_str = match.group(1) # only retrieves match of capturing group
            return json.loads(json_str)
        except Exception:
            # raise ValueError(f"Error: Invalid JSON:\n {json_str}")
            raise ValueError(f"Error: Invalid JSON")
    else:
        raise ValueError(f"Error: No JSON wrapper")


def extract_json_array(txt: str, nested: bool) -> list[str] | list[list[str]]:
    """Return the list representation of the leftmost substring that syntactically correctly represents a JSON array of 
    strings or array of arrays of strings.
    
    Args:
        txt (str): A String that contains a syntactically correct list of strings. 
            Contains no double or single quotes other than the delimiters of the strings.
        nested (bool): True if the input contains a JSON array of arrays of strings. False if it is an array of strings.
        
    Returns:
        list[str] | list[list[str]]: Deserialized Json in the form of a list. If no match was found, the empty list is returned.
    """
    txt = re.sub('#[^\n]*\n', '', txt) # remove all comments

    start = txt.find('[')
    while start != -1:
        depth = 0
        quote_char = None
        escape = False

        for end in range(start, len(txt)):
            char = txt[end]

            if quote_char is not None:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == quote_char:
                    quote_char = None
                continue

            if char in ('"', "'"):
                quote_char = char
            elif char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    list_string = txt[start:end + 1]
                    try:
                        parsed = ast.literal_eval(list_string)
                    except (SyntaxError, ValueError):
                        break

                    if nested:
                        if isinstance(parsed, list) and all(
                            isinstance(item, list) and all(isinstance(value, str) for value in item)
                            for item in parsed
                        ):
                            return parsed
                    elif isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                        return parsed

                    break

        start = txt.find('[', start + 1)

    return []


def mk_prompt(quest_text: str) -> ChatPromptValue: 
    """Return a single zero shot prompt to transform the given questionnaire into a json structure of 4 sub-results.
    
    This prompt serves the purpose of getting a language model to convert a given questionnaire into a
    predefined simple json structure. This structure can then be further parsed to fit into the 
    more intricate R.U.Psycho experiment configuration json structure.

    The output of the prompt is expected to have the following structure:
    ```
    [
        "Title of the questionnaire",
        "Instructions on how to fill out the questionnaire",
        [
            "Question 1",
            "Question 2",
            "etc."
        ],
        [
            ["Answer option 1 of questions 1", "Answer option 2 of questions 1", "etc."],
            ["Answer option 1 of questions 2", "Answer option 2 of questions 2", "etc."]
        ]
    ]
    ```
    Json instances are expected to be enclosed in "\`\`\`json" and "\`\`\`" tags in the output.
    """

    basic_instr = """You are a helpful assistant that retrieves information from a given text in a structured manner. You only answers in JSON. Make sure to wrap the answer in \`\`\`json and \`\`\` tags.
The given text is a questionnaire that contains a numbered list of questions or statements.
These questions or statements can be answered with a set of answer options.
There is either only one single universal set of answer options for the entire questionnaire or each of the questions or statements has its own individual set of answer options.\n"""

    complete_instr = """Create a JSON array that contains the following four elements:
The first element of the JSON array should be a string with the title of the questionnaire.
The second element of the JSON array should be a string with the general instructions to the reader on how to fill out the questionnaire.
The third element of the JSON array should be a JSON array of strings that contains each of the questions or statements as a string.
Retain the order that the questions or statements have in the text regardless of their numbering.
The fourth element of the JSON array should be a JSON array of arrays that contains all sets of answer options as arrays of strings.
Retain the order that the sets of answer options have in the text.
If there is only one single universal set of answer options for all questions or statements then only use this.
If a set of answer options is enumerated then include these numbers in the answer options.\n"""

    prompt_str = basic_instr + complete_instr + """The following text is the questionnaire that you should convert into such a JSON array:\n"""

    prompt = ChatPromptValue(messages=[
        SystemMessage(content=prompt_str),
        HumanMessage(content=quest_text)
    ])
    return prompt


if __name__ == "__main__":
    pass