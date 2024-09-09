# ===========================================================================
#                       Data Model: Questionnaire
# ===========================================================================
# This file contains the data model for a psychological questionnaire.

from collections import defaultdict
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class DemographicAttributes(BaseModel):
    """
    Represents the demographic attributes of a participant in a psychological test.
    This model includes default fields such as 'age', 'title', and 'name', but it is
    extendable to include additional demographic details as needed.

    Attributes:
        age (Optional[int]): The age of the participant. Default is None.
        title (Optional[str]): The title of the participant (e.g., Mr, Ms, Dr). Default is None.
        name (Optional[str]): The name of the participant. Default is None.

    The model is flexible to accept additional fields beyond the ones specified.
    """

    age: Optional[int] = None
    title: Optional[str] = None
    name: Optional[str] = None

    class Config:
        extra = "allow"


class DemographicProfile(BaseModel):
    """
    Represents the demographic profile of a participant in a psychological test.
    This includes attributes like age, title, name, and a template for formatting purposes.
    """

    attributes: DemographicAttributes = Field(
        ...,
        description="Attributes containing demographic information such as age, title, and name.",
    )
    template: str = Field(
        default="{title} {name} is {age} years old.",
        description="A template string to format the demographic information. "
        "Use placeholders for attributes (e.g., '{title}', '{name}', '{age}').",
    )

    def __str__(self):
        return self.template.format(**self.attributes.model_dump())

    class Config:
        extra = "allow"


class AnswerOption(BaseModel):
    """
    Represents an answer option in a psychological test.

    Attributes:
        text (str): The text of the answer option. This should describe the option in a way that is clear to the participant.
        ignored_for_scale (bool): Indicates whether this answer option should be ignored when calculating scores on a scale. This can be useful for neutral options or non-applicable responses.
        weight (int): The numerical weight assigned to this answer option. This is typically used in scoring the test, where different options have different point values.

    The default values are set to represent a common scenario in psychological testing. Adjust as necessary for specific test requirements.
    """

    text: str = "Choose an option"
    ignored_for_scale: bool = False
    weight: int = 0


class InstructionItem(BaseModel):
    """
    Represents a single question item in a psychological test, along with its answer options and specific attributes.

    Attributes:
        question (str): The text of the question presented to the participant.
        reversed (bool): A flag indicating whether the scoring for this question should be reversed. In some psychological tests, certain questions are scored in the opposite direction for certain scales.
        answer_options (Optional[List[AnswerOption]]): A list of possible answer options that a participant can choose from in response to the question. This field is optional and can be None if the question does not have predefined answer options.
        attributes (InstructionItemAttribute): Additional attributes related to the question, such as its dimension in a multi-dimensional test structure.

    The default values and structure are designed to be flexible and can be adjusted to suit different types of psychological tests.
    """

    question: str = "Enter question text here"
    reversed: bool = False
    answer_options: Optional[Dict[str, AnswerOption]] = None
    attributes: Dict = Field(
        default_factory=dict,
        description="Additional attributes related to the question, such as its dimension in a multi-dimensional test structure.",
    )
    answers: Optional[Dict[int, Dict[int, Dict[int, str]]]] = Field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))),
        description="A nested dictionary storing answers indexed by model, profile, and run.",
    )


class Questionnaire(BaseModel):
    """
    Represents a questionnaire used in a psychological test.

    Attributes:
        name (str): The name of the questionnaire.
        general_instruction (str): General instructions provided to the participants.
        demographic_profiles (Optional[List[DemographicProfile]]): A list of demographic profiles for the participants. This field is optional.
        default_answer_options (Optional[Dict[str, AnswerOption]]): A dictionary of default answer options for the questionnaire. The keys represent unique identifiers for each option. This field is optional.
        instruction_items (Optional[List[InstructionItem]]): A list of instruction items (questions) included in the questionnaire. This field is optional.

    Methods:
        get_number_of_questions: Returns the number of questions in the questionnaire.
    """

    name: str
    general_instruction: str
    demographic_profiles: Optional[List[DemographicProfile]] = None
    attributes: Dict = Field(
        default_factory=dict,
        description="Additional attributes related to the question, such as its dimension in a multi-dimensional test structure.",
    )
    default_answer_options: Optional[Dict[str, AnswerOption]] = None
    instruction_items: Optional[List[InstructionItem]] = None

    def get_number_of_questions(self) -> int:
        """Returns the number of questions in the questionnaire."""
        if self.instruction_items is None:
            return 0
        return len(self.instruction_items)

    def print_questionnaire(self):
        """Prints the questionnaire in a human-readable format."""

        print(f"Name: {self.name}\n")
        print(f"General Instruction: {self.general_instruction}\n")

        if self.demographic_profiles:
            print("Demographic Profiles:")
            for profile in self.demographic_profiles:
                print(f"- {profile}")

        # if self.default_answer_options:
        #     print("\nDefault Answer Options:")
        #     answer_options_table = [
        #         [key, opt.text, opt.ignored_for_scale, opt.weight]
        #         for key, opt in self.default_answer_options.items()
        #     ]
        #     print(
        #         tabulate(
        #             answer_options_table,
        #             headers=["ID", "Text", "Ignored for Scale", "Weight"],
        #         )
        #     )

        if self.instruction_items:
            print("\nInstruction Items:")
            for item in self.instruction_items:
                print(f"- Question: {item.question}")
                # print("  Answer Options:")
                # if item.answer_options:
                #     answer_options_table = [
                #         [opt.text, opt.ignored_for_scale, opt.weight]
                #         for opt in item.answer_options
                #     ]
                #     print(
                #         tabulate(
                #             answer_options_table,
                #             headers=["Text", "Ignored for Scale", "Weight"],
                #         )
                #     )
                # print("  Attributes:")
                # print(f"    {item.attributes}")
                # print()
