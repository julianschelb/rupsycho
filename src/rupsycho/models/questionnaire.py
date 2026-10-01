# ===========================================================================
#                       Data Model: Questionnaire
# ===========================================================================
# This file contains the data model for a psychological questionnaire.

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from tabulate import tabulate


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

    age: Any | None = None
    title: Any | None = None
    name: Any | None = None

    model_config = ConfigDict(extra="allow")


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

    def get_profile_desc(self):
        """Returns a string representation of the profile."""
        return self.__str__()

    model_config = ConfigDict(extra="allow")


class AnswerOption(BaseModel):
    """
    Represents an answer option in a psychological test.
    """

    text: str = "Choose an option"
    ignored_for_scale: bool = False
    weight: int = 0


class AnswerOptions(BaseModel):
    """
    Represents a collection of answer options in a psychological test along with a delimiter for joining them.
    """

    options: dict[str, AnswerOption] = Field(
        default_factory=dict, description="A dictionary of answer options."
    )
    delimiter: str = Field(
        default=", ",
        description="The delimiter used to join the answer options when displayed as a string. Default is ', '. Insert a line break '\\n' for a new line.",
    )
    prepend_delimiter: bool = Field(
        default=False,
        description="If True, the delimiter will be added before the first answer option as well.",
    )

    def join_options(self) -> str:
        """Join the answer options' text using the specified delimiter."""
        option_texts = [option.text for option in self.options.values()]

        # If prepend_delimiter is True, add the delimiter before the first option
        if self.prepend_delimiter and option_texts:
            return self.delimiter + self.delimiter.join(option_texts)
        else:
            return self.delimiter.join(option_texts)

    def get_options_as_list(self) -> list[str]:
        """Return the list of answer options' text."""
        return [option.text for option in self.options.values()]


def _to_answer_options(value: Any) -> Any:
    """Convert the configuration forms of answer options into an ``AnswerOptions`` model.

    Accepted forms: a ready ``AnswerOptions``; ``{"options": {...}, "delimiter": ..,
    "prepend_delimiter": ..}``; or just ``{id: option}``, where an option is a dictionary or an
    ``AnswerOption``. Anything else is passed on for pydantic to validate (and reject).
    """
    if not isinstance(value, dict):
        return value
    if "options" in value:
        options = {key: AnswerOption.model_validate(opt) for key, opt in value["options"].items()}
        return AnswerOptions(
            options=options,
            delimiter=value.get("delimiter", ", "),
            prepend_delimiter=value.get("prepend_delimiter", False),
        )
    return AnswerOptions(
        options={key: AnswerOption.model_validate(opt) for key, opt in value.items()}
    )


class InstructionItem(BaseModel):
    """
    Represents a single question item in a psychological test, along with its answer options and specific attributes.
    """

    question: str = "Enter question text here"
    reversed: bool = False
    answer_options: AnswerOptions | None = None
    attributes: dict = Field(
        default_factory=dict,
        description="Additional attributes related to the question, such as its dimension in a multi-dimensional test structure.",
    )
    answers: dict[Any, dict[Any, dict[Any, str]]] | None = Field(  # type: ignore[assignment]
        default_factory=lambda: defaultdict(lambda: defaultdict(dict)),
        description="A nested dictionary storing answers indexed by model, profile, and run.",
    )

    @field_validator("answer_options", mode="before")
    @classmethod
    def ensure_answer_options_is_proper_model(cls, v: Any) -> Any:
        """Convert a dictionary to an AnswerOptions model if necessary."""
        return _to_answer_options(v)

    def update_answer(self, model_key: str, profile_key: str, run_idx: int, answer: Any) -> None:
        """Store the answer in the appropriate location."""
        if answer is None:
            return
        if self.answers is None:
            self.answers = {}
        # setdefault keeps this working for plain dicts, e.g. after loading an exported experiment
        self.answers.setdefault(model_key, {}).setdefault(profile_key, {})[run_idx] = answer

    def get_answer(self, model_key: str, profile_key: str, run_idx: int) -> Any:
        """Retrieve the answer from the appropriate location."""
        assert self.answers is not None
        return self.answers[model_key][profile_key][run_idx]

    def get_all_answers(self) -> dict[str, dict[str, dict[int, Any]]]:
        """Retrieve all answers."""
        return self.answers  # type: ignore[return-value]

    def get_answer_options_as_list(self) -> list[str]:
        """Return the list of answer options' text."""
        if self.answer_options:
            return self.answer_options.get_options_as_list()
        return []


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
    demographic_profiles: list[DemographicProfile] | None = None
    attributes: dict = Field(
        default_factory=dict,
        description="Additional attributes related to the question, such as its dimension in a multi-dimensional test structure.",
    )
    # Dict[str, AnswerOption]
    default_answer_options: AnswerOptions | None = None
    instruction_items: list[InstructionItem] | None = None

    @field_validator("default_answer_options", mode="before")
    @classmethod
    def ensure_default_answer_options_is_proper_model(cls, v: Any) -> Any:
        """Convert a dictionary to an AnswerOptions model if necessary."""
        return _to_answer_options(v)

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

        if self.default_answer_options:
            print("\nDefault Answer Options:")
            answer_options_table = [
                [key, opt.text, opt.ignored_for_scale, opt.weight]
                for key, opt in self.default_answer_options.options.items()
            ]
            print(
                tabulate(
                    answer_options_table,
                    headers=["ID", "Text", "Ignored for Scale", "Weight"],
                )
            )

        if self.instruction_items:
            print("\nInstruction Items:")
            for item in self.instruction_items:
                print(f"- Question: {item.question}")
                print("  Answer Options:")
                if item.answer_options:
                    answer_options_table = [
                        [opt.text, opt.ignored_for_scale, opt.weight]
                        for opt in item.answer_options.options.values()
                    ]
                    print(
                        tabulate(
                            answer_options_table,
                            headers=["Text", "Ignored for Scale", "Weight"],
                        )
                    )
                # print("  Attributes:")
                # print(f"    {item.attributes}")
                print()
