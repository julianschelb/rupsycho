# ===========================================================================
#                           Data Model: Instruction
# ===========================================================================
# This file contains the data model for the instruction items.


from typing import Dict
from pydantic import BaseModel


class AnswerOption(BaseModel):
    text: str
    weight: int
    reversed: bool
    ignored_for_scale: bool = False  # Optional, default to False if not provided


class InstructionItem(BaseModel):
    question: str
    answer_options: Dict[str, AnswerOption]
