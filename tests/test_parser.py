import pytest
import rupsycho as rup
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner
from rupsycho.parsers.judges import MultipleChoiceJudge, ModelBasedAnswerJudge
from rupsycho.parsers.validators import (
    ApologiesValidatorParser,
    BeingAiValidatorParser,
    RefusalValidatorParser,
    ValidatorParser,
    ModelBasedValidator
)

def test_basic_cleaner():
    cleaner = BasicCleaner()
    raw_text = "Hello, world!\nThis is a test text with some emojis 😊 and line breaks.\n"
    expected_output = "Hello, world! This is a test text with some emojis and line breaks."
    assert cleaner.parse(raw_text) == expected_output


def test_apologies_validator():
    validator = ApologiesValidatorParser()
    response = "I apologize for the inconvenience."
    result = validator.parse(response)
    assert result.get("validation_status", None) == "invalid"


def test_being_ai_validator():
    validator = BeingAiValidatorParser()
    response = "As an AI, I do not have personal opinions."
    result = validator.parse(response)
    assert result.get("validation_status", None) == "invalid"


def test_refusal_validator():
    validator = RefusalValidatorParser()
    response = "I cannot provide that information."
    result = validator.parse(response)
    assert result.get("validation_status", None) == "invalid"


def test_combined_validator():
    validator = ValidatorParser()
    response = "I apologize for any confusion, but as an AI, I cannot provide that information."
    result = validator.parse(response)
    assert result.get("validation_status", None) == "invalid"


def test_multiple_choice_judge():
    possible_answers = ["1. strongly disagree",
                        "2. somewhat agree", "3. agree"]
    judge = MultipleChoiceJudge(possible_answers)
    response = "I think I would choose option 1 because I strongly disagree."
    assert judge.invoke(response) == "1. strongly disagree"


def test_model_based_answer_judge():
    repo_id = "julian-schelb/rup-answer-option-likert-scale"
    possible_answers = ["1. never or seldom",
                        "2.", "3. sometimes", "4.", "5. always"]
    judge = ModelBasedAnswerJudge(
        possible_answers=possible_answers, model_name=repo_id, device="cpu")
    response = "I always do my best to be honest: 5."
    assert judge.parse(response) == "5. always"


def test_model_based_validator():
    repo_id = "protectai/distilroberta-base-rejection-v1"
    validator = ModelBasedValidator(repo_id, device="cpu")
    response = "Sorry, but I can't assist with that."
    result = validator.parse(response)
    assert result.get("validation_status", None) == "invalid"
