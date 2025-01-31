# ===========================================================================
#                              Judges Parser Classes
# ===========================================================================
# This module defines a specialized parser class called Judges, designed to
# evaluate and process input answers. The Judges parser takes in a textual
# answer and applies a series of checks and transformations to determine a
# final prediction or verdict.

from pydantic import Field
from typing import Any
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from rupsycho.parsers.parser_utils import check_multiple_choice_answers
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.stats import entropy
import numpy as np
import torch


# ========================== Multiple Choice Parser ========================


class MultipleChoiceJudge(BaseOutputParser[str]):
    """
    A custom parser that processes a text input and returns the most likely
    answer from a set of possible multiple-choice answers using the
    check_multiple_choice_answers function.

    This parser is designed to work with LangChain and can be integrated
    into various chains or agents that require decision-making based on
    textual analysis.
    """

    possible_answers: list[str] = Field(...)
    ignore_case: bool = Field(True)

    def __init__(self, possible_answers: list[str], ignore_case: bool = True):
        """
        Initializes the parser with a list of possible answers.

        Parameters
        ----------
        possible_answers : list[str]
            A list of possible answers to be considered in the parsing process.
        """
        super().__init__()
        object.__setattr__(self, 'possible_answers', possible_answers)
        object.__setattr__(self, 'ignore_case', ignore_case)

    def parse(self, text: str, possible_answers=None) -> str:
        """
        Parses the input text to determine the most likely answer from the
        possible answers.
        """
        try:
            if possible_answers is None or not possible_answers:
                possible_answers = self.possible_answers
                if not possible_answers:
                    raise ValueError("No possible answers provided")

            # Use the check_multiple_choice_answers function to analyze the text
            results = check_multiple_choice_answers(
                text, possible_answers, self.ignore_case)
            max_value = max(results.values())

            if max_value == 0:
                return "not present"
            else:
                # Find all options with the max_value
                max_keys = [key for key, value in results.items()
                            if value == max_value]
                if len(max_keys) > 1:
                    return "inconclusive"
                else:
                    return max_keys[0]

        except Exception as e:
            raise OutputParserException(
                f"MultipleChoiceJudge encountered an error: {e}")

    @property
    def _type(self) -> str:
        """
        Returns the type of the parser as a string identifier.
        """
        return "multiple_choice_parser"


# ---------------------- Model-Based Answer Judge ---------------------

class ModelBasedAnswerJudge(BaseOutputParser[str]):
    """
    A custom parser that processes a text input and returns the most likely
    answer from a set of possible answers using a Hugging Face model.
    If the entropy of the decision probabilities is greater than a threshold, it returns "inconclusive".
    """

    model_name: str = Field(...)
    possible_answers: list[str] = Field(...)
    device: str = Field('cuda:0')
    model: Any = Field(...)
    tokenizer: Any = Field(...)
    entropy_threshold: float = Field(0.359)  # Default entropy threshold

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_name: str, possible_answers: list[str], device: str = 'cuda:0', entropy_threshold: float = 0.359):
        """
        Initializes the parser with a Hugging Face model and a list of possible answers.

        Parameters
        ----------
        model_name : str
            The Hugging Face model name or path for the sequence classification model.
        possible_answers : list[str]
            A list of possible answers to be considered during prediction.
        device : str
            The device to run the model on (e.g., 'cuda:0' for GPU or 'cpu').
        entropy_threshold : float
            The threshold for entropy above which the result is considered inconclusive.
        """
        super().__init__()

        # Load the tokenizer and model from Hugging Face
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=2)

        # Move model to the specified device
        model.to(device)

        # Manually set the field
        object.__setattr__(self, 'model_name', model_name)
        object.__setattr__(self, 'possible_answers', possible_answers)
        object.__setattr__(self, 'device', device)
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'tokenizer', tokenizer)
        object.__setattr__(self, 'entropy_threshold', entropy_threshold)

    def calculate_entropy(self, decision_list):
        """
        Calculates entropy from a list of decision probabilities.
        """
        probabilities = np.array(
            [item.get('positive_probability', 0.0) for item in decision_list])
        probabilities /= np.sum(probabilities) if np.sum(
            probabilities) > 0 else 1e-12
        return entropy(probabilities, base=2)

    def predict_answer(self, answer_option: str, answer: str):
        """
        Predicts the probability and label for a given answer option.
        """
        self.model.eval()
        inputs = self.tokenizer(
            answer_option,
            answer,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        positive_probability = probabilities[0][1].item()

        return predicted_label, positive_probability

    def predict_for_all_options(self, answer: str, answer_options=None):
        """
        Iterates over all answer options and predicts for each.
        """

        # Use the default possible answers if none are provided
        if answer_options is None or not answer_options:
            answer_options = self.possible_answers
            if not answer_options:
                raise ValueError("No possible answers provided")

        results = []
        for answer_option in answer_options:
            predicted_label, positive_probability = self.predict_answer(
                answer_option, answer)
            results.append({
                'answer_option': answer_option,
                'predicted_label': predicted_label,
                'positive_probability': positive_probability
            })
        return results

    def parse(self, text: str, possible_answers=None) -> str:
        """
        Parses the input text to determine the most likely answer from
        the possible answers or returns "inconclusive" if the entropy is above the threshold.
        """
        try:
            results = self.predict_for_all_options(text, possible_answers)

            # Calculate entropy based on the decision probabilities
            entropy = self.calculate_entropy(results)
            if entropy is not None and entropy > self.entropy_threshold:
                return "inconclusive"

            # Sort results by probability of the positive class
            best_option = max(results, key=lambda x: x['positive_probability'])
            return best_option['answer_option']
        except Exception as e:
            raise OutputParserException(
                f"ModelBasedAnswerJudge encountered an error: {e}")

    @property
    def _type(self) -> str:
        return "model_based_answer_parser"


if __name__ == "__main__":

    # Define possible answers
    possible_answers_1 = ["1. strongly disagree",
                          "2. somewhat agree", "3. agree"]
    possible_answers_2 = ["A. strongly disagree",
                          "B. somewhat agree", "C. agree"]

    # Instantiate the custom parser
    parser_1 = MultipleChoiceJudge(possible_answers_1)
    parser_2 = MultipleChoiceJudge(possible_answers_2)

    # Example text to parse
    text = "I think I would choose option 1 because it seems the best. Also, I somewhat disagree with option 2."

    # Parse the output using both sets of possible answers
    result_1 = parser_1.invoke(text)
    print(f'Most likely answer with possible answers 1: {result_1}')

    result_2 = parser_2.invoke(text)
    print(f'Most likely answer with possible answers 2: {result_2}')
