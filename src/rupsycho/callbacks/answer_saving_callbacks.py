# ===========================================================================
#                         AnswerSavingCallbacks Implementation
# ===========================================================================
#  This module defines abstract and concrete callback classes for saving the
#  answers generated during the experiment process. The callbacks provide
#  flexibility to store the results in various formats (e.g., JSONL, databases)
#  by accessing the experiment context, instruction items, model details,
#  demographic profiles, and generated answers. Each callback implements a
#  `save_answer` method, which can be customized based on the storage requirement.
# ===========================================================================

import csv
import json
from abc import ABC, abstractmethod


# --------------------------------- JSONL --------------------------------

class Callback(ABC):
    """
    Abstract base class for implementing custom callbacks for saving answers.
    """

    @abstractmethod
    def save_answer(self, experiment, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        """
        Method to be implemented by subclasses to save the answer.

        Args:
        - experiment: The experiment instance (`self`) being processed.
        - instruction_item_id: The unique ID of the instruction item.
        - instruction_item: The instruction item being processed.
        - model_id: The identifier of the model used for generating the answer.
        - profile_id: The identifier of the demographic profile.
        - random_seed: The random seed used for generating the answer.
        - answer: The generated answer for the instruction item.
        """
        pass


class JSONLCallback(Callback):
    """
    Callback implementation for saving answers to a JSONL file.
    """

    def __init__(self, file_path="experiment_output.jsonl"):
        self.file_path = file_path

    def save_answer(self, experiment, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        output_data = {
            "experiment_name": experiment.name,  # Accessing experiment details
            "instruction_item_id": instruction_item_id,
            # Assuming instruction_item can be converted to dict
            "instruction_item": instruction_item.model_dump(exclude=['answer']),
            "model_id": model_id,
            "profile_id": profile_id,
            "random_seed": random_seed,
            "answer": answer
        }

        # Open the JSONL file in append mode and write the data
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(output_data) + '\n')


# --------------------------------- CSV --------------------------------

class CSVCallback(Callback):
    """
    Callback implementation for saving answers to a CSV file.
    """

    def __init__(self, file_path="experiment_output.csv"):
        self.file_path = file_path

        # Initialize the CSV file with headers if it doesn't exist
        try:
            with open(self.file_path, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["experiment_name", "instruction_item_id", "instruction_item", "model_id",
                                 "profile_id", "random_seed", "answer"])
        except FileExistsError:
            # File already exists, do nothing
            pass

    def save_answer(self, experiment, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        output_data = [
            experiment.name,  # Accessing experiment details
            instruction_item_id,
            instruction_item.question,  # Assuming instruction_item has a 'question' attribute
            model_id,
            profile_id,
            random_seed,
            answer
        ]

        # Open the CSV file in append mode and write the data
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(output_data)


# --------------------------------- Print --------------------------------

class PrintCallback(Callback):
    """
    Callback implementation for printing answers to the screen.
    """

    def save_answer(self, experiment, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        # Accessing experiment and instruction item details
        print(f"Experiment: {experiment.name}")
        print(f"Instruction Item ID: {instruction_item_id}")
        # Assuming instruction_item has a 'question' attribute
        print(f"Instruction Item: {instruction_item.question}")
        print(f"Model ID: {model_id}")
        print(f"Profile ID: {profile_id}")
        print(f"Random Seed: {random_seed}")
        print(f"Answer: {answer}")
        print("=" * 50)  # Divider for clarity between different outputs


# --------------------------------- Table --------------------------------

class PrintTableCallback(Callback):
    """
    Callback implementation for printing answers in a fixed-width, table-like format without line breaks,
    restricting the question and answer to 100 characters each, and printing headers only once.
    """

    def __init__(self):
        # Define column widths for each field
        self.column_widths = {
            "instruction_id": 15,
            "model_id": 20,
            "profile_id": 20,
            "random_seed": 12,
            "question": 25,
            "answer": 25
        }

        # Initialize a flag to track whether headers have been printed
        self.headers_printed = False

    def _print_headers(self):
        """Print the table headers."""
        header = (f"{'Instruction ID':<{self.column_widths['instruction_id']}} | "
                  f"{'Model ID':<{self.column_widths['model_id']}} | "
                  f"{'Profile ID':<{self.column_widths['profile_id']}} | "
                  f"{'Random Seed':<{self.column_widths['random_seed']}} | "
                  f"{'Question (truncated)':<{self.column_widths['question']}} | "
                  f"{'Answer (truncated)':<{self.column_widths['answer']}}")
        print(header)
        print("=" * len(header))  # Separator line
        self.headers_printed = True

    def save_answer(self, experiment, instruction_item_id, instruction_item, model_id, profile_id, random_seed, answer):
        # Print headers the first time save_answer is called
        if not self.headers_printed:
            self._print_headers()

        # Limit question and answer to the first 100 characters
        clean_question = (instruction_item.question.replace('\n', ' ').replace('\r', ' ')[:22] + '...') if len(
            instruction_item.question) > 22 else instruction_item.question.replace('\n', ' ').replace('\r', ' ')
        truncated_answer = (answer.replace('\n', ' ').replace('\r', ' ')[
                            :22] + '...') if len(answer) > 25 else answer.replace('\n', ' ').replace('\r', ' ')

        # Print each row in a fixed-width format
        row = (f"{str(instruction_item_id):<{self.column_widths['instruction_id']}} | "
               f"{model_id:<{self.column_widths['model_id']}} | "
               f"{profile_id:<{self.column_widths['profile_id']}} | "
               f"{str(random_seed):<{self.column_widths['random_seed']}} | "
               f"{clean_question:<{self.column_widths['question']}} | "
               f"{truncated_answer:<{self.column_widths['answer']}}")
        print(row)
