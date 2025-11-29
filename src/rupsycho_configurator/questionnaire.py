""" 
This module defines a class to represent questionnaires.

Instances of this class are easily convertible into the 'questionnaire' section of a 
R.U.Psycho experiment configuration.
"""

import json
import re
from utils import *
from pprint import pprint
from difflib import SequenceMatcher
import traceback
import datetime
import os

class Questionnaire():
    """
    This class represents a questionnaire for the configurator GUI.
    
    The class stores the important information of a psychometric questionnaire: the title, the instructions to the reader,
    the questions and the answer sets to the questions. Instantiation is possible from different sources. Questionnaire objects 
    can be written to JSON files with the structure of the "questionnaire" attribute in the R.U.Psycho JSON format.
    """

    def __init__(self, title: str, instr: str, questions: list[str], anss: list[list[str]]):
        """Construct a questionnaire object.
        
        Args:
            title (str): The title of the questionnaire
            str (str): The instruction to the reader on how to fill out the questionnaire
            questions (list[str]): The questions / items of the questionnaire each as a separate string element in the list
            anss (list[list[str]]): All sets of answer options. Each set is a list where each answer option is contained as a separate string element. Must follow the order of the questions in the 'questions' argument.
        """
        self.title = title
        self.instructions = instr
        self.questions = questions
        self.answers = anss # a single set of answers counts as global


    @classmethod
    def from_raw_output(cls, title_instr: str, questions: str, answers: str):
        """Create a questionnaire instance directly from the outputs of a LM that follow the structure specified in the 'Prompts' module."""
        try:
            title, instructions  = extract_json(title_instr, False)
        except:
            raise ValueError("beeg problem")
        questions = extract_json(questions)
        answers = extract_json(answers)
        return(cls(title, instructions, questions, answers))


    @classmethod
    def from_dict(cls, quest: dict[str, any]):
        """Create a questionnaire instance from a dictionary that follows the format of the 'questionnaire' section of a R.U.psycho experiment configuration."""
        quest = quest['questionnaire']
        title = quest['name']
        instructions = quest['general_instruction']
        questions = []
        answers = []
        items = quest['instruction_items']
        for item in items:
            questions.append(item['question'])
            ans_set = [opt['text'] for opt in item['answer_options'].values()]
            answers.append(ans_set)

        return cls(title, instructions, questions, answers)


    @classmethod
    def from_file(cls, path: str):
        """Create questionnaire instance from a JSON file that follows the format of the 'questionnaire' section 
        of a R.U.psycho experiment configuration.
        """
        with open(path) as f:
            quest_dict = json.load(f)
            return Questionnaire.from_dict(quest_dict)
    
        
    def merge_questions_answers(self) -> list[tuple[str,list[str]]]:
        """Return a list of tuples that contain position matched pairs of questions and answer option sets.
        
        Each answer sets is mapped to the questions with the same position if possible.
        If there are x less answer sets than there are questions then the the last x questions are mapped to the empty answer set.
        If there are x more answer sets than there are questions then the empty string is mapped to the last x answer sets.
        If there is only a single answer set then this answer set is considered global and is used for every question.
        """
        questions = self.questions
        answers = self.answers

        match_len = min(len(questions), len(answers))
        overlap = abs(len(questions) - len(answers))

        if len(answers) == 1: # use global answer set for all quest items
            instr_items = [(q, answers[0]) for q in questions]

        else: # use individual answer sets
            instr_items = [(questions[i], answers[i]) for i in range(match_len)]
            if len(questions) > len(answers):
                instr_items.extend([(questions[match_len + j], []) for j in range(overlap)])
            else:
                instr_items.extend([("", answers[match_len + k]) for k in range(overlap)])

        return instr_items


    def mk_dict(self) -> dict:
        """Return a dictionary representation of the the questionnaire object that follows the format 
        of the 'questionnaire' section of a R.U.psycho experiment configuration.
        """
        questions_answers = self.merge_questions_answers()
        quest_dict = {
                "questionnaire": {
                    "name": self.title,
                    "general_instruction": self.instructions,
                    "attributes": {},
                    # "default_answer_options": {},
                    "instruction_items": []
                }
            }
        for i in range(len(questions_answers)):
            q, a = questions_answers[i]
            # adds instruction item
            quest_dict["questionnaire"]["instruction_items"].append({
                "question": q,
                "reversed": False,
                "answer_options": {},
                "attributes": {}
            })
            # adds all answer options for this instruction item
            for j in range(len(a)):
                # to the i-th instruction item, this adds the j-th answer option under key "j+1"
                quest_dict["questionnaire"]["instruction_items"][i]["answer_options"][str(j+1)] = {
                    "text": a[j], # if an answer set is a dict instead of list, then here is where it fails
                    "weight": j+1,
                    "ignored_for_scale": False
                }
        return quest_dict


    def to_json_file(self, path:str) -> str:
        "Write the questionnaire to a JSON file at the specified location in R.U.Psycho format and return it as a serialized json."
        json_quest = json.dumps(self.mk_dict(), indent=4)
        with open(path, "w") as f:
            f.write(json_quest)
        # print('JSON created')
        return json_quest
    

    def get_as_dict(self):
        """Return a dictionary representation of the the questionnaire object that follows the format 
        of the 'questionnaire' section of a R.U.psycho experiment configuration.
        """
        return self.mk_dict()
    

    def get_merged_questions_answers(self):
        """Return a list of tuples that contains all position matched pairs of questions and answer option sets.
        
        Each answer sets is mapped to the questions with the same position if possible.
        If there are x less answer sets than there are questions then the the last x questions are mapped to the empty answer set.
        If there are x more answer sets than there are questions then the empty string is mapped to the last x answer sets.
        If there is only a single answer set then this answer set is considered global and is used for every question.
        """
        return self.merge_questions_answers()


    def get_as_list(self):
        """Return the four content parts of the questionnaire (title, instruction, questions, answers) as a list."""
        return [self.title, self.instructions, self.questions, self.answers]
    

    def __str__(self):
        """Return a nice string representation of the questionnaire for printing."""
        print_str = ""
        print_str += f"title: {self.title}\n\n"
        print_str += f"instructions: {self.instructions}\n\n"
        print_str += f"questionnaire items ({len(self.questions)}):\n"
        print_str += "    " + "\n    ".join(self.questions) + "\n\n"
        print_str += f"answer sets ({len(self.answers)}):\n"
        for set in self.answers:
            print_str += "    " + set.__str__() + "\n"
        
        # print_str += f"{self.title}\n\n"
        # print_str += f"{self.instructions}\n\n"    

        # for question, anss in self.merge_questions_answers():
        #     print_str += f"{question}\n" if question else '<EMPTY>\n'
        #     print_str += f"    {anss.__str__()}\n"
        return print_str.strip()
    
    
if __name__ == "__main__":
    pass