# ===========================================================================
#                     Utilities for file handling
# ===========================================================================
# This file contains utility functions for handling files, such as loading and
# saving files.


import os
import json


def json_loader(path: str) -> dict:
    """
    Load a JSON file from the given path and return its contents as a dictionary.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def json_saver(data: dict, name: str = 'output', path: str = '') -> None:
    """
    Save a dictionary as a JSON file at the specified path and file name.
    If no path is provided, saves to the current directory.
    Prints the full path upon successful save.

    Args:
        data (dict): Dictionary to save as JSON.
        name (str, optional): Name of the file to save. Defaults to 'output'.
        path (str, optional): Directory path where the file will be saved. Defaults to the current directory.

    Returns:
        None
    """
    # Use current directory if no path is provided
    if not path:
        path = os.getcwd()
    if not name.endswith('.json'):
        full_path = os.path.join(path, f"{name}.json")
    else:
        full_path = os.path.join(path, f"{name}")
    with open(full_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    print(f"File saved successfully at: {full_path}")
