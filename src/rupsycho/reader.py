# ===========================================================================
#                            Experiment Reader Class
# ===========================================================================
#  A document loader class for reading and validating JSON files against the
#  ExperimentDocument model. Supports both synchronous and asynchronous
#  loading of experiment data.

from typing import AsyncIterator, Iterator, List, Optional
from langchain_core.document_loaders import BaseLoader
from pydantic import ValidationError
import aiofiles
import json
import glob
import os

from .experiment import ExperimentDocument

# ================================= Helpers ================================


def _experiment_from_json(json_data: dict) -> ExperimentDocument:
    """Create a new experiment from a JSON dictionary."""
    return ExperimentDocument(**json_data)


async def _experiment_from_json_async(json_data: dict) -> ExperimentDocument:
    """Asynchronously create a new experiment from a JSON dictionary."""
    # Simulating an async operation, even though this operation is synchronous
    return ExperimentDocument(**json_data)


def _experiment_from_file(path: str) -> ExperimentDocument:
    """Create a new experiment from a JSON file."""
    with open(path, encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        return _experiment_from_json(json_data)


async def _experiment_from_file_async(path: str) -> ExperimentDocument:
    """Create a new experiment from a JSON file asynchronously."""
    async with aiofiles.open(path, encoding="utf-8") as json_file:
        json_data = json.loads(await json_file.read())
        return _experiment_from_json(json_data)


# ================================= Loader Class ================================


class ExperimentLoader(BaseLoader):
    """A document loader that reads JSON files validated by the ExperimentDocument model."""

    def __init__(self, path_pattern: Optional[str] = None) -> None:
        """
        Initialize the loader with a path pattern.

        Args:
            path_pattern (Optional[str]): A glob pattern or single file path to load JSON files from.
        """
        self.file_paths = self._resolve_paths(
            path_pattern) if path_pattern else None

    def _resolve_paths(self, path_pattern: str) -> List[str]:
        """Resolve the provided path pattern to a list of file paths.

        Args:
            path_pattern (str): A glob pattern or single file path.

        Returns:
            List[str]: A list of resolved file paths.
        """
        if os.path.isfile(path_pattern):
            return [path_pattern]
        return glob.glob(path_pattern, recursive=True)

    def lazy_load(self, path_pattern: Optional[str] = None) -> Iterator[ExperimentDocument]:
        """
        A lazy loader that reads JSON files from the resolved paths and validates them
        against the ExperimentDocument model.

        Args:
            path_pattern (Optional[str]): A glob pattern or single file path to load JSON files from.

        Yields:
            ExperimentDocument: A document object with the validated JSON content.

        Raises:
            ValueError: If neither constructor nor method arguments provide a valid path pattern.
        """
        file_paths = self._resolve_paths(
            path_pattern) if path_pattern else self.file_paths

        if not file_paths:
            raise ValueError(
                "No file paths provided. Please provide a path pattern.")

        for file_path in file_paths:
            try:
                yield _experiment_from_file(file_path)
            except ValidationError as e:
                print(f"Validation error in file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    async def alazy_load(self, path_pattern: Optional[str] = None) -> AsyncIterator[ExperimentDocument]:
        """
        An async lazy loader that reads JSON files from the resolved paths and validates them
        against the ExperimentDocument model.

        Args:
            path_pattern (Optional[str]): A glob pattern or single file path to load JSON files from.

        Yields:
            ExperimentDocument: A document object with the validated JSON content.

        Raises:
            ValueError: If neither constructor nor method arguments provide a valid path pattern.
        """
        file_paths = self._resolve_paths(
            path_pattern) if path_pattern else self.file_paths

        if not file_paths:
            raise ValueError(
                "No file paths provided. Please provide a path pattern.")

        for file_path in file_paths:
            try:
                yield await _experiment_from_file_async(file_path)
            except ValidationError as e:
                print(f"Validation error in file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    def lazy_load_from_dicts(self, dicts: List[dict]) -> Iterator[ExperimentDocument]:
        """
        A lazy loader that reads from a list of dictionaries (JSON objects) and validates them
        against the ExperimentDocument model.

        Args:
            dicts (List[dict]): A list of dictionaries representing experiments.

        Yields:
            ExperimentDocument: A document object with the validated JSON content.
        """
        for json_data in dicts:
            try:
                yield _experiment_from_json(json_data)
            except ValidationError as e:
                print(f"Validation error in provided dictionary: {e}")
            except Exception as e:
                print(f"Error processing provided dictionary: {e}")

    async def alazy_load_from_dicts(self, dicts: List[dict]) -> AsyncIterator[ExperimentDocument]:
        """
        An async lazy loader that reads from a list of dictionaries (JSON objects) and validates them
        against the ExperimentDocument model.

        Args:
            dicts (List[dict]): A list of dictionaries representing experiments.

        Yields:
            ExperimentDocument: A document object with the validated JSON content.
        """
        for json_data in dicts:
            try:
                yield await _experiment_from_json_async(json_data)
            except ValidationError as e:
                print(f"Validation error in provided dictionary: {e}")
            except Exception as e:
                print(f"Error processing provided dictionary: {e}")

# ================================= Main ================================


if __name__ == "__main__":
    loader = ExperimentLoader("../examples/data/bfi_experiment*.json")
    experiments = loader.lazy_load()

    for experiment in experiments:
        print(experiment.questionnaire)
