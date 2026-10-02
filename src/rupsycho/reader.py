# reader.py
"""Load experiments from JSON files or dictionaries.

``ExperimentLoader`` is a LangChain document loader that validates configurations against
[`ExperimentDocument`][rupsycho.experiment.ExperimentDocument]. When loading *several*
experiments, invalid ones are skipped and reported through the ``rupsycho.reader`` logger
(pass ``strict=True`` to raise instead). The convenience functions
``experiment_from_file`` / ``experiment_from_dict`` always raise on invalid input.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import aiofiles
from langchain_core.document_loaders import BaseLoader
from pydantic import ValidationError

from rupsycho.experiment import ExperimentDocument
from rupsycho.experiment_collection import ExperimentCollection

__all__ = [
    "ExperimentLoader",
    "experiment_from_dict",
    "experiment_from_file",
    "experiments_from_dicts",
    "experiments_from_files",
]

logger = logging.getLogger(__name__)

# ================================= Helpers ================================


def _experiment_from_json(json_data: dict[str, Any]) -> ExperimentDocument:
    """Create a new experiment from a JSON dictionary."""
    if not isinstance(json_data, Mapping):
        raise ValueError(
            f"An experiment configuration must be a JSON object, got {type(json_data).__name__}"
        )
    return ExperimentDocument(**json_data)


def _experiment_from_file(path: str | os.PathLike[str]) -> ExperimentDocument:
    """Create a new experiment from a JSON file."""
    with open(path, encoding="utf-8") as json_file:
        return _experiment_from_json(json.load(json_file))


async def _experiment_from_file_async(path: str | os.PathLike[str]) -> ExperimentDocument:
    """Create a new experiment from a JSON file asynchronously."""
    async with aiofiles.open(path, encoding="utf-8") as json_file:
        return _experiment_from_json(json.loads(await json_file.read()))


# ================================= Loader Class ================================


class ExperimentLoader(BaseLoader):
    """A document loader that reads JSON files validated by the ExperimentDocument model.

    Args:
        path_pattern: A file path or glob pattern (``**`` is supported) of JSON files.
        strict: Raise on the first invalid experiment instead of logging and skipping it.

    Example:
        ```python
        from rupsycho.reader import ExperimentLoader

        experiments = ExperimentLoader("configs/*.json").load()
        ```
    """

    def __init__(self, path_pattern: str | None = None, *, strict: bool = False) -> None:
        self.strict = strict
        self.file_paths = self._resolve_paths(path_pattern) if path_pattern else None

    def _resolve_paths(self, path_pattern: str) -> list[str]:
        """Resolve a path or glob pattern to a sorted list of file paths.

        Args:
            path_pattern: A glob pattern or single file path.

        Returns:
            The matching files in alphabetical order (deterministic across platforms).
        """
        if os.path.isfile(path_pattern):
            return [path_pattern]
        return sorted(glob.glob(path_pattern, recursive=True))

    def _skip_or_raise(self, error: Exception, source: str) -> None:
        if self.strict:
            raise error
        kind = "Invalid experiment" if isinstance(error, ValidationError) else "Cannot read"
        logger.warning("%s %s: %s", kind, source, error)

    def lazy_load(  # type: ignore[override]  # yields ExperimentDocument, not langchain Document
        self, path_pattern: str | None = None
    ) -> Iterator[ExperimentDocument]:
        """Lazily load and validate the experiments in the resolved files.

        Args:
            path_pattern: A glob pattern or single file path; defaults to the pattern the
                loader was created with.

        Yields:
            One validated experiment per readable, valid file.

        Raises:
            ValueError: If no file paths are available.
        """
        file_paths = self._resolve_paths(path_pattern) if path_pattern else self.file_paths
        if not file_paths:
            raise ValueError("No file paths provided. Please provide a path pattern.")

        for file_path in file_paths:
            try:
                yield _experiment_from_file(file_path)
            except (ValidationError, ValueError, OSError) as e:
                self._skip_or_raise(e, file_path)

    async def alazy_load(  # type: ignore[override]  # yields ExperimentDocument, not langchain Document
        self, path_pattern: str | None = None
    ) -> AsyncIterator[ExperimentDocument]:
        """Asynchronous version of [`lazy_load`][rupsycho.reader.ExperimentLoader.lazy_load]."""
        file_paths = self._resolve_paths(path_pattern) if path_pattern else self.file_paths
        if not file_paths:
            raise ValueError("No file paths provided. Please provide a path pattern.")

        for file_path in file_paths:
            try:
                yield await _experiment_from_file_async(file_path)
            except (ValidationError, ValueError, OSError) as e:
                self._skip_or_raise(e, file_path)

    def lazy_load_from_dicts(self, dicts: list[dict[str, Any]]) -> Iterator[ExperimentDocument]:
        """Lazily validate dictionaries as experiments.

        Args:
            dicts: Experiment configurations.

        Yields:
            One validated experiment per valid dictionary.
        """
        for index, json_data in enumerate(dicts):
            try:
                yield _experiment_from_json(json_data)
            except (ValidationError, ValueError, TypeError) as e:
                self._skip_or_raise(e, f"dictionary #{index}")

    async def alazy_load_from_dicts(
        self, dicts: list[dict[str, Any]]
    ) -> AsyncIterator[ExperimentDocument]:
        """Asynchronous version of
        [`lazy_load_from_dicts`][rupsycho.reader.ExperimentLoader.lazy_load_from_dicts]."""
        for index, json_data in enumerate(dicts):
            try:
                yield _experiment_from_json(json_data)
            except (ValidationError, ValueError, TypeError) as e:
                self._skip_or_raise(e, f"dictionary #{index}")


def experiment_from_file(path: str | os.PathLike[str]) -> ExperimentDocument:
    """Load one experiment from a JSON file.

    Args:
        path: Path of the JSON file.

    Returns:
        The validated experiment.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or not a valid experiment (pydantic's
            ``ValidationError`` is a ``ValueError``).

    Example:
        ```python
        import rupsycho as rup

        experiment = rup.experiment_from_file("config.json")
        ```
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")
    return _experiment_from_file(path)


def experiment_from_dict(json_data: dict[str, Any]) -> ExperimentDocument:
    """Create one experiment from a configuration dictionary.

    Args:
        json_data: The experiment configuration (see the configuration reference).

    Returns:
        The validated experiment.

    Raises:
        ValueError: If the configuration is invalid (pydantic's ``ValidationError`` is a
            ``ValueError``).
    """
    return _experiment_from_json(json_data)


def experiments_from_dicts(
    json_data_list: list[dict[str, Any]], *, strict: bool = False
) -> Iterator[ExperimentDocument]:
    """Lazily create experiments from configuration dictionaries.

    Args:
        json_data_list: The experiment configurations.
        strict: Raise on the first invalid configuration instead of logging and skipping it.

    Returns:
        An iterator over the valid experiments.
    """
    return ExperimentLoader(strict=strict).lazy_load_from_dicts(json_data_list)


def experiments_from_files(path_pattern: str, *, strict: bool = False) -> ExperimentCollection:
    """Load all experiments matching a path or glob pattern.

    Args:
        path_pattern: A JSON file or glob pattern such as ``"configs/*.json"``.
        strict: Raise on the first invalid file instead of logging and skipping it.

    Returns:
        A collection of the valid experiments, ordered by file name.

    Raises:
        ValueError: If no file matches the pattern.
    """
    loader = ExperimentLoader(path_pattern, strict=strict)
    if not loader.file_paths:
        raise ValueError(f"No experiment files match {path_pattern!r}")
    return ExperimentCollection(list(loader.lazy_load()))
