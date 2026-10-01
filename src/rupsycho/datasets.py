# datasets.py
"""Bundled example experiments.

The wheel ships a ready-to-run Big Five Inventory configuration so that the documentation
examples work after a plain ``pip install``. Larger study configurations live in the
``examples/data`` folder of the repository.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from rupsycho.experiment import ExperimentDocument

__all__ = ["list_examples", "load_example_config", "load_example_experiment"]

_PACKAGE = "rupsycho.data.examples"


def list_examples() -> list[str]:
    """Return the names of the bundled example configurations.

    Returns:
        Alphabetically sorted names, usable with ``load_example_config``.

    Example:
        ```python
        from rupsycho import list_examples

        print(list_examples())  # ['bfi']
        ```
    """
    files = resources.files(_PACKAGE)
    return sorted(
        entry.name.removesuffix(".json")
        for entry in files.iterdir()
        if entry.name.endswith(".json")
    )


def load_example_config(name: str = "bfi") -> dict[str, Any]:
    """Load a bundled example configuration as a dictionary.

    Args:
        name: Name of the example (see ``list_examples``).

    Returns:
        A fresh dictionary that may be modified freely, for instance to swap the model.

    Raises:
        KeyError: If there is no example with that name.

    Example:
        ```python
        from rupsycho import load_example_config

        config = load_example_config("bfi")
        config["parameters"]["seeds"] = ["1", "2", "3"]
        ```
    """
    path = resources.files(_PACKAGE).joinpath(f"{name}.json")
    if not path.is_file():
        raise KeyError(
            f"Unknown example {name!r}. Available examples: {', '.join(list_examples())}"
        )
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def load_example_experiment(
    name: str = "bfi",
    *,
    models: dict[str, Any] | None = None,
    seeds: list[int] | list[str] | None = None,
) -> ExperimentDocument:
    """Create an experiment from a bundled example.

    Args:
        name: Name of the example (see ``list_examples``).
        models: Replacement for the ``models`` section of the example, e.g. ``{}`` to add
            your own model with ``experiment.add_model(...)`` without the example's
            Hugging Face model being used.
        seeds: Replacement for the example's seeds, e.g. ``[1, 2, 3]``.

    Returns:
        The validated experiment.

    Example:
        ```python
        from rupsycho import load_example_experiment

        experiment = load_example_experiment("bfi", models={})
        ```
    """
    config = load_example_config(name)
    if models is not None:
        config["models"] = models
    if seeds is not None:
        config.setdefault("parameters", {})["seeds"] = seeds
    return ExperimentDocument(**config)
