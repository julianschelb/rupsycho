# parameters.py
"""Data model: experiment-level parameters."""

from __future__ import annotations

from random import randint
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

__all__ = ["ExperimentParameters"]


def _normalise_seeds(value: Any) -> Any:
    """Accept integers or numeric strings (a single one or a list) and keep them as strings.

    Seeds are stored as strings because they are keys of the stored answers (and JSON object
    keys), but every seed must be an integer because that is what the models receive.
    """
    if value is None:
        return None
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"seeds must be a list of integers, got {value!r}")
    seeds = []
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, (str, int)):
            raise ValueError(f"seeds must be integers, got {seed!r}")
        try:
            int(seed)
        except ValueError:
            raise ValueError(f"seeds must be integers, got {seed!r}") from None
        seeds.append(str(seed).strip())
    return seeds


Seeds = Annotated[list[str] | None, BeforeValidator(_normalise_seeds)]


class ExperimentParameters(BaseModel):
    """Parameters of an experiment.

    Attributes:
        seeds: Seeds of the repetitions. Every model is asked every question once per seed.
            Integers (``[1, 2, 3]``) and numeric strings (``["1", "2"]``) are accepted. If
            omitted, one random seed is drawn **per experiment** - set seeds explicitly for
            reproducible experiments.
        lazy_load_models: Load each model only when the run reaches it (default). With
            ``False`` all models are loaded when the experiment is created.

    Additional keys are kept as they are.

    Example:
        ```python
        ExperimentParameters(seeds=[1, 2, 3]).seeds  # ['1', '2', '3']
        ```
    """

    seeds: Seeds = Field(
        default_factory=lambda: [str(randint(0, 999999))],
        description="A list of seeds for random number generation in the experiment.",
    )

    lazy_load_models: bool = Field(
        default=True,
        description="If True, models will be loaded only when they are needed during the experiment.",
    )

    model_config = ConfigDict(extra="allow")
