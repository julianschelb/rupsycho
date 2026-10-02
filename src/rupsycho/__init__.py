"""R.U.Psycho: robust, unified and reproducible psychometric testing of language models.

Describe an experiment (questionnaire, personas, models, prompt, seeds) in one JSON file or
dictionary, load it, run it and collect the answers:

```python
import rupsycho as rup

experiment = rup.experiment_from_file("config.json")
experiment.run()
answers = experiment.get_answers_as_dataframe()
```

``import rupsycho`` is deliberately cheap: the public names and the sub-packages
(``rup.parsers``, ``rup.callbacks``, ``rup.postprocessing``, ``rup.seeding``, ``rup.models``)
are imported on first access, so the command line starts instantly and no model back-end is
loaded until it is needed.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from rupsycho import callbacks, models, parsers, postprocessing, scoring, seeding
    from rupsycho.datasets import list_examples, load_example_config, load_example_experiment
    from rupsycho.experiment import ExperimentDocument
    from rupsycho.experiment_collection import ExperimentCollection
    from rupsycho.mixins.experiment_processing import RunSummary
    from rupsycho.reader import (
        ExperimentLoader,
        experiment_from_dict,
        experiment_from_file,
        experiments_from_dicts,
        experiments_from_files,
    )

__all__ = [
    "__version__",
    # Loading
    "experiment_from_file",
    "experiment_from_dict",
    "experiments_from_files",
    "experiments_from_dicts",
    "ExperimentLoader",
    # Experiments
    "ExperimentDocument",
    "ExperimentCollection",
    "RunSummary",
    # Bundled examples
    "list_examples",
    "load_example_config",
    "load_example_experiment",
]

# public name -> module that defines it (resolved on first access)
_LAZY_ATTRIBUTES = {
    "experiment_from_file": "rupsycho.reader",
    "experiment_from_dict": "rupsycho.reader",
    "experiments_from_files": "rupsycho.reader",
    "experiments_from_dicts": "rupsycho.reader",
    "ExperimentLoader": "rupsycho.reader",
    "ExperimentDocument": "rupsycho.experiment",
    "ExperimentCollection": "rupsycho.experiment_collection",
    "RunSummary": "rupsycho.mixins.experiment_processing",
    "list_examples": "rupsycho.datasets",
    "load_example_config": "rupsycho.datasets",
    "load_example_experiment": "rupsycho.datasets",
}

_LAZY_SUBMODULES = frozenset(
    {
        "callbacks",
        "datasets",
        "models",
        "parsers",
        "postprocessing",
        "scoring",
        "seeding",
        "utils",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve public names and sub-packages on first access (PEP 562)."""
    if name in _LAZY_ATTRIBUTES:
        value = getattr(importlib.import_module(_LAZY_ATTRIBUTES[name]), name)
        globals()[name] = value  # cache: later lookups bypass this function
        return value
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"rupsycho.{name}")
    raise AttributeError(f"module 'rupsycho' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES) | _LAZY_SUBMODULES)


def example_experiment_bfi(
    model_name: str = "google/flan-t5-small",
    pipeline_type: str = "text2text-generation",
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    api_key: str | None = None,
) -> ExperimentDocument:
    """Create a tiny Big Five experiment around a Hugging Face pipeline.

    Deprecated:
        Use [`load_example_experiment`][rupsycho.datasets.load_example_experiment] and add your
        model with ``experiment.add_model``.

    Args:
        model_name: Name of the Hugging Face model to use.
        pipeline_type: Pipeline task, e.g. ``"text2text-generation"``.
        temperature: Sampling temperature.
        max_new_tokens: Maximum number of generated tokens.
        api_key: Optional Hugging Face token for gated models.

    Returns:
        A configured experiment with the model already added.

    Raises:
        ImportError: If the ``huggingface`` extra is not installed.
    """
    warnings.warn(
        "example_experiment_bfi is deprecated; use load_example_experiment('bfi', models={}) "
        "and experiment.add_model(...)",
        DeprecationWarning,
        stacklevel=2,
    )
    from rupsycho._compat import require
    from rupsycho.datasets import load_example_experiment

    transformers = require("transformers", "huggingface", feature="example_experiment_bfi")
    lc_hf = require("langchain_huggingface", "huggingface", feature="example_experiment_bfi")

    if api_key:
        require("huggingface_hub", "huggingface").login(api_key)

    experiment = load_example_experiment("bfi", models={})
    pipe = transformers.pipeline(
        pipeline_type,
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    experiment.add_model(lc_hf.HuggingFacePipeline(pipeline=pipe), identifier="hf_generative_model")
    return experiment
