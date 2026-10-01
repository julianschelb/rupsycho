"""R.U.Psycho: robust, unified and reproducible psychometric testing of language models.

Describe an experiment (questionnaire, personas, models, prompt, seeds) in one JSON file or
dictionary, load it, run it and collect the answers:

```python
import rupsycho as rup

experiment = rup.experiment_from_file("config.json")
experiment.run()
answers = experiment.get_answers_as_dataframe()
```

Sub-packages are imported on first access (``rup.parsers``, ``rup.callbacks``,
``rup.postprocessing``, ``rup.seeding``, ``rup.models``) so that ``import rupsycho`` stays
fast and does not require the heavy model back-ends.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

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

if TYPE_CHECKING:
    from rupsycho import callbacks, models, parsers, postprocessing, seeding

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

_LAZY_SUBMODULES = frozenset(
    {"callbacks", "models", "parsers", "postprocessing", "seeding", "datasets", "utils"}
)


def __getattr__(name: str) -> Any:
    """Import sub-packages on first access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"rupsycho.{name}")
    raise AttributeError(f"module 'rupsycho' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


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
