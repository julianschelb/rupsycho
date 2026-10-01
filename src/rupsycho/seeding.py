# seeding.py
"""Reproducible sampling: map an experiment seed onto each model back-end.

Every model call of an experiment carries a *seed* (``parameters.seeds``). How that seed
reaches the model differs per back-end, and silently ignoring it would defeat the purpose
of a seeded experiment:

* **Local Hugging Face pipelines** have no ``seed`` argument. Their sampling is driven by
  global random number generators, so the seed is applied with ``transformers.set_seed``
  right before every call.
* **API / server back-ends with a ``seed`` field** (OpenAI, DeepSeek, Ollama, ...) receive the
  seed through a copy of the model with that field set; chat models around a **Hugging Face
  endpoint** receive it as a request parameter.
* **Back-ends without any seed support** (e.g. Google Gemini) cannot be seeded. A warning is
  emitted once per model type; repetitions are then independent samples.

Custom model types can be registered with :func:`register_seeder`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from rupsycho._compat import require

__all__ = ["is_thread_safe", "register_seeder", "seed_model", "supports_seeding"]

Seeder = Callable[[Any, int], Runnable]
"""A function ``(model, seed) -> seeded runnable``."""

_REGISTRY: list[tuple[type, Seeder]] = []
_WARNED: set[str] = set()


def register_seeder(model_type: type, seeder: Seeder) -> None:
    """Register how models of ``model_type`` are seeded.

    The most recent registration wins and takes precedence over the built-in strategies.

    Args:
        model_type: The model class (subclasses match too).
        seeder: Function ``(model, seed)`` returning a runnable that behaves like ``model``
            but samples reproducibly for ``seed``.

    Example:
        ```python
        from rupsycho.seeding import register_seeder

        register_seeder(MyModel, lambda model, seed: model.model_copy(update={"rng_seed": seed}))
        ```
    """
    _REGISTRY.insert(0, (model_type, seeder))


def _inner(model: Any) -> Any:
    """Return the wrapped LLM of chat wrappers such as ``ChatHuggingFace``."""
    return getattr(model, "llm", model)


def _is_local_pipeline(model: Any) -> bool:
    """Whether ``model`` runs a Transformers pipeline in this process (no ``import`` needed)."""
    llm = _inner(model)
    return type(llm).__name__ == "HuggingFacePipeline" and hasattr(llm, "pipeline")


def _has_seed_field(model: Any) -> bool:
    return "seed" in getattr(type(model), "model_fields", {})


def is_thread_safe(model: Any) -> bool:
    """Whether calls to ``model`` may run concurrently in several threads.

    Local pipelines rely on process-global state (random generators, accelerator memory)
    and are therefore not thread safe; API clients are.
    """
    return not _is_local_pipeline(model)


def supports_seeding(model: Any) -> bool:
    """Whether :func:`seed_model` can make ``model`` reproducible."""
    if any(isinstance(model, cls) for cls, _ in _REGISTRY):
        return True
    return (
        _is_local_pipeline(model)
        or type(_inner(model)).__name__ == "HuggingFaceEndpoint"
        or _has_seed_field(model)
    )


def _set_global_seed(seed: int) -> Callable[[Any], Any]:
    def apply(value: Any) -> Any:
        # transformers.set_seed feeds numpy, which only accepts 0 <= seed < 2**32
        require("transformers", "huggingface", feature="Seeding local models").set_seed(
            seed % 2**32
        )
        return value

    return apply


def seed_model(model: Any, seed: int) -> Runnable:
    """Return a runnable that samples from ``model`` reproducibly for ``seed``.

    Args:
        model: The model (any LangChain runnable).
        seed: The seed of the current experiment run.

    Returns:
        A runnable to use in place of ``model``. For models that cannot be seeded the model
        itself is returned and a warning is emitted (once per model type).

    Example:
        ```python
        chain = prompt | seed_model(model, 42) | parser
        ```
    """
    seed = int(seed)

    for model_type, seeder in _REGISTRY:
        if isinstance(model, model_type):
            return seeder(model, seed)

    if _is_local_pipeline(model):
        return RunnableLambda(_set_global_seed(seed), name=f"set_seed({seed})") | model

    llm = _inner(model)
    if llm is not model and type(llm).__name__ == "HuggingFaceEndpoint":
        # ChatHuggingFace builds the request from its own model_kwargs and forwards them to
        # InferenceClient.chat_completion, which accepts ``seed``
        model_kwargs = {**(getattr(model, "model_kwargs", None) or {}), "seed": seed}
        return model.model_copy(update={"model_kwargs": model_kwargs})
    if _has_seed_field(model):
        return model.model_copy(update={"seed": seed})

    name = type(model).__name__
    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(
            f"Models of type {name} cannot be seeded; the experiment seeds have no effect on "
            "them and repeated runs are independent samples. Register a strategy with "
            "rupsycho.seeding.register_seeder to change this.",
            UserWarning,
            stacklevel=2,
        )
    return model
