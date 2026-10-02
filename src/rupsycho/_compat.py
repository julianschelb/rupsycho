# _compat.py
"""Helpers for optional dependencies.

Heavy back-ends (PyTorch, Transformers, provider SDKs) are imported lazily so that
``import rupsycho`` stays fast and works without them. Use :func:`require` at the point
where a back-end is actually needed to get an actionable error message.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any

__all__ = ["default_device", "is_available", "load_serialized", "require"]


def require(module: str, extra: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module`` or raise an ``ImportError`` that names the extra to install.

    Args:
        module: Dotted module name, e.g. ``"transformers"``.
        extra: Name of the ``rupsycho`` extra that provides it, e.g. ``"huggingface"``.
        feature: Human readable description of what needs the module (for the message).

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.

    Example:
        ```python
        transformers = require("transformers", "huggingface", feature="local models")
        ```
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = feature or f"'{module}'"
        raise ImportError(
            f"{what} requires the optional dependency '{module}'. "
            f"Install it with: pip install 'rupsycho[{extra}]'"
        ) from exc


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported (without keeping it imported on failure)."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def load_serialized(definition: dict[str, Any]) -> Any:
    """Deserialize a LangChain object produced by ``langchain_core.load.dumpd``.

    Newer ``langchain-core`` versions only deserialize core classes unless told otherwise.
    Serialized models in an experiment configuration are *explicitly* requested by the user
    (``"type": "langchain"``), so partner classes such as ``ChatOpenAI`` are allowed. Only load
    configurations you trust.

    Args:
        definition: The serialized object.

    Returns:
        The deserialized object.
    """
    from langchain_core.load import load

    if "allowed_objects" in inspect.signature(load).parameters:
        return load(definition, allowed_objects="all")
    return load(definition)


def default_device() -> str:
    """Return ``"cuda:0"`` if a CUDA GPU is available, otherwise ``"cpu"``.

    Raises:
        ImportError: If PyTorch (the ``huggingface`` extra) is not installed.
    """
    torch = require("torch", "huggingface", feature="Model-based parsers")
    return "cuda:0" if torch.cuda.is_available() else "cpu"
