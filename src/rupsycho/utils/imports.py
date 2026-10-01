# imports.py
"""Utilities for import handling."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

__all__ = ["import_tqdm"]


def import_tqdm() -> Any:
    """Return the ``tqdm`` flavour that fits the environment.

    In a Jupyter kernel the notebook widget variant is returned, everywhere else the console
    progress bar. IPython is only inspected if it has already been imported, so plain scripts
    never pay for importing it.

    Returns:
        The ``tqdm`` class.
    """
    ipython = sys.modules.get("IPython")
    if ipython is not None:
        shell = ipython.get_ipython()
        # Jupyter notebook or qtconsole
        in_kernel = shell is not None and "IPKernelApp" in shell.config
        # tqdm.notebook cannot draw its widget without ipywidgets: fall back to the console bar
        if in_kernel and importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.notebook import tqdm as notebook_tqdm

            return notebook_tqdm

    from tqdm import tqdm

    return tqdm
