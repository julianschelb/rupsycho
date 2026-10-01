# ===========================================================================
#                     Utilities for import handling
# ===========================================================================
# This file contains utility functions for handling imports, such as conditionally
# importing modules based on the environment.

from typing import Any


def import_tqdm() -> Any:
    """Import the ``tqdm`` flavour that fits the environment (notebook widget or plain console)."""
    try:
        from IPython import get_ipython

        ipython_instance = get_ipython()
        # Jupyter notebook or qtconsole
        if ipython_instance is not None and "IPKernelApp" in ipython_instance.config:
            from tqdm.notebook import tqdm as notebook_tqdm

            return notebook_tqdm
    except ImportError:
        # IPython is not available: fall back to the console progress bar
        pass

    from tqdm import tqdm

    return tqdm
