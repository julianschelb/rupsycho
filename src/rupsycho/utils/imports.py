# ===========================================================================
#                     Utilities for import handling
# ===========================================================================
# This file contains utility functions for handling imports, such as conditionally
# importing modules based on the environment.

def import_tqdm():
    """ Import tqdm based on the environment."""

    # Conditionally import tqdm based on the environment
    try:
        from IPython import get_ipython
        ipython_instance = get_ipython()
        # Jupyter notebook or qtconsole
        if ipython_instance is not None and 'IPKernelApp' in ipython_instance.config:
            from tqdm.notebook import tqdm
        else:  # Other environments
            from tqdm import tqdm
    except ImportError:
        # fallback if IPython is not available
        from tqdm import tqdm

    return tqdm
