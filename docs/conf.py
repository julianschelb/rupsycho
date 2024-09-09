# Configuration file for the Sphinx documentation builder.

import sys
import os

# -- Project information -----------------------------------------------------
project = "RUPsycho"
copyright = ""
author = "Orr, Andreas, Julian"
release = "0.10"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",  # Added to include links to the source code
    "sphinx_copybutton",  # Added to include a copy button for code snippets
]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
# Display members in the order they appear in the source code
autodoc_member_order = 'bysource'

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Include type hints in the documentation
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinx_rtd_theme'  # 'furo', sphinxawesome_theme' "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    'display_version': False,
    'prev_next_buttons_location': None,
}

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))
