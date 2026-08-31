"""Sphinx configuration for cfad documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "cfad"
author = "Diogo Ribeiro"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_numpy_docstring = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
