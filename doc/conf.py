"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

# -- Copy example notebooks ---------------------------------------------------
print("Copy example notebook files")
# Remove old notebooks
PATH_TARGET = "usage"
os.makedirs(PATH_TARGET, exist_ok=True)
for root, _, files in os.walk(PATH_TARGET):
    for notebook in files:
        if notebook.endswith(".ipynb"):
            full_path = os.path.join(root, notebook)
            print("  removing notebook", full_path)
            os.remove(full_path)
# Copy notebooks from example directory
PATH_SOURCE = "../examples"
for root, _, files in os.walk(PATH_SOURCE):
    for notebook in files:
        if ".ipynb_checkpoints" in root:
            continue
        if not notebook.endswith(".ipynb"):
            continue
        path_from = os.path.join(root, notebook)
        path_to = os.path.join(PATH_TARGET, notebook)
        print("  copy", path_from, "to", path_to)
        shutil.copyfile(path_from, path_to, follow_symlinks=True)

# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    "sphinx-apidoc "
    "--force "
    "--no-toc "
    "--templatedir _templates "
    "--separate "
    "-o api/ ../expertsystem/ "
    "../expertsystem/solvers/constraint; ",
    shell=True,
)


# -- Project information -----------------------------------------------------
project = "ExpertSystem"
copyright = "2020, ComPWA"
author = "The ComPWA Team"


# -- General configuration ---------------------------------------------------
source_suffix = [
    ".rst",
    ".ipynb",
]
# The master toctree document.
master_doc = "index"

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "test",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__call__, __eq__",
}
html_copy_source = False  # do not copy rst files
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
todo_include_todos = False
viewcode_follow_imported_members = True

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "StateTransitionGraph"),
    ("py:class", "expertsystem.solvers.constraint.Constraint"),
    ("py:class", "expertsystem.state.propagation.GraphElementTypes"),
]

# Intersphinx settings
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pycompwa": ("https://compwa.github.io/", None),
    "python": ("https://docs.python.org/3", None),
    "tensorwaves": (
        "https://pwa.readthedocs.io/projects/tensorwaves/en/latest/",
        None,
    ),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for linkcheck
linkcheck_anchors = False
linkcheck_ignore = [
    "https://pypi.org/project/expertsystem",
    "https://pypi.org/project/expertsystem",
]


# Settings for nbsphinx
if "NBSPHINX_EXECUTE" in os.environ:
    nbsphinx_execute = "always"
else:
    nbsphinx_execute = "never"
nbsphinx_timeout = -1
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
