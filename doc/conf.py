"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

import sphobjinv as soi

# -- Copy example notebooks ---------------------------------------------------
print("Copy example notebook and data files")
PATH_SOURCE = "../examples"
PATH_TARGET = "usage"
FILES_TO_COPY = [
    "additional_particles.yml",
    "particles.ipynb",
    "quickstart.ipynb",
    "visualization.ipynb",
]
shutil.rmtree(PATH_TARGET, ignore_errors=True)
os.makedirs(PATH_TARGET, exist_ok=True)
for file_to_copy in FILES_TO_COPY:
    path_from = os.path.join(PATH_SOURCE, file_to_copy)
    path_to = os.path.join(PATH_TARGET, file_to_copy)
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
    "-o api/ ../expertsystem/; ",
    shell=True,
)

# -- Convert sphinx object inventory -----------------------------------------
inv = soi.Inventory()
inv.project = "constraint"

constraint_object_names = [
    "Constraint",
    "Domain",
    "Problem",
    "Solver",
    "Variable",
]
for object_name in constraint_object_names:
    inv.objects.append(
        soi.DataObjStr(
            name=f"{inv.project}.{object_name}",
            domain="py",
            role="class",
            priority="1",
            uri=f"{inv.project}.{object_name}-class.html",
            dispname="-",
        )
    )

text = inv.data_file(contract=True)
ztext = soi.compress(text)
soi.writebytes("constraint.inv", ztext)


# -- Project information -----------------------------------------------------
project = "ExpertSystem"
copyright = "2020, ComPWA"
author = "The ComPWA Team"


# -- General configuration ---------------------------------------------------
source_suffix = [
    ".rst",
    ".ipynb",
    ".md",
]

# The master toctree document.
master_doc = "index"
modindex_common_prefix = [
    "expertsystem.",
]

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr/template.md",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join(
        [
            "__call__",
            "__eq__",
        ]
    ),
}
html_copy_source = True  # needed for download notebook button
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/ComPWA/expertsystem",
    "repository_branch": "master",
    "path_to_docs": "doc",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org/v2/gh/ComPWA/expertsystem/master?filepath=examples",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "expand_sections": ["usage"],
}
html_title = "PWA Expert System"
pygments_style = "sphinx"
todo_include_todos = False
viewcode_follow_imported_members = True

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "NoneType"),
    ("py:class", "StateTransitionGraph"),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "_T"),
    ("py:class", "an object providing a view on D's values"),
]

# Intersphinx settings
intersphinx_mapping = {
    "attrs": ("https://www.attrs.org/en/stable", None),
    "constraint": (
        "https://labix.org/doc/constraint/public/",
        "constraint.inv",
    ),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
    "jsonschema": (
        "https://python-jsonschema.readthedocs.io/en/latest/",
        None,
    ),
    "mypy": ("https://mypy.readthedocs.io/en/stable", None),
    "pwa": ("https://pwa.readthedocs.io/en/latest/", None),
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

# Settings for nbsphinx
if "NBSPHINX_EXECUTE" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    nbsphinx_execute = "always"
else:
    nbsphinx_execute = "never"
nbsphinx_timeout = -1
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Settings for myst-parser
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
