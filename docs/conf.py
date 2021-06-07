# type: ignore

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess
import sys

from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------
project = "AmpForm"
package = "ampform"
repo_name = "ampform"
copyright = "2020, ComPWA"
author = "Common Partial Wave Analysis"

if os.path.exists(f"../src/{package}/version.py"):
    __release = get_distribution(package).version
    version = ".".join(__release.split(".")[:3])

# -- Generate API ------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
import abbreviate_signature
import extend_docstrings

extend_docstrings.insert_math()

shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            f"../src/{package}/",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
        ]
    ),
    shell=True,
)

# -- General configuration ---------------------------------------------------
master_doc = "index.md"
source_suffix = {
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"
modindex_common_prefix = [
    f"{package}.",
]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.hep.pdgref",
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
    "exclude-members": ", ".join(
        [
            "default_assumptions",
            "doit",
            "evaluate",
        ]
    ),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join(
        [
            "__call__",
            "__eq__",
            "__getitem__",
        ]
    ),
}
autodoc_insert_signature_linebreaks = False
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_css_files = []
if autodoc_insert_signature_linebreaks:
    html_css_files.append("linebreaks-api.css")
html_favicon = "_static/favicon.ico"
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/ComPWA/{repo_name}",
    "repository_branch": "stable",
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "theme_dev_mode": True,
}
html_title = "AmpForm"
pygments_style = "sphinx"
todo_include_todos = False
viewcode_follow_imported_members = True

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "ipywidgets.widgets.widget_float.FloatSlider"),
    ("py:class", "ipywidgets.widgets.widget_int.IntSlider"),
    ("py:class", "typing_extensions.Protocol"),
]

# Intersphinx settings
intersphinx_mapping = {
    "attrs": ("https://www.attrs.org/en/stable", None),
    "compwa-org": ("https://compwa-org.readthedocs.io", None),
    "expertsystem": ("https://expertsystem.readthedocs.io/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mpl_interactions": (
        "https://mpl-interactions.readthedocs.io/en/stable",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": ("https://qrules.readthedocs.io/en/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": ("https://tensorwaves.readthedocs.io/en/stable", None),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for bibtex
bibtex_bibfiles = ["bibliography.bib"]
suppress_warnings = [
    "myst.domains",
]
bibtex_reference_style = "author_year_no_comma"

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False
linkcheck_ignore = [
    "https://doi.org/10.1093/ptep/ptaa104",
    "https://hss-opus.ub.ruhr-uni-bochum.de",
]

# Settings for myst_nb
execution_timeout = -1
nb_output_stderr = "remove"
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}
nb_render_priority["doctest"] = nb_render_priority["html"]

jupyter_execute_notebooks = "off"
if "EXECUTE_NB" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    jupyter_execute_notebooks = "force"

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
BINDER_LINK = (
    f"https://mybinder.org/v2/gh/ComPWA/{repo_name}/stable?filepath=docs/usage"
)
myst_substitutions = {
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <pwa:develop:Jupyter Notebooks>` to
interactively modify the parameters.
```
"""
}
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}


# Specify bibliography style
import dataclasses
from typing import Union

import sphinxcontrib.bibtex.plugin
from pybtex.database import Entry
from pybtex.plugin import register_plugin
from pybtex.richtext import Tag, Text
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import (
    FieldIsMissing,
    Node,
    _format_list,
    field,
    href,
    join,
    node,
    sentence,
    words,
)
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import (
    AuthorYearReferenceStyle,
)

no_brackets = BracketStyle(
    left="",
    right="",
)


@dataclasses.dataclass
class NoCommaReferenceStyle(AuthorYearReferenceStyle):
    author_year_sep: Union["BaseText", str] = " "
    bracket_parenthetical: BracketStyle = no_brackets


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing",
    "author_year_no_comma",
    NoCommaReferenceStyle,
)


@node
def et_al(children, data, sep="", sep2=None, last_sep=None):
    if sep2 is None:
        sep2 = sep
    if last_sep is None:
        last_sep = sep
    parts = [part for part in _format_list(children, data) if part]
    if len(parts) <= 1:
        return Text(*parts)
    elif len(parts) == 2:
        return Text(sep2).join(parts)
    elif len(parts) == 3:
        return Text(last_sep).join([Text(sep).join(parts[:-1]), parts[-1]])
    else:
        return Text(parts[0], Tag("em", " et al"))


@node
def names(children, context, role, **kwargs):
    """Return formatted names."""
    assert not children
    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])

    style = context["style"]
    formatted_names = [
        style.format_name(person, style.abbreviate_names) for person in persons
    ]
    return et_al(**kwargs)[formatted_names].format_data(context)


class MyStyle(UnsrtStyle):
    def __init__(self):
        super().__init__(abbreviate_names=True)

    def format_names(self, role, as_sentence=True) -> Node:
        formatted_names = names(
            role, sep=", ", sep2=" and ", last_sep=", and "
        )
        if as_sentence:
            return sentence[formatted_names]
        else:
            return formatted_names

    def format_eprint(self, e):
        if "doi" in e.fields:
            return ""
        return super().format_eprint(e)

    def format_url(self, e: Entry) -> Node:
        if "doi" in e.fields or "eprint" in e.fields:
            return ""
        return words[
            href[
                field("url", raw=True),
                field("url", raw=True, apply_func=remove_http),
            ]
        ]

    def format_isbn(self, e: Entry) -> Node:
        return href[
            join[
                "https://isbnsearch.org/isbn/",
                field("isbn", raw=True, apply_func=remove_dashes_and_spaces),
            ],
            join[
                "ISBN:",
                field("isbn", raw=True),
            ],
        ]


def remove_dashes_and_spaces(isbn: str) -> str:
    to_remove = ["-", " "]
    for remove in to_remove:
        isbn = isbn.replace(remove, "")
    return isbn


def remove_http(input: str) -> str:
    to_remove = ["https://", "http://"]
    for remove in to_remove:
        input = input.replace(remove, "")
    return input


register_plugin("pybtex.style.formatting", "unsrt_et_al", MyStyle)
