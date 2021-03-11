# type: ignore

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

import sphobjinv as soi
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------
project = "ExpertSystem"
package = "expertsystem"
repo_name = "expertsystem"
copyright = "2020, ComPWA"
author = "Common Partial Wave Analysis"

if os.path.exists(f"../src/{package}/version.py"):
    __release = get_distribution(package).version
    version = ".".join(__release.split(".")[:3])

# -- Generate API skeleton ----------------------------------------------------
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
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_favicon = "_static/favicon.ico"
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
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
html_title = "PWA Expert System"
pygments_style = "sphinx"
todo_include_todos = False
viewcode_follow_imported_members = True

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "EdgeType"),
    ("py:class", "NoneType"),
    ("py:class", "StateTransitionGraph"),
    ("py:class", "ValueType"),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "numpy.typing._array_like._SupportsArray"),
    ("py:class", "numpy.typing._dtype_like._DTypeDict"),
    ("py:class", "numpy.typing._dtype_like._SupportsDType"),
    ("py:class", "typing_extensions.Protocol"),
    ("py:obj", "expertsystem.amplitude.helicity.ValueType"),
    ("py:obj", "expertsystem.reaction.topology._K"),
    ("py:obj", "expertsystem.reaction.topology._V"),
]

# Intersphinx settings
intersphinx_mapping = {
    "attrs": ("https://www.attrs.org/en/stable", None),
    "constraint": (
        "https://labix.org/doc/constraint/public",
        "constraint.inv",
    ),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "jsonschema": (
        "https://python-jsonschema.readthedocs.io/en/latest",
        None,
    ),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "pycompwa": ("https://compwa.github.io", None),
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": (
        "https://pwa.readthedocs.io/projects/tensorwaves/en/stable",
        None,
    ),
    "tox": ("https://tox.readthedocs.io/en/stable", None),
}

# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for bibtex
bibtex_bibfiles = ["bibliography.bib"]
suppress_warnings = [
    "myst.domains",
]

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False

# Settings for myst_nb
execution_excludepatterns = [
    "adr/001/*",
    "adr/002/*",
]
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
]
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}


# Specify bibliography style
from pybtex.plugin import register_plugin
from pybtex.richtext import Tag, Text
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import (
    FieldIsMissing,
    _format_list,
    field,
    href,
    join,
    node,
    sentence,
    words,
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

    def format_names(self, role, as_sentence=True):
        formatted_names = names(
            role, sep=", ", sep2=" and ", last_sep=", and "
        )
        if as_sentence:
            return sentence[formatted_names]
        else:
            return formatted_names

    def format_url(self, e):
        return words[
            href[
                field("url", raw=True),
                field("url", raw=True, apply_func=remove_http),
            ]
        ]

    def format_isbn(self, e):
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
