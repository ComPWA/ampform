"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
import shutil
import subprocess
import sys

import requests

# pyright: reportMissingImports=false
# pyright: reportUntypedBaseClass=false
# pyright: reportUntypedFunctionDecorator=false
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

if sys.version_info < (3, 8):
    from importlib_metadata import PackageNotFoundError
    from importlib_metadata import version as get_package_version
else:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as get_package_version

# -- Project information -----------------------------------------------------
project = "AmpForm"
PACKAGE = "ampform"
REPO_NAME = "ampform"
copyright = "2020, ComPWA"  # noqa: A001
author = "Common Partial Wave Analysis"

# https://docs.readthedocs.io/en/stable/builds.html
def get_branch_name() -> str:
    branch_name = os.environ.get("READTHEDOCS_VERSION", "stable")
    if branch_name == "latest":
        return "main"
    if re.match(r"^\d+$", branch_name):  # PR preview
        return "stable"
    return branch_name


BRANCH = get_branch_name()

try:
    __VERSION = get_package_version(PACKAGE)
    version = ".".join(__VERSION.split(".")[:3])
except PackageNotFoundError:
    pass


# -- Fetch logo --------------------------------------------------------------
def fetch_logo(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    online_content = requests.get(url, allow_redirects=True)
    with open(output_path, "wb") as stream:
        stream.write(online_content.content)


LOGO_PATH = "_static/logo.svg"
try:
    fetch_logo(
        url="https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg",
        output_path=LOGO_PATH,
    )
except requests.exceptions.ConnectionError:
    pass
if os.path.exists(LOGO_PATH):
    html_logo = LOGO_PATH

# -- Generate API ------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
from _extend_docstrings import extend_docstrings  # noqa: E402
from _relink_references import relink_references  # noqa: E402

extend_docstrings()
relink_references()

shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            f"../src/{PACKAGE}/",
            f"../src/{PACKAGE}/version.py",
            f"../src/{PACKAGE}/io/latex.py",
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
    f"{PACKAGE}.",
]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_codeautolink",
    "sphinx_comments",
    "sphinx_copybutton",
    "sphinx_design",
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
            "as_explicit",
            "default_assumptions",
            "doit",
            "evaluate",
            "is_commutative",
            "is_extended_real",
            "items",
            "keys",
            "precedence",
            "values",
        ]
    ),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join(
        [
            "__call__",
        ]
    ),
}
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "BuilderReturnType": "ampform.dynamics.builder.BuilderReturnType",
    "FourMomenta": "ampform.kinematics.FourMomenta",
    "FourMomentumSymbol": "ampform.kinematics.FourMomentumSymbol",
    "RangeDefinition": "symplot.RangeDefinition",
    # https://github.com/sphinx-doc/sphinx/pull/10183
    # "ParameterValue": "ampform.helicity.ParameterValue",
    # "Slider": "symplot.Slider",
}
autodoc_typehints_format = "short"
codeautolink_concat_default = True
codeautolink_global_preface = """
import numpy
import numpy as np
import sympy as sp
from IPython.display import display

"""
AUTODOC_INSERT_SIGNATURE_LINEBREAKS = False
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_css_files = [
    "custom.css",
]
if AUTODOC_INSERT_SIGNATURE_LINEBREAKS:
    html_css_files.append("linebreaks-api.css")
html_favicon = "_static/favicon.ico"
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/ComPWA/{REPO_NAME}",
    "repository_branch": BRANCH,
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "show_navbar_depth": 2,
    "show_toc_level": 2,
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
    ("py:class", "ArraySum"),
    ("py:class", "ampform.sympy._array_expressions.MatrixMultiplication"),
    ("py:class", "ipywidgets.widgets.widget_float.FloatSlider"),
    ("py:class", "ipywidgets.widgets.widget_int.IntSlider"),
    ("py:class", "typing_extensions.Protocol"),
]


# Intersphinx settings
version_remapping = {
    "matplotlib": {"3.5.1": "3.5.0"},
}


def get_version(package_name: str) -> str:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    constraints_path = f"../.constraints/py{python_version}.txt"
    package_name = package_name.lower()
    with open(constraints_path) as stream:
        constraints = stream.read()
    for line in constraints.split("\n"):
        line = line.split("#")[0]  # remove comments
        line = line.strip()
        line = line.lower()
        if not line.startswith(package_name):
            continue
        if not line:
            continue
        line_segments = tuple(line.split("=="))
        if len(line_segments) != 2:
            continue
        _, installed_version, *_ = line_segments
        installed_version = installed_version.strip()
        remapped_versions = version_remapping.get(package_name)
        if remapped_versions is not None:
            existing_version = remapped_versions.get(installed_version)
            if existing_version is not None:
                return existing_version
        return installed_version
    return "stable"


def get_minor_version(package_name: str) -> str:
    installed_version = get_version(package_name)
    if installed_version == "stable":
        return installed_version
    matches = re.match(r"^([0-9]+\.[0-9]+).*$", installed_version)
    if matches is None:
        raise ValueError(
            f"Could not find documentation for {package_name} v{installed_version}"
        )
    return matches[1]


intersphinx_mapping = {
    "IPython": (
        f"https://ipython.readthedocs.io/en/{get_version('IPython')}",
        None,
    ),
    "attrs": (f"https://www.attrs.org/en/{get_version('attrs')}", None),
    "compwa-org": ("https://compwa-org.readthedocs.io", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "ipywidgets": (
        f"https://ipywidgets.readthedocs.io/en/{get_version('ipywidgets')}",
        None,
    ),
    "matplotlib": (
        f"https://matplotlib.org/{get_version('matplotlib')}",
        None,
    ),
    "mpl_interactions": (
        f"https://mpl-interactions.readthedocs.io/en/{get_version('mpl-interactions')}",
        None,
    ),
    "numpy": (f"https://numpy.org/doc/{get_minor_version('numpy')}", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (
        f"https://qrules.readthedocs.io/en/{get_version('qrules')}",
        None,
    ),
    "sympy": ("https://docs.sympy.org/latest", None),
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
linkcheck_ignore = [
    "https://doi.org/10.1093/ptep/ptaa104",
    "https://home.fnal.gov/~kutschke/Angdist/angdist.ps",
    "https://hss-opus.ub.ruhr-uni-bochum.de",
    "https://physique.cuso.ch",
    "https://suchung.web.cern.ch",
]


# Settings for myst_nb
def get_execution_mode() -> str:
    if "FORCE_EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run ALL Jupyter notebooks!\033[0m")
        return "force"
    if "EXECUTE_NB" in os.environ:
        return "cache"
    return "off"


nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "remove"
EXECUTE_NB = nb_execution_mode != "off"

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
BINDER_LINK = (
    f"https://mybinder.org/v2/gh/ComPWA/{REPO_NAME}/{BRANCH}?filepath=docs/usage"
)
myst_substitutions = {
    "branch": BRANCH,
    "EXECUTE_NB": EXECUTE_NB,
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <compwa-org:develop:Jupyter Notebooks>` to interactively
modify the parameters.
```
""",
}
myst_update_mathjax = False
suppress_warnings = [
    # skipping unknown output mime type: application/json
    # https://github.com/ComPWA/ampform/runs/8132373732?check_suite_focus=true#step:5:127
    "mystnb.unknown_mime_type",
]

# Settings for sphinx_comments
comments_config = {
    "hypothesis": True,
    "utterances": {
        "repo": f"ComPWA/{REPO_NAME}",
        "issue-term": "pathname",
        "label": "📝 Docs",
    },
}

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}


# Specify bibliography style
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
    def __init__(self) -> None:
        super().__init__(abbreviate_names=True)

    def format_names(self, role: Entry, as_sentence: bool = True) -> Node:
        formatted_names = names(role, sep=", ", sep2=" and ", last_sep=", and ")
        if as_sentence:
            return sentence[formatted_names]
        else:
            return formatted_names

    def format_eprint(self, e: Entry) -> Node:
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


def remove_http(url: str) -> str:
    to_remove = ["https://", "http://"]
    for remove in to_remove:
        url = url.replace(remove, "")
    return url


register_plugin("pybtex.style.formatting", "unsrt_et_al", MyStyle)
