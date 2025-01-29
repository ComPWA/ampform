from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import is_dataclass

from sphinx_api_relink.helpers import (
    get_branch_name,
    get_execution_mode,
    get_package_version,
    pin,
    pin_minor,
    set_intersphinx_version_remapping,
)

from ampform.sympy._decorator import get_sympy_fields  # noqa: PLC2701

sys.path.insert(0, os.path.abspath("."))
from _extend_docstrings import extend_docstrings  # noqa: PLC2701


def _get_excluded_members() -> list[str]:
    default_exclusions = {
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
    }
    for cls in [
        *_get_dataclasses_recursive("ampform"),
    ]:
        fields = get_sympy_fields(cls)
        arg_names = {f.name for f in fields}
        default_exclusions.update(arg_names)
    return sorted(default_exclusions)


def _get_dataclasses_recursive(module_name: str) -> list[type]:
    module = importlib.import_module(module_name)
    dataclass_list = _get_dataclasses(module)
    for _, submodule in inspect.getmembers(module, inspect.ismodule):
        if submodule.__name__.startswith(module_name):
            dataclass_list.extend(_get_dataclasses_recursive(submodule.__name__))
    return dataclass_list


def _get_dataclasses(module):
    dataclass_list = []
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and is_dataclass(obj):
            dataclass_list.append(obj)
    return dataclass_list


extend_docstrings()
set_intersphinx_version_remapping({
    "ipython": {
        "8.12.2": "8.12.1",
        "8.12.3": "8.12.1",
    },
    "ipywidgets": {
        "8.0.3": "8.0.5",
        "8.0.4": "8.0.5",
        "8.0.6": "8.0.5",
        "8.1.1": "8.1.2",
    },
    "mpl-interactions": {
        "0.24.1": "0.24.0",
        "0.24.2": "0.24.0",
    },
})

BRANCH = get_branch_name()
ORGANIZATION = "ComPWA"
PACKAGE = "ampform"
REPO_NAME = "ampform"
REPO_TITLE = "AmpForm"

BINDER_LINK = (
    f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?urlpath=lab/docs"
)
EXECUTE_NB = get_execution_mode() != "off"


add_module_names = False
api_github_repo = f"{ORGANIZATION}/{REPO_NAME}"
api_target_substitutions: dict[str, str | tuple[str, str]] = {
    "BuilderReturnType": ("obj", "ampform.dynamics.builder.BuilderReturnType"),
    "DecoratedClass": ("obj", "ampform.sympy.deprecated.DecoratedClass"),
    "DecoratedExpr": ("obj", "ampform.sympy.deprecated.DecoratedExpr"),
    "FourMomenta": ("obj", "ampform.kinematics.lorentz.FourMomenta"),
    "FourMomentumSymbol": ("obj", "ampform.kinematics.lorentz.FourMomentumSymbol"),
    "InteractionProperties": "qrules.quantum_numbers.InteractionProperties",
    "LatexPrinter": "sympy.printing.printer.Printer",
    "Literal[(-1, 1)]": "typing.Literal",
    "Literal[-1, 1]": "typing.Literal",
    "NumPyPrintable": ("class", "ampform.sympy.NumPyPrintable"),
    "NumPyPrinter": "sympy.printing.printer.Printer",
    "P": "typing.ParamSpec",
    "ParameterValue": ("obj", "ampform.helicity.ParameterValue"),
    "Particle": "qrules.particle.Particle",
    "ReactionInfo": "qrules.transition.ReactionInfo",
    "Slider": ("obj", "symplot.Slider"),
    "State": "qrules.transition.State",
    "StateTransition": "qrules.topology.Transition",
    "T": "typing.TypeVar",
    "Topology": "qrules.topology.Topology",
    "WignerD": "sympy.physics.quantum.spin.WignerD",
    "ampform.helicity._T": "typing.TypeVar",
    "ampform.sympy._decorator.ExprClass": ("obj", "ampform.sympy.ExprClass"),
    "sp.Basic": "sympy.core.basic.Basic",
    "sp.Expr": "sympy.core.expr.Expr",
    "sp.Float": "sympy.core.numbers.Float",
    "sp.Indexed": "sympy.tensor.indexed.Indexed",
    "sp.IndexedBase": "sympy.tensor.indexed.IndexedBase",
    "sp.Rational": "sympy.core.numbers.Rational",
    "sp.Symbol": "sympy.core.symbol.Symbol",
    "sp.acos": "sympy.functions.elementary.trigonometric.acos",
    "sympy.printing.numpy.NumPyPrinter": "sympy.printing.printer.Printer",
    "sympy.tensor.array.expressions.array_expressions.ArraySymbol": (
        "mod",
        "sympy.tensor.array.expressions",
    ),
}
api_target_types: dict[str, str] = {
    "RangeDefinition": "obj",
    "ampform.helicity.align.dpd.T": "obj",
}
author = "Common Partial Wave Analysis"
autodoc_default_options = {
    "exclude-members": ", ".join(_get_excluded_members()),
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join([  # noqa: FLY002
        "__call__",
    ]),
}
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
autosectionlabel_prefix_document = True
bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "unsrt_et_al"
bibtex_reference_style = "author_year"
codeautolink_concat_default = True
codeautolink_global_preface = """
import numpy
import numpy as np
import sympy as sp
from IPython.display import display
"""
comments_config = {
    "hypothesis": True,
    "utterances": {
        "repo": f"{ORGANIZATION}/{REPO_NAME}",
        "issue-term": "pathname",
        "label": "📝 Docs",
    },
}
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest
copyright = f"2020, {ORGANIZATION}"
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr/template.md",
    "tests",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_codeautolink",
    "sphinx_comments",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_hep_pdgref",
    "sphinx_pybtex_etal_style",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
]
generate_apidoc_package_path = f"../src/{PACKAGE}"
graphviz_output_format = "svg"
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%-d %B %Y"
html_logo = (
    "https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg"
)
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "_static/favicon.ico",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{PACKAGE}",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Conda",
            "url": f"https://anaconda.org/conda-forge/{PACKAGE}",
            "icon": "https://avatars.githubusercontent.com/u/22454001?s=100",
            "type": "url",
        },
        {
            "name": "Launch on Binder",
            "url": f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?urlpath=lab",
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/{ORGANIZATION}/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "logo": {"text": REPO_TITLE},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = REPO_TITLE
intersphinx_mapping = {
    "IPython": (f"https://ipython.readthedocs.io/en/{pin('IPython')}", None),
    "attrs": (f"https://www.attrs.org/en/{pin('attrs')}", None),
    "compwa": ("https://compwa.github.io", None),
    "compwa-report": ("https://compwa.github.io/report", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable", None),
    "ipywidgets": (f"https://ipywidgets.readthedocs.io/en/{pin('ipywidgets')}", None),
    "matplotlib": (f"https://matplotlib.org/{pin('matplotlib')}", None),
    "mpl_interactions": (
        f"https://mpl-interactions.readthedocs.io/en/{pin('mpl-interactions')}",
        None,
    ),
    "numpy": (f"https://numpy.org/doc/{pin_minor('numpy')}", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (f"https://qrules.readthedocs.io/{pin('qrules')}", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}
linkcheck_anchors = False
linkcheck_ignore = [
    "http://www.curtismeyer.com",
    "https://doi.org/10.1002",  # 403 for onlinelibrary.wiley.com
    "https://doi.org/10.1093",  # 403 for PTEP
    "https://doi.org/10.1103",  # 403 for Phys Rev D
    "https://doi.org/10.1155",  # 403 for hindawi.com
    "https://home.fnal.gov/~kutschke/Angdist/angdist.ps",
    "https://hss-opus.ub.ruhr-uni-bochum.de",
    "https://journals.aps.org/prd",  # 403 for Phys Rev D
    "https://physique.cuso.ch",
    "https://suchung.web.cern.ch",
    "https://www.bookfinder.com",
]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 2
myst_substitutions = {
    "branch": BRANCH,
    "EXECUTE_NB": EXECUTE_NB,
    "run_interactive": f"""
```{{margin}}
Run this notebook [on Binder]({BINDER_LINK}) or
{{ref}}`locally on Jupyter Lab <compwa:develop:Jupyter Notebooks>` to interactively
modify the parameters.
```
""",
}
myst_update_mathjax = False
nb_execution_allow_errors = False
nb_execution_mode = get_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "remove"
nitpick_ignore = [
    ("py:class", "ArraySum"),
    ("py:class", "ExprClass"),
    ("py:class", "MatrixMultiplication"),
    ("py:class", "ampform.sympy._decorator.SymPyAssumptions"),
]
nitpicky = True
primary_domain = "py"
project = REPO_TITLE
pygments_style = "sphinx"
release = get_package_version(PACKAGE)
source_suffix = {
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}
suppress_warnings = [
    "myst.domains",
    # skipping unknown output mime type: application/json
    # https://github.com/ComPWA/ampform/runs/8132373732?check_suite_focus=true#step:5:127
    "mystnb.unknown_mime_type",
]
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
version = get_package_version(PACKAGE)
