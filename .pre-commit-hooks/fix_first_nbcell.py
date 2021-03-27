# cspell:ignore nargs
"""Add install statements to first cell in a Jupyter notebook.

Google Colaboratory does not install a package automatically, so this has to be
done through a code cell. At the same time, this cell needs to be hidden from
the documentation pages, when viewing through Jupyter Lab (Binder), and when
viewing Jupyter slides.

Additionally, this scripts sets the IPython InlineBackend.figure_formats option
to SVG. This is because the Sphinx configuration can't set this externally.
"""

import argparse
import configparser
import sys
from typing import Optional, Sequence

import nbformat  # type: ignore

cfg = configparser.ConfigParser()
cfg.read("setup.cfg")

PACKAGE_NAME = cfg["metadata"]["name"]
EXPECTED_CELL_CONTENT = f"""
%%capture
%config Completer.use_jedi = False
%config InlineBackend.figure_formats = ['svg']

# Install on Google Colab
import subprocess
import sys

install_packages = "google.colab" in str(get_ipython())
if install_packages:
    for package in ["{PACKAGE_NAME}", "graphviz"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )
"""

EXPECTED_CELL_METADATA = {
    "jupyter": {"source_hidden": True},
    "slideshow": {"slide_type": "skip"},
    "tags": ["remove-cell"],
}


def fix_first_cell(
    filename: str, new_content: str, replace: bool = False
) -> None:
    notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
    new_cell = nbformat.v4.new_code_cell(
        new_content,
        metadata=EXPECTED_CELL_METADATA,
    )
    del new_cell["id"]  # following nbformat_minor = 4
    if replace:
        notebook["cells"][0] = new_cell
    else:
        notebook["cells"] = [new_cell] + notebook["cells"]
    nbformat.validate(notebook)
    nbformat.write(notebook, filename)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace first cell instead of prepending a new cell.",
    )
    args = parser.parse_args(argv)

    expected_cell_content = EXPECTED_CELL_CONTENT.strip("\n")
    exit_code = 0
    for filename in args.filenames:
        fix_first_cell(
            filename,
            new_content=expected_cell_content,
            replace=args.replace,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
