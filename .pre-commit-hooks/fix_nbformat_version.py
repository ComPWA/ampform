# cspell:ignore nargs
"""Set nbformat minor version to 4.

nbformat adds random cell ids since version 5.x. This is annoying for git
diffs. The solution is to set the version to v4 and removes those cell ids.
"""

import argparse
import sys
from typing import Optional, Sequence

import nbformat  # type: ignore


def set_nbformat_version(filename: str) -> None:
    notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
    if notebook["nbformat_minor"] != 4:
        notebook["nbformat_minor"] = 4
        nbformat.write(notebook, filename)


def remove_cell_ids(filename: str) -> None:
    notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
    for cell in notebook["cells"]:
        if "id" in cell:
            del cell["id"]
    nbformat.write(notebook, filename)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("filenames", nargs="*", help="Filenames to fix.")
    args = parser.parse_args(argv)
    exit_code = 0
    for filename in args.filenames:
        set_nbformat_version(filename)
        remove_cell_ids(filename)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
