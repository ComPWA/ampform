"""Pin developer requirements to a constraint file with ``pip-tools``.

See `Constraints Files
<https://pip.pypa.io/en/stable/user_guide/#constraints-files>_ and `pip-tools
<https://github.com/jazzband/pip-tools>`_.
"""

import os
import re
import subprocess
import sys

PYTHON_VERSION = ".".join(map(str, sys.version_info[:2]))
CONSTRAINTS_DIR = ".constraints"
OUTPUT_FILE = f"{CONSTRAINTS_DIR}/py{PYTHON_VERSION}.txt"


def upgrade_constraints_file() -> None:
    os.makedirs(CONSTRAINTS_DIR, exist_ok=True)
    pip_compile_command = " ".join(
        [
            "pip-compile",
            "--extra dev",
            "--upgrade",
            "--no-annotate",
            f'-o "{OUTPUT_FILE}"',
        ]
    )
    print("Running the following command:")
    print(f"   {pip_compile_command}")
    subprocess.call(pip_compile_command, shell=True)


def remove_extras_syntax() -> None:
    """Remove extras syntax package[extras] from constraints file.

    pip-compile contains a bug that inserts extras syntax, e.g.:

    .. code-block:: pip-requirements

        coverage[toml]==5.5

    pip cannot handle this in a constraints file, so this has to be removed.
    """

    def remove_extras(line: str) -> str:
        return re.sub(
            r"^(.*)\[.+\]==(.*$)",
            r"\1==\2",
            line,
        )

    def perform_replacements(line: str) -> str:
        replacements = {
            "typing_extensions": "typing-extensions",
        }
        for old, new in replacements.items():
            line = line.replace(old, new)
        return line

    with open(OUTPUT_FILE) as stream:
        lines = stream.readlines()
    new_lines = map(remove_extras, lines)
    new_lines = map(perform_replacements, new_lines)
    new_content = "".join(new_lines)
    with open(OUTPUT_FILE, "w") as stream:
        stream.write(new_content)


if "__main__" in __name__:
    upgrade_constraints_file()
    remove_extras_syntax()
