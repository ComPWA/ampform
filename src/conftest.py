"""Configuration for collecting doctests in the :code:`src` directory."""

from ampform._qrules import get_qrules_version  # ruff: ignore[import-private-name]

collect_ignore = []
if get_qrules_version() < (0, 10):
    collect_ignore.append("ampform/adapter/qrules.py")
