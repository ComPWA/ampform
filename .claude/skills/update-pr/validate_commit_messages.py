#!/usr/bin/env python3
"""Validate a conventional commit message block from a PR description.

Rules enforced:
- Each line starts with '* ' (asterisk, space)
- Type prefix in UPPER CASE followed by colon and space
- Valid types: FEAT, ENH, FIX, BREAK, BEHAVIOR, DOC, MAINT, DX
- Total line length <= 72 characters (including '* ')
- Lines sorted alphabetically (case-insensitive, ignoring '* ')

PR title validation (--title):
- Length <= 50 characters (subject line limit; GitHub also appends
  ' (#NNNN)' on squash-merge, so keep titles well under 50 chars)

Reads from a file argument or stdin.
"""

from __future__ import annotations

import sys

VALID_TYPES = {"FEAT", "ENH", "FIX", "BREAK", "BEHAVIOR", "DOC", "MAINT", "DX"}
MAX_LINE_LENGTH = 72
MAX_TITLE_LENGTH = 50  # conventional subject-line limit


def validate_title(title: str) -> list[str]:
    """Validate a single PR title (no '* ' prefix required)."""
    errors: list[str] = []
    title = title.strip()
    if len(title) > MAX_TITLE_LENGTH:
        errors.append(
            f"Title: {len(title)} chars exceeds {MAX_TITLE_LENGTH}"
            f" (conventional subject-line limit): {title!r}"
        )
    if ":" not in title:
        errors.append(f"Title: missing colon after type prefix: {title!r}")
        return errors
    type_prefix = title.split(":")[0]
    if type_prefix not in VALID_TYPES:
        errors.append(
            f"Title: invalid type {type_prefix!r}"
            f" (valid: {', '.join(sorted(VALID_TYPES))})"
        )
    return errors


def validate(text: str) -> list[str]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    errors: list[str] = []

    for i, line in enumerate(lines, 1):
        if not line.startswith("* "):
            errors.append(f"Line {i}: must start with '* ': {line!r}")
            continue
        if len(line) > MAX_LINE_LENGTH:
            errors.append(
                f"Line {i}: {len(line)} chars exceeds {MAX_LINE_LENGTH}: {line}"
            )
        content = line[2:]
        if ":" not in content:
            errors.append(f"Line {i}: missing colon after type prefix: {line!r}")
            continue
        type_prefix = content.split(":")[0]
        if type_prefix not in VALID_TYPES:
            errors.append(
                f"Line {i}: invalid type {type_prefix!r}"
                f" (valid: {', '.join(sorted(VALID_TYPES))})"
            )

    sorted_lines = sorted(lines, key=lambda ln: ln.lower().removeprefix("* "))
    if lines != sorted_lines:
        errors.append("Lines are not alphabetically sorted. Expected order:")
        errors.extend(f"  {ln}" for ln in sorted_lines)

    return errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="file to validate (default: stdin)")
    parser.add_argument("--title", help="validate a single PR title string instead")
    args = parser.parse_args()

    if args.title:
        errors = validate_title(args.title)
    elif args.file:
        with open(args.file) as f:
            errors = validate(f.read())
    else:
        errors = validate(sys.stdin.read())

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    print("✓ Valid")
