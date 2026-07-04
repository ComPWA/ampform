#!/bin/bash
# Detect public API, Poe task, and documentation workflow changes between main and HEAD.
# Prints what to manually verify; exits 1 if relevant changes are found.

set -euo pipefail

changed=$(git diff --name-only main...HEAD)
has_changes=0

if echo "$changed" | grep -q "pyproject.toml"; then
    echo "pyproject.toml changed - check for:"
    echo "   - New or removed package metadata, extras, dependency groups, or entry points"
    echo "   - New, removed, or renamed Poe tasks in [tool.poe.*]"
    echo "   - Changed pytest, Ruff, coverage, or ty configuration"
    echo "   Run: poe --help"
    has_changes=1
fi

if echo "$changed" | grep -qE "^src/ampform/"; then
    echo "src/ampform changed - verify public API, symbolic behavior, and docs still match CLAUDE.md"
    has_changes=1
fi

if echo "$changed" | grep -qE "^(docs/|README.md|CONTRIBUTING.md)"; then
    echo "Documentation changed - verify notebook/doc build guidance still matches CLAUDE.md"
    has_changes=1
fi

if [ "$has_changes" -eq 0 ]; then
    echo "No interface-relevant files changed"
fi

exit $has_changes
