#!/usr/bin/env python3
"""Sync GitHub PR labels to match the sections present in the PR description.

Labels managed by this script (matched by leading emoji):
  ✨ Feature, ⚙️ Enhancement, ⚠️ Interface, ❗ Behavior,
  🐛 Bug, 🖱️ DX, 📝 Docs, 🔨 Maintenance

A label is applied when its section exists in the PR body AND has content
(at least one non-empty line under the heading before the next ## heading).
A managed label is removed when its section is absent or empty.
Unmanaged labels (e.g. ⚪ Duplicate, 💫 Good first issue) are never touched.

Usage:
  python3 sync_pr_labels.py <pr_number> [--repo <owner/repo>]
"""

import json
import re
import subprocess
from argparse import ArgumentParser

MANAGED_EMOJIS = {"✨", "⚙️", "⚠️", "❗", "🐛", "🖱️", "📝", "🔨"}


def leading_emoji(text: str) -> str:
    """Return the leading emoji token(s) of a string, or ''."""
    # Split on whitespace and take the first token; keep only if non-ASCII
    token = text.split(maxsplit=1)[0] if text.split() else ""
    return token if not token.isascii() else ""


def sections_with_content(body: str) -> set[str]:
    """Return the set of leading emojis for sections that have content."""
    found: set[str] = set()
    current_emoji = ""
    current_lines: list[str] = []

    for line in body.splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            if current_emoji and any(ln.strip() for ln in current_lines):
                found.add(current_emoji)
            current_emoji = leading_emoji(heading.group(1))
            current_lines = []
        elif current_emoji:
            current_lines.append(line)

    if current_emoji and any(ln.strip() for ln in current_lines):
        found.add(current_emoji)

    return found


def gh(*args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["gh", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def get_pr(repo: str, pr_number: int) -> tuple[str, list[str]]:
    """Return (body, current_label_names) for the PR."""
    data = json.loads(
        gh("pr", "view", str(pr_number), "--repo", repo, "--json", "body,labels")
    )
    body: str = data["body"] or ""
    labels: list[str] = [lbl["name"] for lbl in data["labels"]]
    return body, labels


def get_repo_labels(repo: str) -> dict[str, str]:
    """Return {emoji: full_label_name} for managed labels in the repo."""
    raw = json.loads(gh("label", "list", "--repo", repo, "-L", "50", "--json", "name"))
    result: dict[str, str] = {}
    for item in raw:
        name: str = item["name"]
        emoji = leading_emoji(name)
        if emoji in MANAGED_EMOJIS:
            result[emoji] = name
    return result


def sync(repo: str, pr_number: int) -> None:
    body, current_labels = get_pr(repo, pr_number)
    repo_labels = get_repo_labels(repo)

    active_emojis = sections_with_content(body)

    to_add = [
        repo_labels[e]
        for e in active_emojis
        if e in repo_labels and repo_labels[e] not in current_labels
    ]
    to_remove = [
        repo_labels[e]
        for e in MANAGED_EMOJIS
        if e in repo_labels
        and repo_labels[e] in current_labels
        and e not in active_emojis
    ]

    if to_add:
        gh(
            "pr",
            "edit",
            str(pr_number),
            "--repo",
            repo,
            "--add-label",
            ",".join(to_add),
        )
        print(f"Added: {', '.join(to_add)}")
    if to_remove:
        gh(
            "pr",
            "edit",
            str(pr_number),
            "--repo",
            repo,
            "--remove-label",
            ",".join(to_remove),
        )
        print(f"Removed: {', '.join(to_remove)}")
    if not to_add and not to_remove:
        print("Labels already in sync.")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", type=int)
    parser.add_argument("--repo", default="ComPWA/ampform")
    args = parser.parse_args()
    sync(args.repo, args.pr_number)


if __name__ == "__main__":
    main()
