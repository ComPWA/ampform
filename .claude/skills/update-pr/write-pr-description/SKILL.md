---
name: write-pr-description
description: Analyse git diff and draft a structured PR description with categorised changes and a conventional commit message block.
disable-model-invocation: true
user-invocable: true
---

Analyse the diff between the current branch and `main`, then draft a structured PR description.

## Step 1 - Gather the diff

Run these commands in parallel:

```shell
git log main..HEAD --oneline
git diff main...HEAD --stat
git diff main...HEAD -- src/ tests/ docs/ benchmarks/ pyproject.toml
```

Also fetch the current PR body so you know what is already there:

```shell
gh pr view --json number,body
```

## Step 2 - Categorise the changes

Sort every meaningful change into one of these categories. Omit a section entirely if it has no entries:

<!-- prettier-ignore-start -->
| Section header              | What belongs here                                        |
| --------------------------- | -------------------------------------------------------- |
| **✨ New features**         | New capabilities that did not exist before               |
| **⚙️ Enhancements**         | Improvements to existing behaviour, performance, UX      |
| **⚠️ Interface changes**    | **Breaking** changes to public names: renamed or removed functions, classes, protocols, modules, config keys; removed/renamed arguments, a new required argument, or a reordered signature that breaks existing callers |
| **❗ Behavioral changes**   | Changes that silently alter output or runtime behaviour  |
| **🐛 Bug fixes**            | Corrections to incorrect behaviour                       |
| **🖱️ Developer experience** | Test additions, tooling, CI, dependency updates          |
| **🔨 Maintenance**          | Refactors, type hygiene, code cleanup, lock-file/dependency upkeep |
| **📝 Documentation**        | Documentation-only changes (docstrings, `docs/`, README) |
<!-- prettier-ignore-end -->

Skip maintenance-only commits that do not affect the effective PR content, such as lock-file-only bumps, whitespace, or reverts within this PR.

Only breaking interface changes belong in **⚠️ Interface changes**. Backward-compatible additions are not interface changes: a new optional argument, a new public function/class, or a new optional config key is part of the **✨ New features** or **⚙️ Enhancements** entry that introduced it. The **⚠️ Interface changes** section exists to warn callers about what will break, so reserve it for exactly that.

Only public names count. Renaming or changing a private, underscore-prefixed name is an internal refactor no external caller can depend on, so it does not belong in **⚠️ Interface changes**. Leave it out, or include it under **🔨 Maintenance** only when the refactor is noteworthy in its own right.

Also omit changes that a reviewer already assumes from a headline entry:

- Call-site updates forced by a rename or signature change.
- Test or fixture additions that merely cover the new feature, rather than testing something independently noteworthy.
- Notebook, example, or doc code updated only to track new syntax.

The test is whether the entry carries information beyond "we did the obvious follow-up". Keep it only when it stands on its own: a new reusable test helper or harness, a genuinely new doc page or guide, a migration the reader must act on, or behaviour discovered while updating callers.

This filtering also governs labels: `/apply-pr` syncs PR labels from sections that have content, so a section omitted here drops its label automatically. The `🖱️ DX`, `📝 Docs`, and `🔨 Maintenance` labels should appear only when the PR carries DX/docs/maintenance work that is noteworthy in its own right.

## Step 3 - Decide what needs updating

Split the existing PR body at the first `##` heading:

- Preamble: everything before the first `##` heading, such as to-do checklists, `Closes #NNN` lines, prose notes, or images. Preserve this verbatim at the top of the new body.
- Release notes: everything from the first `##` heading onward. Regenerate this part.

Then update only the release notes:

- If an existing item is no longer in the diff, remove it.
- If new significant commits are present that are not yet reflected, add them.
- If the release notes already cover everything accurately, confirm and stop.

## Step 4 - Write the PR body

Structure:

1. Keep any existing `Closes #NNN` lines at the very top.
2. One `## Heading` section per category, with bullet points explaining what changed and why it matters.
3. For **Interface changes**, use a table with columns **Old name** and **New name** (or "removed").
4. Close with a `## Squash commit messages` section containing a fenced ` ```text ` block of conventional commit messages, one per logical change. This block becomes the squash-merge commit body, so it should list only the changes not already described by the PR title. The most prominent change becomes the title in `/apply-pr`, which then prunes any duplicate entry from this block; it is fine to list every logical change here and let that step reconcile them.

## Step 5 - Save draft to handoff file

Write the complete PR body to disk so `/apply-pr` can read it directly rather than reconstructing it from context:

```shell
cat > .claude/skills/update-pr/.pr_body.md << 'EOF'
<body>
EOF
```

## Next steps

Pass the draft PR body to `/format-commit-messages` to validate the commit message block.
