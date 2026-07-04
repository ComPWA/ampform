---
name: apply-pr
description: Finalise PR title and apply the validated PR description to GitHub.
disable-model-invocation: true
user-invocable: true
---

Finalise the PR title and push the description to GitHub.

## Step 1 - Write the PR title

Derive a single conventional commit message that summarises the most important change in the PR. Use the same type prefix conventions as the squash commit messages block (`FEAT`, `ENH`, `FIX`, etc.) in upper case, followed by a colon and an imperative-mood description. Keep it to at most 50 characters. GitHub also appends ` (#NNNN)` when squash-merging, so aim for 42 characters or fewer when possible.

The title's type prefix must agree with the PR's primary section. Each section maps to exactly one keyword:

| Section                     | Keyword    |
| --------------------------- | ---------- |
| **✨ New features**         | `FEAT`     |
| **⚙️ Enhancements**         | `ENH`      |
| **⚠️ Interface changes**    | `BREAK`    |
| **❗ Behavioral changes**   | `BEHAVIOR` |
| **🐛 Bug fixes**            | `FIX`      |
| **🖱️ Developer experience** | `DX`       |
| **🔨 Maintenance**          | `MAINT`    |
| **📝 Documentation**        | `DOC`      |

If the existing title already matches this format, agrees with the primary section, and accurately reflects the most important change, keep it unchanged. Otherwise re-derive it. In particular, if the PR was re-categorised, update the title so its prefix matches the new section rather than leaving the previous prefix in place.

## Step 1b - Reconcile the squash messages with the title

The title becomes the squash-merge subject, and the `## Squash commit messages` block becomes its body. The body must therefore list only the additional effective changes that the title does not already describe; never repeat the change captured by the title. Once the title is fixed:

- Remove any entry from the block that is already fully described by the title.
- If that empties the block, remove the `## Squash commit messages` section entirely.
- Re-run `/format-commit-messages` on the pruned block if it still has entries.

Validate the title before continuing:

```shell
python3 .claude/skills/update-pr/validate_commit_messages.py --title "<title>"
```

Fix any reported issues and re-run until it passes.

## Step 2 - Apply to GitHub

```shell
PR=$(gh pr view --json number -q .number)
gh pr edit "$PR" --title "<title>" --body "$(cat .claude/skills/update-pr/.pr_body.md)"
```

## Step 3 - Sync labels

After applying the body, sync the PR labels to match the sections present in the description:

```shell
python3 .claude/skills/update-pr/sync_pr_labels.py "$PR"
```

The script applies labels for sections that have content (e.g. `✨ Feature`, `🐛 Bug`) and removes managed labels for sections that were removed or are now empty. Unmanaged labels such as `⚪ Duplicate` are never touched.
