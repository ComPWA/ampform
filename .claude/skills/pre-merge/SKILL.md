---
name: pre-merge
description: Run the full local CI suite before merging a PR; check public API/task interface changes and keep CLAUDE.md in sync.
disable-model-invocation: true
user-invocable: true
---

Run the full CI suite locally before merging a PR and report any failures. Also keep CLAUDE.md in sync with any public API, task, or documentation workflow changes introduced by the PR.

## Step 1 - Check for interface changes

Run the interface check script to detect files that may affect public API, Poe tasks, or documentation workflows:

```shell
bash .claude/skills/pre-merge/check_interface_changes.sh
```

For each item the script flags, compare the current implementation against what CLAUDE.md documents:

- If `pyproject.toml` changed, check `[project]`, `[project.optional-dependencies]`, dependency groups, `[tool.poe.*]`, `[tool.pytest]`, `[tool.ruff.*]`, and `[tool.ty.*]`.
- If `src/ampform/` changed, verify whether public names, signatures, symbolic-expression behavior, or documented return values changed. Update docs/tests and call out breaking changes.
- If `docs/` changed, verify notebook execution and documentation build guidance still matches the project.

If anything differs - new tasks, renamed tasks, changed defaults, altered public API expectations, or changed documentation commands - update CLAUDE.md to match.

## Step 2 - Run CI

```shell
poe all
```

This runs style checks, coverage, documentation with forced notebook execution, link checking, and tests across supported Python versions. It can take 30-60 minutes. Run it with a timeout of at least 3600000 ms (60 minutes).

After it completes, report the results task by task:

- List each task that passed with a short confirmation.
- For each failure, explain what went wrong in plain language and suggest a concrete fix.
- If all tasks pass, confirm the branch is ready to merge.

## Step 3 - Update the PR description

Read `.claude/skills/update-pr/SKILL.md` and follow its instructions to analyse the diff and write a structured PR description with conventional commit messages.
