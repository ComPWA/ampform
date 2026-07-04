---
name: update-pr
description: Orchestrate the full PR description update: analyse the diff, validate commit messages, and apply to GitHub.
disable-model-invocation: true
user-invocable: true
---

Analyse the diff between the current branch and `main`, then write (or update) a structured PR description and apply it to GitHub.

Run the three sub-skills in sequence:

## Phase 1 - Write the PR description

Read `.claude/skills/update-pr/write-pr-description/SKILL.md` and follow its instructions.

Analyses the git diff, categorises changes, and drafts the PR body including the commit message block.

## Phase 2 - Validate commit messages

Read `.claude/skills/update-pr/format-commit-messages/SKILL.md` and follow its instructions.

Validates the commit message block against strict formatting rules using a deterministic script. Fix any reported issues before continuing.

## Phase 3 - Apply to GitHub

Read `.claude/skills/update-pr/apply-pr/SKILL.md` and follow its instructions.

Derives the PR title and pushes the validated description to GitHub.
