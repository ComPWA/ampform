---
name: format-commit-messages
description: Validate the conventional commit message block in a PR description against strict formatting rules (72-char limit, alphabetical order, valid type prefixes).
disable-model-invocation: true
user-invocable: true
---

Validate the `## Squash commit messages` block. Rules are enforced by the script below; it is the single source of truth.

## Validate

Extract the commit messages block and pipe it to the validation script:

```shell
python3 .claude/skills/update-pr/validate_commit_messages.py << 'EOF'
* FEAT: example line one
* FIX: example line two
EOF
```

The script reports lines that are too long, have invalid type prefixes, are not properly prefixed, or are out of order. Fix all reported issues and re-run until it passes.

## Next steps

Once validation passes, pass the complete PR body to `/apply-pr`.
