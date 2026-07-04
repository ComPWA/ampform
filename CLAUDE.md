# CLAUDE.md

## Environment

The environment is managed through [uv](https://docs.astral.sh/uv), and activated automatically via [direnv](https://direnv.net).

When calling Python or CLI tools that have been installed into the environment, call them through `uv run` to ensure the correct environment is used.

## Style Checks

Style checks (autoformatting, linting, spelling, and type checking) run through [pre-commit](https://pre-commit.com) and are enforced on commit. To run all checks over all files manually:

```shell
poe style
```

Always run this and fix any issues before committing.

## Testing

Run the fast test suite (excludes `slow`-marked tests):

```shell
pytest
```

Run all unit tests, including slow tests, with parallel execution:

```shell
poe test
```

Run the coverage task:

```shell
poe cov
```

Doctests are enabled globally (`--doctest-modules`), so all docstring examples must be correct and executable.

The `tests/` tree mirrors the `src/ampform/` package hierarchy: a module's tests live at the same relative path, named `test_<module>.py`. For example, `src/ampform/dynamics/builder.py` is tested by `tests/dynamics/test_builder.py`, and `src/ampform/sympy/_decorator.py` is tested under `tests/sympy/decorator/`. Place new test files accordingly. The directory already encodes the package, so do not repeat it in the filename. Pytest uses `--import-mode=importlib`, so subpackages need no `__init__.py` and duplicate basenames across directories are fine.

## Comments

Treat inline code comments as a last resort, used only when intent cannot be expressed through the code itself. Always prefer mechanisms that carry syntactic meaning to Python, language navigation, editors, and Sphinx over a comment:

- Descriptive function and variable names.
- Type annotations and function signatures.
- Docstrings and doctests (`>>>`) - see [Docstrings](#docstrings) below.
- Extracting a well-named, documented helper function instead of an explanatory comment over a code block.
- In notebooks, add a Markdown cell or name a helper clearly rather than adding a comment that describes plumbing.

Reserve comments for genuinely non-obvious rationale, such as a physics convention, symbolic-algebra limitation, or numerical-stability reason that none of the above can capture. Keep them concise.

## Docstrings

- Do not add docstrings that merely restate a descriptive function name. Keep docstrings only when they add information the signature does not convey: parameter semantics, units, edge-case behavior, or non-obvious conventions.
- When a docstring would illustrate behavior with an example, write it as a doctest (`>>>`) rather than prose. Doctests are executed globally, so the example stays correct and doubles as a test. Prefer this for small pure functions where the example is cheap to run.
- A docstring may never consist of only a doctest. Always precede the doctest with a short description line, even a single sentence, so the docstring summary is meaningful.
- For a small pure function, prefer a doctest over a standalone test in `tests/`. Reserve `tests/` for cases that need real setup a doctest cannot express cleanly: fixtures, constructed objects, files on disk, parametrized sweeps, or slow symbolic expressions.

## Type Checking

The type checker is [`ty`](https://github.com/astral-sh/ty). It runs as part of `poe style` via pre-commit. For notebooks and other generated-style sources, overrides are configured under `[tool.ty.overrides]` in `pyproject.toml`; add new suppressions there rather than inline `# ty:ignore` comments unless the ignore is narrowly tied to a specific expression and cannot be captured cleanly by configuration.

## Public API

AmpForm is a library, not a command-line application. There are currently no `[project.scripts]` entry points. Treat exported modules, classes, functions, signatures, documented behavior, and generated API documentation as the public interface.

Before changing public names, argument order, defaults, symbolic expression structure, or documented return values, check downstream impact and update tests and docs in the same change. Breaking API changes should be called out explicitly in PR descriptions.

## Documentation

Build documentation without executing notebooks:

```shell
poe doc
```

Build documentation with cached notebook execution:

```shell
poe docnb
```

Build documentation and force re-execution of all notebooks:

```shell
poe docnb-force
```

Check external links:

```shell
poe linkcheck
```

Run all notebooks directly:

```shell
poe nb
```

Notebook and documentation examples are part of the tested interface. Keep them synchronized with the API and prefer executable examples over prose-only descriptions.

### Cell Folding

Notebooks under `docs/` are rendered into the Sphinx documentation, so a reader should see the results without scrolling through plumbing that produces them. Fold a code cell whose source is not itself the point of the notebook - imports, helper/builder definitions, model construction, lambdification, widget/slider setup, or plotting boilerplate - by giving it folded metadata:

- `metadata.jupyter.source_hidden = true` collapses the source in JupyterLab.
- `metadata.mystnb.code_prompt_show = "<short description>"` sets the label shown on the collapsed cell's toggle button in the rendered docs.
- `metadata.tags` containing `"hide-input"` folds the input in myst-nb. Add `"scroll-input"` for long cells. Use `"remove-input"` only when the source should be dropped entirely from the docs.

Do not fold a cell when its code is the thing the notebook is teaching. Keep those cells visible.

## Poe Tasks

Useful project tasks:

```shell
poe --help       # list all tasks
poe style        # lint, format, spell-check, and type-check
poe test         # run all unit tests
poe test-all     # run tests on each supported Python version
poe cov          # run coverage
poe benchmark    # run benchmark tests
poe doc          # build docs without notebook execution
poe docnb        # build docs with cached notebook execution
poe docnb-force  # build docs with forced notebook execution
poe nb           # run notebooks with nbmake
poe all          # run the full local CI suite
```

Before relying on a task interface, check `poe --help` or `pyproject.toml` if it may have changed.

## Commit Conventions

Use [conventional commit messages](https://compwa.github.io/develop#commit-conventions) with semantic keywords in upper case, followed by a colon, then an imperative-mood description:

```text
TYPE: description of what the commit does
```

| Keyword    | When to use                                        |
| ---------- | -------------------------------------------------- |
| `FEAT`     | New feature added to the package                   |
| `ENH`      | Improvement or optimization of an existing feature |
| `FIX`      | Bug fix                                            |
| `BREAK`    | Breaking API change                                |
| `BEHAVIOR` | Change that may affect framework output            |
| `DOC`      | Documentation improvement or addition              |
| `MAINT`    | Maintenance and upkeep                             |
| `DX`       | Developer experience improvement                   |
