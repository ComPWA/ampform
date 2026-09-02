# How to contribute?

[![Open in Visual Studio Code](https://img.shields.io/badge/vscode-open-blue?logo=visualstudiocode)](https://github.dev/ComPWA/ampform)

> [!TIP]
> This package is part of the [ComPWA Organization](https://github.com/ComPWA). For more information about how to contribute to the packages, go to **[compwa.github.io/develop](https://compwa.github.io/develop)**!

To contribute to the project, you need to install the package in a virtual environment. This can be done best with [`uv`](https://docs.astral.sh/uv) (see installation instructions [here](https://docs.astral.sh/uv/getting-started/installation)). For this, you first need to get the source code with [Git](https://git-scm.com):

```shell
git clone https://github.com/ComPWA/ampform
cd ampform
```

Now it's simply a matter of creating and activating the [virtual environment](https://docs.astral.sh/uv/pip/environments) with [`uv sync`](https://docs.astral.sh/uv/reference/cli/#uv-sync). The dependencies for the project are 'pinned' in each commit through the [`uv.lock` file](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile).

```shell
uv sync --all-extras
source .venv/bin/activate
```

Formatting and linting checks are automatically performed when committing changes. This is done with [pre-commit](https://pre-commit.com). To install the hooks in your local repository, run install `pre-commit` with `uv`:

```shell
uv tool install pre-commit --with pre-commit-uv --force-reinstall --python=3.13
```

and [`pre-commit install`](https://pre-commit.com/#3-install-the-git-hook-scripts) **once**:

```shell
pre-commit install --install-hooks
```

[Poe the Poet](https://poethepoet.natn.io) is used as a task runner. Install it globally (within your home folder) with `uv`:

```shell
uv tool install poethepoet --force-reinstall --python=3.13
```

You can see which local CI checks it defines by running

```shell
poe
```

For instance, all style checks can be run with

```shell
poe style
```
