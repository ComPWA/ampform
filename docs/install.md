# Installation

[![PyPI package](https://badge.fury.io/py/ampform.svg)](https://pypi.org/project/ampform)
[![Conda package](https://anaconda.org/conda-forge/ampform/badges/version.svg)](https://anaconda.org/conda-forge/ampform)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform)](https://pypi.org/project/ampform)

## Quick installation

The fastest way of installing this package is through PyPI or Conda:

::::{tab-set}
:::{tab-item} PyPI

```shell
python3 -m pip install ampform
```

:::
:::{tab-item} Conda

```shell
conda install -c conda-forge ampform
```

:::
::::

This installs the [latest release](https://github.com/ComPWA/ampform/releases) that you
can find on the [`stable`](https://github.com/ComPWA/ampform/tree/stable) branch.

Optionally, you can install the dependencies required for
{doc}`visualizing topologies <qrules:usage/visualize>` with the following
{ref}`optional dependency syntax <compwa:develop:Optional dependencies>`:

```shell
pip install ampform[viz]  # installs ampform with graphviz
```

The latest version on the [`main`](https://github.com/ComPWA/ampform/tree/main) branch
can be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform@main
```

## Developer installation

:::{include} ../CONTRIBUTING.md
:start-after: **[compwa.github.io/develop](https://compwa.github.io/develop)**!
:end-before: If the repository provides a Tox configuration
:::

That's all! Have a look at {doc}`/usage` to try out the package. You can also have a look at {doc}`compwa:develop` for tips on how to work with this 'editable' developer setup!
