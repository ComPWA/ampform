# Installation

[![PyPI package](https://badge.fury.io/py/ampform.svg)](https://pypi.org/project/ampform)
[![Conda package](https://anaconda.org/conda-forge/ampform/badges/version.svg)](https://anaconda.org/conda-forge/ampform)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform)](https://pypi.org/project/ampform)

AmpForm is available on [PyPI](https://pypi.org/project/ampform) and [conda-forge](https://anaconda.org/conda-forge/ampform), so you can install it with your favorite package manager:

::::{tab-set}
:sync-group: package-manager

:::{tab-item} uv
:sync: uv

```shell
uv add ampform
```

:::
:::{tab-item} Pixi
:sync: Pixi

```shell
pixi add ampform
```

:::
:::{tab-item} PyPI
:sync: PyPI

```shell
python3 -m pip install ampform
```

:::
:::{tab-item} Conda
:sync: Conda

```shell
conda install -c conda-forge ampform
```

:::
::::

This installs the [latest release](https://github.com/ComPWA/ampform/releases) that you can find on the [`stable`](https://github.com/ComPWA/ampform/tree/stable) branch.

Optionally, you can install the dependencies required for {doc}`visualizing topologies <qrules:usage/visualize>` with the following {ref}`optional dependency syntax <compwa:develop:Optional dependencies>`:

::::{tab-set}
:sync-group: package-manager

:::{tab-item} uv
:sync: uv

```shell
uv add 'ampform[viz]'
```

:::
:::{tab-item} Pixi
:sync: Pixi

```shell
pixi add ampform graphviz python-graphviz
```

:::
:::{tab-item} PyPI
:sync: PyPI

```shell
pip install 'ampform[viz]'
```

:::
:::{tab-item} Conda
:sync: Conda

```shell
conda install -c conda-forge ampform graphviz python-graphviz
```

:::
::::

The latest version on the [`main`](https://github.com/ComPWA/ampform/tree/main) branch can be installed as follows:

::::{tab-set}
:sync-group: package-manager

:::{tab-item} uv
:sync: uv

```shell
uv add git+https://github.com/ComPWA/ampform --branch main
```

:::
:::{tab-item} Pixi
:sync: Pixi

```shell
pixi add ampform --git https://github.com/ComPWA/ampform --branch main
```

:::
:::{tab-item} PyPI
:sync: PyPI

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform@main
```

:::
::::

## Developer installation

:::{include} ../CONTRIBUTING.md
:start-after: **[compwa.github.io/develop](https://compwa.github.io/develop)**!
:::

That's all! Have a look at {doc}`index` to try out the package. You can also have a look at {doc}`compwa:develop` for tips on how to work with this 'editable' developer setup!
