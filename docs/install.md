# Installation

[![PyPI package](https://badge.fury.io/py/ampform.svg)](https://pypi.org/project/ampform)
[![Conda package](https://anaconda.org/conda-forge/ampform/badges/version.svg)](https://anaconda.org/conda-forge/ampform)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform)](https://pypi.org/project/ampform)

The fastest way of installing this package is through PyPI or Conda:

:::{tabbed} PyPI

```shell
python3 -m pip install ampform
```

::::

:::{tabbed} Conda

```shell
conda install -c conda-forge ampform
```

:::

This installs the
[latest, stable release](https://github.com/ComPWA/ampform/releases) that you
can find on the [`stable`](https://github.com/ComPWA/ampform/tree/stable)
branch.

The latest version on the [`main`](https://github.com/ComPWA/ampform/tree/main)
branch can be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform@main
```

In that case, however, we highly recommend using the more dynamic
{ref}`'editable installation' <compwa-org:develop:Editable installation>`
instead. This goes as follows:

1. Get the source code:

   ```shell
   git clone https://github.com/ComPWA/ampform.git
   cd ampform
   ```

2. **[Recommended]** Create a virtual environment (see
   {ref}`here <compwa-org:develop:Virtual environment>`).

3. Install the project as an
   {ref}`'editable installation' <compwa-org:develop:Editable installation>`
   and install
   {ref}`additional packages <compwa-org:develop:Optional dependencies>` for
   the developer:

   ```shell
   python3 -m pip install -e .[dev]
   ```

   :::{dropdown} Pinning dependency versions

   In order to install the _exact same versions_ of the dependencies with which
   the framework has been tested, use the provided
   [constraints files](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
   for the specific Python version `3.x` you are using:

   ```shell
   python3 -m pip install -c .constraints/py3.x.txt -e .[dev]
   ```

   ```{seealso}

   {ref}`develop:Pinning dependency versions`

   ```

   :::

That's all! Have a look at the {doc}`/usage` page to try out the package. You
can also have a look at the {doc}`compwa-org:develop` page for tips on how to
work with this 'editable' developer setup!
