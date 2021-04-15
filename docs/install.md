# Installation

The fastest way of installing this package is through PyPI:

```shell
python3 -m pip install ampform
```

This installs the [latest, stable release](https://pypi.org/project/ampform)
that you can find on the
[`stable`](https://github.com/ComPWA/ampform/tree/stable) branch. The latest
version on the [`main`](https://github.com/ComPWA/ampform/tree/main) branch can
be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/ampform@main
```

In that case, however, we highly recommend using the more dynamic,
{ref}`'editable installation' <pwa:develop:Editable installation>` instead.
This goes as follows:

1. Get the source code (see {doc}`pwa:software/git`):

   ```shell
   git clone https://github.com/ComPWA/ampform.git
   cd ampform
   ```

2. **[Recommended]** Create a virtual environment (see
   {ref}`here <pwa:develop:Virtual environment>`).

3. Install the project in
   {ref}`'editable installation' <pwa:develop:Editable installation>`, as well
   as {ref}`optional dependencies <pwa:develop:Optional dependencies>` for the
   developer:

   ```shell
   python3 -m pip install -e .[dev]
   ```

   :::{dropdown} Pinning dependencies

   In order to install the _exact same versions_ of the dependencies with which
   the framework has been tested, use the provided
   [constraints files](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
   for the specific Python version `3.x` you are using:

   ```shell
   python3 -m pip install -c .constraints/py3.x.txt -e .[dev]
   ```

   :::

That's all! Have a look at the {doc}`/usage` page to try out the package, and
see {doc}`pwa:develop` for tips on how to work with this 'editable' developer
setup!
