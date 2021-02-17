# Stubs for external libraries

To speed up linting and code navigation, it's often useful to generate
[stub files](https://mypy.readthedocs.io/en/stable/stubs.html) (`.pyi` files)
for the external libraries. You can do this using
[`stubgen`](https://mypy.readthedocs.io/en/stable/stubgen.html). For instance:

```shell
stubgen -p sympy -o typings
```

Alternatively, use [Pyright](https://github.com/microsoft/pyright) to generate
stub files that contain docstrings as well:

```shell
pyright --createstub sympy
```
