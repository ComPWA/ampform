# AmpForm

[![Documentation build status](https://readthedocs.org/projects/ampform/badge/?version=latest)](https://ampform.readthedocs.io)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/ampform/stable?filepath=docs/usage)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/ampform/blob/stable)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)
[![GitPod](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ComPWA/ampform)
[![PyPI package](https://badge.fury.io/py/ampform.svg)](https://pypi.org/project/ampform)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ampform)](https://pypi.org/project/ampform)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy.readthedocs.io)
[![pytest](https://github.com/ComPWA/ampform/workflows/pytest/badge.svg)](https://github.com/ComPWA/ampform/actions?query=branch%3Amain+workflow%3Apytest)
[![Test coverage](https://codecov.io/gh/ComPWA/ampform/branch/main/graph/badge.svg)](https://codecov.io/gh/ComPWA/ampform)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/70fc5fb0f3954a9d82d142efeff4df31)](https://www.codacy.com/gh/ComPWA/ampform)
[![pre-commit](https://github.com/ComPWA/ampform/workflows/pre-commit/badge.svg)](https://github.com/ComPWA/ampform/actions?query=branch%3Amain+workflow%3Apre-commit)
[![Spelling checked](https://img.shields.io/badge/cspell-checked-brightgreen.svg)](https://github.com/streetsidesoftware/cspell/tree/master/packages/cspell)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)

Automatically formulate amplitude models in different formalisms. Visit
[ampform.rtfd.io](https://ampform.readthedocs.io) for more info!

For an overview of **upcoming releases and planned functionality**, see
[here](https://github.com/ComPWA/ampform/milestones?direction=asc&sort=title&state=open).

## Available features

- **PWA formalisms (for amplitude model generation)**
  - [x] Helicity formalism
  - [x] Canonical formalism
  - [ ] Tensor formalisms
- **Amplitude model**: Convert the state transition graphs to an amplitude
  model that is _mathematically expressed_ with [SymPy](https://docs.sympy.org)
  and can be _converted to any backend_ (see
  [`tensorwaves`](https://tensorwaves.rtfd.io)).

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
