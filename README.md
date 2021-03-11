# PWA Expert System

[![Documentation build status](https://readthedocs.org/projects/expertsystem/badge/?version=latest)](https://expertsystem.readthedocs.io)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/expertsystem/stable?filepath=docs/usage)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/expertsystem/blob/stable)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)
[![GitPod](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ComPWA/expertsystem)
<br>
[![PyPI package](https://badge.fury.io/py/expertsystem.svg)](https://pypi.org/project/expertsystem)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/expertsystem)](https://pypi.org/project/expertsystem)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy.readthedocs.io)
[![CI status](https://github.com/ComPWA/expertsystem/workflows/CI/badge.svg)](https://github.com/ComPWA/expertsystem/actions?query=branch%3Amaster+workflow%3ACI)
[![Test coverage](https://codecov.io/gh/ComPWA/expertsystem/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/expertsystem)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db355758fb0e4654818b85997f03e3b8)](https://www.codacy.com/gh/ComPWA/expertsystem)
<br>
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)

Visit [expertsystem.rtfd.io](https://pwa.readthedocs.io/projects/expertsystem)
for an introduction to the Particle Wave Analysis Expert System!

For an overview of **upcoming releases and planned functionality**, see
[here](https://github.com/ComPWA/expertsystem/milestones?direction=asc&sort=title&state=open).

## Available features

- **Input**: Particle database
  - [x] Source of truth: PDG
  - [x] Predefined particle list file
  - [x] Option to overwrite and append with custom particle definitions
- **State transition graph**
  - [x] Feynman graph like description of the reactions
  - [x] Visualization of the decay topology
- **Conservation rules**
  - [x] Open-closed design
  - [x] Large set of predefined rules
    - [x] Spin/Angular momentum conservation
    - [x] Quark and Lepton flavor conservation (incl. isospin)
    - [x] Baryon number conservation
    - [x] EM-charge conservation
    - [x] Parity, C-Parity, G-Parity conservation
    - [ ] CP-Parity conservation
    - [x] Mass conservation
  - [x] Predefined sets of conservation rules representing Strong, EM, Weak
        interactions
- **PWA formalisms (for amplitude model generation)**
  - [x] Helicity formalism
  - [x] Canonical formalism
  - [ ] Tensor formalisms
- **Amplitude model**: Convert the state transition graphs to an amplitude
  model that is _mathematically expressed_ with [SymPy](https://docs.sympy.org)
  and can be _converted to any backend_ (see
  [`tensorwaves`](http://tensorwaves.rtfd.io)).

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
