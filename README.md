[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/expertsystem/master?filepath=examples%2Fquickstart.ipynb)
[![PyPI](https://badge.fury.io/py/expertsystem.svg)](https://pypi.org/project/expertsystem)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/expertsystem)](https://pypi.org/project/expertsystem)
[![Travis CI](https://travis-ci.com/ComPWA/expertsystem.svg?branch=master)](https://travis-ci.com/ComPWA/expertsystem)
[![Test coverage](https://codecov.io/gh/ComPWA/expertsystem/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/expertsystem)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db355758fb0e4654818b85997f03e3b8)](https://www.codacy.com/gh/ComPWA/expertsystem)
[![Documentation build status](https://readthedocs.org/projects/expertsystem/badge/?version=latest)](https://pwa.readthedocs.io/projects/expertsystem/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PWA Expert System

Visit
[expertsystem.rtfd.io](https://pwa.readthedocs.io/projects/expertsystem/en/latest/)
for an introduction to the Particle Wave Analysis Expert System!

## Available features

- **Input**: Particle database
  - [ ] Source of truth: PDG
  - [x] Predefined particle list file
  - [x] Option to overwrite and append with custom particle definitions
- **State transition graph**
  - [x] Feynman graph like description of the reactions
  - [ ] Visualization of the topology
        ([`graphviz`](https://pypi.org/project/graphviz/))
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
- **Output**: Write transition graph to human-readable recipe file
  - [x] XML (_old format for [ComPWA](https://compwa.github.io/)_)
  - [x] YAML (_new format for
        [tensorwaves](https://pwa.readthedocs.io/projects/tensorwaves/en/latest)_)
