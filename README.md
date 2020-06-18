[![PyPI](https://badge.fury.io/py/expertsystem.svg)](https://pypi.org/project/expertsystem)
[![Travis CI](https://travis-ci.com/ComPWA/expertsystem.svg?branch=master)](https://travis-ci.com/ComPWA/expertsystem)
[![Test coverage](https://codecov.io/gh/ComPWA/expertsystem/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/expertsystem)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db355758fb0e4654818b85997f03e3b8)](https://www.codacy.com/gh/ComPWA/expertsystem)
[![Documentation build status](https://readthedocs.org/projects/expertsystem/badge/?version=latest)](https://pwa.readthedocs.io/projects/expertsystem/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
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
- **Conservation rules**
  - [x] Open-closed design
  - [x] Large set of predefined rules
    - [x] Spin/Angular momentum conservation
    - [x] Quark and Lepton flavour conservation (incl. isospin)
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
  - [x] XML (*old format for [ComPWA](https://compwa.github.io/)*)
  - [ ] YAML (*new format for
    [tensorwaves](https://pwa.readthedocs.io/projects/tensorwaves/en/latest)*)
