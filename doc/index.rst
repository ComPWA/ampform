.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/ComPWA/expertsystem/master?filepath=examples%2Fquickstart.ipynb

.. image:: https://badge.fury.io/py/expertsystem.svg
  :alt: PyPI
  :target: https://pypi.org/project/expertsystem

.. image:: https://img.shields.io/pypi/pyversions/expertsystem
  :alt: PyPI - Python Version
  :target: https://pypi.org/project/expertsystem

.. image:: https://github.com/ComPWA/expertsystem/workflows/CI/badge.svg
  :alt: CI status
  :target: https://github.com/ComPWA/expertsystem/actions?query=branch%3Amaster+workflow%3A%22CI%22

.. image:: https://codecov.io/gh/ComPWA/expertsystem/branch/master/graph/badge.svg
  :alt: Test Coverage
  :target: https://codecov.io/gh/ComPWA/expertsystem

.. image:: https://api.codacy.com/project/badge/Grade/db355758fb0e4654818b85997f03e3b8
  :alt: Codacy Badge
  :target: https://www.codacy.com/gh/ComPWA/expertsystem

.. image:: https://readthedocs.org/projects/expertsystem/badge/?version=latest
  :alt: Documentation build status
  :target: https://expertsystem.readthedocs.io

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
  :target: https://github.com/pre-commit/pre-commit
  :alt: pre-commit

.. image:: https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265
  :alt: Code style: Prettier
  :target: https://prettier.io/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :alt: Code style: black
  :target: https://github.com/psf/black

|

Welcome to PWA Expert System!
=============================

The two purposes of the Partial Wave Analysis Expert System are to:

1. validate a particle reaction, based on given information. E.g.: Can a :math:`\pi^0`
   decay into 1, 2, 3 :math:`\gamma` particles?
2. create partial wave analysis amplitude models, based on basic information of
   a reaction, for instance, an amplitude model for :math:`J/\psi →
   \gamma\pi^0\pi^0` in the helicity or canonical formalism.

The user only has to provide a basic information of the particle reaction, such
as an initial state and a final state. Helper functions provide easy ways to
configure the system, but the user still has full control. The expert system
then constructs several hypotheses for what happens during the transition from
initial to final state.

Internal design
---------------

Internally, the PWA Expert System consists of three major components.

1. State Transition Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

A `.StateTransitionGraph` is a `directed graph
<https://en.wikipedia.org/wiki/Directed_graph>`_ that consists of **nodes** and
**edges**. In a directed graph, each edge must be connected to at least one
node (in correspondence to Feynman graphs). This way, a graph describes the
transition from one state to another.

- The edges correspond to particles/states, in other words a collection of
  properties such as the quantum numbers that characterize the particle state.

- Each node represents an interaction and contains all information for the
  transition of this specific step. Most importantly, a node contains a
  collection of conservation rules that have to be satisfied. An interaction
  node has :math:`M` ingoing lines and :math:`N` outgoing lines, where
  :math:`M,N \in \mathbb{Z}`, :math:`M > 0, 𝑁 > 0`.

2. Conservation Rules
~~~~~~~~~~~~~~~~~~~~~

The central component of the expert system are the :mod:`conservation rules
<.conservation_rules>`. They belong to individual nodes and receive properties
about the node itself, as well as properties of the ingoing and outgoing edges
of that node. Based on those properties the conservation rules determine
whether edges pass or not.

3. Solvers
~~~~~~~~~~

The propagation of the correct state properties through the graph is done by
solvers. New properties are set for intermediate edges and interaction nodes
and their validity is checked with the conservation rules.

Workflow of the Expert System
-----------------------------

1. Preparation

   1.1. Build all possible topologies. A **topology** is represented by a
   :ref:`graph <index:1. State Transition Graphs>`, in which the edges and
   nodes are empty (no particle information).

   1.2. Fill the topology graphs with the user provided information. Typically
   these are the graph's ingoing edges (initial state) and outgoing edges
   (final state).

2. Solving

   2.1. *Propagate* quantum number information through the complete graph while
   respecting the specified conservation laws. Information like mass is not
   used in this first solving step.

   2.2. *Clone* graphs while inserting concrete matching particles for the
   intermediate edges (mainly adds the mass variable).

   2.3. *Validate* the complete graphs, so run all conservation law check that
   were postponed from the first step.

3. Generate an amplitude model, e.g. helicity or canonical amplitude.


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   install
   usage
   contribute


.. toctree::
   :maxdepth: 1
   :hidden:

   api
   adr


expertsystem API
================

* :ref:`General Index <genindex>`
* :ref:`Python Modules Index <modindex>`
* :ref:`Search <search>`
