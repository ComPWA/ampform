.. cspell:ignore literalinclude

Interactive examples
====================

This page exposes some of the functionality of the `expertsystem` as online
utilities.

.. thebe-button::

Quantum number search
---------------------

The `.load_pdg` function creates a `.ParticleCollection` containing the latest
PDG info. Its `~.ParticleCollection.find` and `~.ParticleCollection.filter`
methods allows you to quickly look up the quantum numbers of a particle and,
vice versa, look up particle candidates based on a set of quantum numbers.

.. margin::

  Use `~.ParticleCollection.find` to search for a `.Particle` by name or by ID
  `as defined by the PDG
  <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-monte-carlo-numbering.pdf>`_.

.. code-block:: python
  :class: thebe, thebe-init

  import expertsystem as es
  pdg = es.io.load_pdg()

  pdg.find(22)
  pdg.find("Delta(1920)++")

.. margin::

  `~.ParticleCollection.filter` can perform any type of search using a `lambda
  <https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions>`_.
  Have a look at `.Particle` for the available properties.

.. code-block:: python
  :class: thebe, thebe-init

  subset = pdg.filter(
    lambda p:
    p.spin in [2.5, 3.5, 4.5]
    and p.name.startswith("N")
  )
  subset.names

.. tip:: See :doc:`/usage/particles`


Check allowed reactions
-----------------------

.. margin::

  The :code:`allowed_interactions` determine which rules are checked. For
  instance, if you allow `~.InteractionTypes.Weak` interactions, the check on
  :math:`C`-parity is not performed and the `expertsystem` would consider this
  reaction to be allowed.

The `expertsystem` can be used to `~.reaction.check` whether a transition
between an initial and final state is valid. If a solution is allowed,
`.check_reaction` returns a `set` of names of the allowed intermediate states,
if not, it will raise a `ValueError` containing the violated conservation
rules:

.. code-block:: python
  :class: thebe, thebe-init

  import expertsystem as es

  es.check_reaction(
    initial_state="pi0",
    final_state=["gamma", "gamma", "gamma"],
    allowed_interactions="EM",
  )


Investigate intermediate resonances
-----------------------------------

.. margin::

  .. warning::
    The larger the number final state particles, the longer the computation
    time. Use the `.StateTransitionManager` directly for fine-tuning.

The `expertsystem` is designed to be a tool when doing Partial Wave Analysis.
It's main features are therefore the `.generate_transitions` and
`.generate_amplitudes` functions. Here's a small applet with which to visualize
these transitions online:

.. code-block:: python
  :class: thebe, thebe-init

  import expertsystem as es
  from graphviz import Source

  result = es.generate_transitions(
    initial_state=("J/psi(1S)", [-1, +1]),
    final_state=["p", "p~", "eta"],
    allowed_interaction_types="strong",
  )
  graphs = result.collapse_graphs()
  Source(es.io.convert_to_dot(graphs))

.. toggle::

  This example takes around **1 minute** to compute on Binder.

.. tip:: See :doc:`/usage` and :doc:`/usage/workflow`
