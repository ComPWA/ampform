.. cspell:ignore literalinclude

Interactive examples
====================

This page exposes some of the functionality of the expertsystem as online
utilities.

Particle database
-----------------

The `.load_pdg` function creates a `.ParticleCollection` instance containing
the latest PDG info. Its `~.ParticleCollection.find` and
`~.ParticleCollection.filter` methods allows you to quickly look up the quantum
numbers of a particle and, vice versa, look up particle candidates based on a
set of quantum numbers.

.. code-block:: python
  :class: thebe, thebe-init

  from expertsystem import io
  pdg = io.load_pdg()

  pdg.find(22)  # by PID
  pdg.find("Delta(1920)++")  # by name

.. code-block:: python
  :class: thebe, thebe-init

  subset = pdg.filter(
    lambda p:
    p.spin in [2.5, 3.5, 4.5]
    and p.name.startswith("N")
  )
  {p.name for p in subset}

.. thebe-button::

The `~.ParticleCollection.filter` function can perform any type of search. For
available search properties, have a look at properties of `.Particle` class.
For more info on the search syntax, read more about `lambda functions in Python
<https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions>`_.
