"""A rule based system that facilitates particle reaction analysis.

The main responsibility of the `expertsystem` is to validate particle reactions
as specified by the user. The user boundary conditions for a particle reaction
problem are for example the initial state, final state, and allowed
interactions.

A further responsibility is to build amplitude models, if a reaction is valid.
These models are based on the found solutions and represent the transition
probability of the process.

The `expertsystem` consists of three main components:

#. `particle`: a stand-alone submodule with which one can investigate specific
   quantum properties of `.Particle` instances (see :doc:`/usage/particles`).

#. `reaction`: the core of the `expertsystem` that computes which transitions
   (represented by a `.StateTransitionGraph`) are allowed between a certain
   initial and final state. Internally, the system propagates the quantum
   numbers defined by `particle` through the `.StateTransitionGraph`, while
   satisfying the rules define by the :mod:`.conservation_rules` module.

#. `amplitude`: a collection of tools to convert the `.StateTransitionGraph`
   solutions found by `reaction` into an `.AmplitudeModel`. This module is
   specifically designed to create amplitude model templates for PWA fitter
   packages.

Finally, the `.ui` module glues these modules together through facade functions
and its `~expertsystem.ui.StateTransitionManager`, while the `.io` module
provides tools that can read and write the objects of `particle`, `reaction`,
and `amplitude`.
"""


__all__ = [
    "amplitude",
    "io",
    "particle",
    "reaction",
    "ui",
]


from . import amplitude, io, particle, reaction, ui
