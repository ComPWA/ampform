"""A rule based system that facilitates particle reaction analysis.

The main responsibility of the `expertsystem` is to validate particle reactions
as specified by the user. The user boundary conditions for a particle reaction
problem are for example the initial state, final state, and allowed
interactions.

A further responsibility is to build amplitude models, if a reaction is valid.
These models are based on the found solutions and represent the transition
probability of the process.

The `expertsystem` consists of two main components:

  `expertsystem.reaction`
    ― the core of the `expertsystem` that computes which transitions
    (represented by a `.StateTransitionGraph`) are allowed between a certain
    initial and final state. Internally, the system propagates the quantum
    numbers defined by the `reaction.particle` module through the
    `.StateTransitionGraph`, while satisfying the rules define by the
    :mod:`.conservation_rules` module. See :doc:`/usage/reaction` and
    :doc:`/usage/particle`.

  `expertsystem.amplitude`
    ― a collection of tools to convert the `.StateTransitionGraph` solutions
    found by `reaction` into an `.HelicityModel`. This module is specifically
    designed to create amplitude model templates for :doc:`PWA fitter packages
    <pwa:software>`. See :doc:`/usage/amplitude`.

Finally, the `.io` module provides tools that can read and write the objects of
this framework.
"""


__all__ = [
    # Main modules
    "amplitude",
    "reaction",
    "io",
    # Facade functions
    "generate_transitions",
    "check_reaction_violations",
    "load_default_particles",
]


from . import amplitude, io, reaction
from .reaction.default_settings import ADDITIONAL_PARTICLES_DEFINITIONS_PATH

generate_transitions = reaction.generate
"""An alias to `.reaction.generate`."""

check_reaction_violations = reaction.check_reaction_violations
"""An alias to `.reaction.check_reaction_violations`."""


def load_default_particles() -> reaction.ParticleCollection:
    """Load the default particle list that comes with the `expertsystem`.

    Runs `.load_pdg` and supplements its output definitions from the file
    :download:`reaction/additional_definitions.yml
    </../src/expertsystem/reaction/additional_definitions.yml>`.
    """
    particles = reaction.load_pdg()
    additional_particles = io.load(ADDITIONAL_PARTICLES_DEFINITIONS_PATH)
    assert isinstance(additional_particles, reaction.ParticleCollection)
    particles.update(additional_particles)
    return particles
