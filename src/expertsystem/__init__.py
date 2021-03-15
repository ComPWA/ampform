"""A rule based system that facilitates particle reaction analysis.

The main responsibility of the `expertsystem` is to validate particle reactions
as specified by the user. The user boundary conditions for a particle reaction
problem are for example the initial state, final state, and allowed
interactions.

A further responsibility is to build amplitude models, if a reaction is valid.
These models are based on the found solutions and represent the transition
probability of the process.

The `expertsystem` consists of three main components:

  `expertsystem.particle`
    ― a stand-alone submodule with which one can investigate specific quantum
    properties of `.Particle` instances (see :doc:`/usage/particle`).

  `expertsystem.reaction`
    ― the core of the `expertsystem` that computes which transitions
    (represented by a `.StateTransitionGraph`) are allowed between a certain
    initial and final state. Internally, the system propagates the quantum
    numbers defined by `particle` through the `.StateTransitionGraph`, while
    satisfying the rules define by the :mod:`.conservation_rules` module. See
    :doc:`/usage/reaction`.

  `expertsystem.amplitude`
    ― a collection of tools to convert the `.StateTransitionGraph` solutions
    found by `reaction` into an `.HelicityModel`. This module is specifically
    designed to create amplitude model templates for :doc:`PWA fitter packages
    <pwa:software>`. See :doc:`/usage/amplitude`.

Finally, the `.io` module provides tools that can read and write the objects of
`particle`, `reaction`, and `amplitude`.
"""


__all__ = [
    # Main modules
    "amplitude",
    "particle",
    "reaction",
    "io",
    # Facade functions
    "generate_transitions",
    "check_reaction_violations",
    "load_default_particles",
]


from . import amplitude, io, particle, reaction
from .reaction.default_settings import ADDITIONAL_PARTICLES_DEFINITIONS_PATH

generate_transitions = reaction.generate
"""An alias to `.reaction.generate`."""

check_reaction_violations = reaction.check_reaction_violations
"""An alias to `.reaction.check_reaction_violations`."""


def load_default_particles() -> particle.ParticleCollection:
    """Load the default particle list that comes with the `expertsystem`.

    Runs `.load_pdg` and supplements its output definitions from the file
    :download:`particle/additional_definitions.yml
    </../src/expertsystem/particle/additional_definitions.yml>`.
    """
    particles = particle.load_pdg()
    additional_particles = io.load(ADDITIONAL_PARTICLES_DEFINITIONS_PATH)
    assert isinstance(additional_particles, particle.ParticleCollection)
    particles.update(additional_particles)
    return particles
