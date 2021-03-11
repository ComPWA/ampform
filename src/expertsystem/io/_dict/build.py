"""Read recipe objects from a `dict` structure."""

from expertsystem.particle import Parity, Particle, ParticleCollection, Spin

from . import validate


def build_particle_collection(
    definition: dict, do_validate: bool = True
) -> ParticleCollection:
    if do_validate:
        validate.particle_collection(definition)
    return ParticleCollection(
        __build_particle(p) for p in definition["particles"]
    )


def __build_particle(definition: dict) -> Particle:
    isospin_def = definition.get("isospin", None)
    if isospin_def is not None:
        definition["isospin"] = Spin(**isospin_def)
    for parity in ["parity", "c_parity", "g_parity"]:
        parity_def = definition.get(parity, None)
        if parity_def is not None:
            definition[parity] = Parity(**parity_def)
    return Particle(**definition)
