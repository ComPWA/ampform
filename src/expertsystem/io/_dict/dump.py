"""Dump recipe objects to `dict` instances for a YAML file."""
from collections import abc
from typing import Any

import attr

from expertsystem.particle import Parity, Particle, ParticleCollection, Spin


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {"particles": [from_particle(p) for p in particles]}


def from_particle(particle: Particle) -> dict:
    return attr.asdict(
        particle,
        recurse=True,
        value_serializer=__value_serializer,
        filter=lambda attr, value: attr.default != value,
    )


def __value_serializer(  # pylint: disable=unused-argument
    inst: type, field: attr.Attribute, value: Any
) -> Any:
    if isinstance(value, abc.Mapping):
        if all(map(lambda p: isinstance(p, Particle), value.values())):
            return {k: v.name for k, v in value.items()}
    if isinstance(value, Particle):
        return value.name
    if isinstance(value, Parity):
        return {"value": value.value}
    if isinstance(value, Spin):
        return {
            "magnitude": value.magnitude,
            "projection": value.projection,
        }
    return value
