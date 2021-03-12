"""Serialization from and to a `dict`."""

import json
from collections import abc
from os.path import dirname, realpath
from typing import Any

import attr
import jsonschema

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


def build_particle_collection(
    definition: dict, do_validate: bool = True
) -> ParticleCollection:
    if do_validate:
        validate_particle_collection(definition)
    return ParticleCollection(
        build_particle(p) for p in definition["particles"]
    )


def build_particle(definition: dict) -> Particle:
    isospin_def = definition.get("isospin", None)
    if isospin_def is not None:
        definition["isospin"] = Spin(**isospin_def)
    for parity in ["parity", "c_parity", "g_parity"]:
        parity_def = definition.get(parity, None)
        if parity_def is not None:
            definition[parity] = Parity(**parity_def)
    return Particle(**definition)


def validate_particle_collection(instance: dict) -> None:
    jsonschema.validate(instance=instance, schema=__SCHEMA_PARTICLES)


__EXPERTSYSTEM_PATH = dirname(dirname(realpath(__file__)))
with open(f"{__EXPERTSYSTEM_PATH}/particle/validation.json") as __STREAM:
    __SCHEMA_PARTICLES = json.load(__STREAM)
