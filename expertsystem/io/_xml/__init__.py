"""Serialization from and to an XML recipe file."""

__all__ = [
    "load_particle_collection",
    "write",
]

import json

import xmltodict

from expertsystem.data import ParticleCollection

from ._build import _build_particle_collection
from ._dump import _from_particle_collection


def load_particle_collection(filename: str) -> ParticleCollection:
    with open(filename, "rb") as stream:
        definition = xmltodict.parse(stream)
    definition = definition.get("root", definition)
    json.loads(json.dumps(definition))  # remove OrderedDict
    return _build_particle_collection(definition)


def write(instance: object, filename: str) -> None:
    if isinstance(instance, ParticleCollection):
        output_dict = _from_particle_collection(instance)
        entries = list(output_dict.values())
        output_dict = {"ParticleList": {"Particle": entries}}
    else:
        raise NotImplementedError(
            f"No XML writer for class {instance.__class__.__name__}"
        )
    with open(filename, "w") as stream:
        xmlstring = xmltodict.unparse(
            {"root": output_dict}, pretty=True, indent="  "
        )
        stream.write(xmlstring)
