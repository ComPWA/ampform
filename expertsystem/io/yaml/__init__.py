"""Serialization from and to a YAML recipe file."""

__all__ = [
    "load_particle_collection",
    "write",
]

import yaml

from expertsystem.data import (
    Particle,
    ParticleCollection,
)

from . import _dump
from ._build import (
    build_particle,
    build_particle_collection,
)


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super(_IncreasedIndent, self).increase_indent(flow, False)

    def write_line_break(self, data=None):  # type: ignore
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def load_particle_collection(filename: str) -> ParticleCollection:
    with open(filename) as yaml_file:
        definition = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return build_particle_collection(definition)


def write(instance: object, filename: str) -> None:
    if isinstance(instance, ParticleCollection):
        output_dict = _dump.from_particle_collection(instance)
    else:
        raise NotImplementedError(
            f"No YAML writer for class {instance.__class__.__name__}"
        )
    with open(filename, "w") as yaml_file:
        yaml.dump(
            output_dict,
            yaml_file,
            sort_keys=False,
            Dumper=_IncreasedIndent,
            default_flow_style=False,
        )


def object_to_dict(instance: object) -> dict:
    if isinstance(instance, ParticleCollection):
        return _dump.from_particle_collection(instance)
    if isinstance(instance, Particle):
        return _dump.from_particle(instance)
    raise NotImplementedError


def dict_to_particle_collection(definition: dict) -> ParticleCollection:
    return build_particle_collection(definition)


def dict_to_particle(definition: dict, name: str) -> Particle:
    return build_particle(name, definition)
