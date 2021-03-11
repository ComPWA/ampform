"""Serialization module for the `expertsystem`.

The `.io` module provides tools to export or import objects from the
:mod:`.particle`, :mod:`.reaction` and :mod:`.amplitude` modules to and from
disk, so that they can be used by external packages, or just to store (cache)
the state of the system.
"""

import json
from collections import abc
from pathlib import Path

import yaml

from expertsystem.amplitude.model import (
    AmplitudeModel,
    FitParameter,
    FitParameters,
)
from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction.topology import StateTransitionGraph, Topology

from . import _dict, _dot, _pdg


def asdict(instance: object) -> dict:
    if isinstance(instance, FitParameter):
        return _dict.dump.from_fit_parameter(instance)
    if isinstance(instance, FitParameters):
        return _dict.dump.from_fit_parameters(instance)
    if isinstance(instance, Particle):
        return _dict.dump.from_particle(instance)
    if isinstance(instance, ParticleCollection):
        return _dict.dump.from_particle_collection(instance)
    if isinstance(instance, AmplitudeModel):
        return _dict.dump.from_amplitude_model(instance)
    raise NotImplementedError(
        f"No conversion for dict available for class {instance.__class__.__name__}"
    )


def fromdict(definition: dict) -> object:
    type_defined = _determine_type(definition)
    if type_defined == AmplitudeModel:
        return _dict.build.build_amplitude_model(definition)
    if type_defined == FitParameters:
        return _dict.build.build_fit_parameters(definition)
    if type_defined == ParticleCollection:
        return _dict.build.build_particle_collection(definition)
    raise NotImplementedError


def validate(instance: dict) -> None:
    type_defined = _determine_type(instance)
    if type_defined == AmplitudeModel:
        return _dict.validate.amplitude_model(instance)
    if type_defined == ParticleCollection:
        return _dict.validate.particle_collection(instance)
    raise NotImplementedError


def load(filename: str) -> object:
    with open(filename) as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            definition = json.load(stream)
            return fromdict(definition)
        if file_extension in ["yaml", "yml"]:
            definition = yaml.load(stream, Loader=yaml.SafeLoader)
            return fromdict(definition)
    raise NotImplementedError(
        f'No loader defined for file type "{file_extension}"'
    )


def load_pdg() -> ParticleCollection:
    """Create a `.ParticleCollection` with all entries from the PDG.

    PDG info is imported from the `scikit-hep/particle
    <https://github.com/scikit-hep/particle>`_ package.
    """
    return _pdg.load_pdg()


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)

    def write_line_break(self, data=None):  # type: ignore
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def write(instance: object, filename: str) -> None:
    with open(filename, "w") as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            json.dump(asdict(instance), stream, indent=2)
            return
        if file_extension in ["yaml", "yml"]:
            yaml.dump(
                asdict(instance),
                stream,
                sort_keys=False,
                Dumper=_IncreasedIndent,
                default_flow_style=False,
            )
            return
        if file_extension == "gv":
            output_str = convert_to_dot(instance)
            with open(filename, "w") as stream:
                stream.write(output_str)
            return
    raise NotImplementedError(
        f'No writer defined for file type "{file_extension}"'
    )


def convert_to_dot(
    instance: object,
    render_edge_id: bool = True,
    render_node: bool = True,
) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph` or a `list` of `.StateTransitionGraph` instances.

    .. seealso:: :doc:`/usage/visualization`
    """
    if isinstance(instance, (StateTransitionGraph, Topology)):
        return _dot.graph_to_dot(
            instance,
            render_edge_id=render_edge_id,
            render_node=render_node,
        )
    if isinstance(instance, abc.Sequence):
        return _dot.graph_list_to_dot(
            instance,
            render_edge_id=render_edge_id,
            render_node=render_node,
        )
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise Exception(f"No file extension in file {filename}")
    extension = extension[1:]
    return extension


def _determine_type(definition: dict) -> type:
    keys = set(definition.keys())
    if keys == {
        "dynamics",
        "intensity",
        "kinematics",
        "parameters",
        "particles",
    }:
        return AmplitudeModel
    if keys == {"parameters"}:
        return FitParameters
    if keys == {"particles"}:
        return ParticleCollection
    raise NotImplementedError(f"Could not determine type from keys {keys}")
