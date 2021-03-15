"""Serialization module for the `expertsystem`.

The `.io` module provides tools to export or import objects from the
:mod:`.particle`, :mod:`.reaction` and :mod:`.amplitude` modules to and from
disk, so that they can be used by external packages, or just to store (cache)
the state of the system.
"""

import json
from collections import abc
from pathlib import Path

import attr
import yaml

from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction import Result, StateTransitionGraph, Topology

from . import _dict, _dot


def asdict(instance: object) -> dict:
    if isinstance(instance, Particle):
        return _dict.from_particle(instance)
    if isinstance(instance, ParticleCollection):
        return _dict.from_particle_collection(instance)
    if isinstance(instance, Result):
        return _dict.from_result(instance)
    if isinstance(instance, StateTransitionGraph):
        return _dict.from_stg(instance)
    if isinstance(instance, Topology):
        return _dict.from_topology(instance)
    raise NotImplementedError(
        f"No conversion for dict available for class {instance.__class__.__name__}"
    )


def fromdict(definition: dict) -> object:
    keys = set(definition.keys())
    if __REQUIRED_PARTICLE_FIELDS <= keys:
        return _dict.build_particle(definition)
    if keys == {"particles"}:
        return _dict.build_particle_collection(definition)
    if keys == {"transitions", "formalism_type"}:
        return _dict.build_result(definition)
    if keys == {"topology", "edge_props", "node_props"}:
        return _dict.build_stg(definition)
    if keys == __REQUIRED_TOPOLOGY_FIELDS:
        return _dict.build_topology(definition)
    raise NotImplementedError(f"Could not determine type from keys {keys}")


__REQUIRED_PARTICLE_FIELDS = {
    field.name
    for field in attr.fields(Particle)
    if field.default == attr.NOTHING
}
__REQUIRED_TOPOLOGY_FIELDS = {
    field.name for field in attr.fields(Topology) if field.init
}


def asdot(
    instance: object,
    render_edge_id: bool = True,
    render_node: bool = True,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph` or a `list` of `.StateTransitionGraph` instances.

    .. seealso:: :doc:`/usage/visualize`
    """
    if isinstance(instance, (StateTransitionGraph, Topology)):
        return _dot.graph_to_dot(
            instance,
            render_edge_id=render_edge_id,
            render_node=render_node,
        )
    if isinstance(instance, (Result, abc.Sequence)):
        if isinstance(instance, Result):
            instance = instance.transitions
        return _dot.graph_list_to_dot(
            instance,
            render_edge_id=render_edge_id,
            render_node=render_node,
            strip_spin=strip_spin,
            collapse_graphs=collapse_graphs,
        )
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


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
            if isinstance(instance, str):  # direct output of asdot
                output_str = instance
            else:
                output_str = asdot(instance)
            with open(filename, "w") as stream:
                stream.write(output_str)
            return
    raise NotImplementedError(
        f'No writer defined for file type "{file_extension}"'
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise Exception(f"No file extension in file {filename}")
    extension = extension[1:]
    return extension
