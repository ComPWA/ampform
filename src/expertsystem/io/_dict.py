"""Serialization from and to a `dict`."""

import json
from collections import abc
from os.path import dirname, realpath
from typing import Any, Dict

import attr
import jsonschema

from expertsystem.particle import Parity, Particle, ParticleCollection, Spin
from expertsystem.reaction import (
    InteractionProperties,
    ParticleWithSpin,
    Result,
    StateTransitionGraph,
    Topology,
)
from expertsystem.reaction.topology import Edge


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {"particles": [from_particle(p) for p in particles]}


def from_particle(particle: Particle) -> dict:
    return attr.asdict(
        particle,
        recurse=True,
        value_serializer=__value_serializer,
        filter=lambda attr, value: attr.default != value,
    )


def from_result(result: Result) -> dict:
    output: Dict[str, Any] = {
        "transitions": [from_stg(graph) for graph in result.transitions],
    }
    if result.formalism_type is not None:
        output["formalism_type"] = result.formalism_type
    return output


def from_stg(graph: StateTransitionGraph[ParticleWithSpin]) -> dict:
    topology = graph.topology
    edge_props_def = dict()
    for i in topology.edges:
        particle, spin_projection = graph.get_edge_props(i)
        if isinstance(spin_projection, float) and spin_projection.is_integer():
            spin_projection = int(spin_projection)
        edge_props_def[i] = {
            "particle": from_particle(particle),
            "spin_projection": spin_projection,
        }
    node_props_def = dict()
    for i in topology.nodes:
        node_prop = graph.get_node_props(i)
        node_props_def[i] = attr.asdict(
            node_prop, filter=lambda a, v: a.init and a.default != v
        )
    return {
        "topology": from_topology(topology),
        "edge_props": edge_props_def,
        "node_props": node_props_def,
    }


def from_topology(topology: Topology) -> dict:
    return attr.asdict(
        topology,
        recurse=True,
        value_serializer=__value_serializer,
        filter=lambda a, v: a.init and a.default != v,
    )


def __value_serializer(  # pylint: disable=unused-argument
    inst: type, field: attr.Attribute, value: Any
) -> Any:
    if isinstance(value, abc.Mapping):
        if all(map(lambda p: isinstance(p, Particle), value.values())):
            return {k: v.name for k, v in value.items()}
        return dict(value)
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


def build_result(definition: dict) -> Result:
    formalism_type = definition.get("formalism_type")
    transitions = [
        build_stg(graph_def) for graph_def in definition["transitions"]
    ]
    return Result(
        transitions=transitions,
        formalism_type=formalism_type,
    )


def build_stg(definition: dict) -> StateTransitionGraph[ParticleWithSpin]:
    topology = build_topology(definition["topology"])
    edge_props_def: Dict[int, dict] = definition["edge_props"]
    edge_props: Dict[int, ParticleWithSpin] = dict()
    for i, edge_def in edge_props_def.items():
        particle = build_particle(edge_def["particle"])
        spin_projection = float(edge_def["spin_projection"])
        if spin_projection.is_integer():
            spin_projection = int(spin_projection)
        edge_props[int(i)] = (particle, spin_projection)
    node_props_def: Dict[int, dict] = definition["node_props"]
    node_props = {
        int(i): InteractionProperties(**node_def)
        for i, node_def in node_props_def.items()
    }
    return StateTransitionGraph(
        topology=topology,
        edge_props=edge_props,
        node_props=node_props,
    )


def build_topology(definition: dict) -> Topology:
    nodes = definition["nodes"]
    edges_def: Dict[int, dict] = definition["edges"]
    edges = {int(i): Edge(**edge_def) for i, edge_def in edges_def.items()}
    return Topology(
        edges=edges,
        nodes=nodes,
    )


def validate_particle_collection(instance: dict) -> None:
    jsonschema.validate(instance=instance, schema=__SCHEMA_PARTICLES)


__EXPERTSYSTEM_PATH = dirname(dirname(realpath(__file__)))
with open(f"{__EXPERTSYSTEM_PATH}/particle/validation.json") as __STREAM:
    __SCHEMA_PARTICLES = json.load(__STREAM)
