"""Generate dot sources.

See :doc:`/usage/visualization` for more info.
"""

from typing import Any, Callable, Iterable, List, Optional, Union

from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction.quantum_numbers import (
    InteractionProperties,
    ParticleWithSpin,
)
from expertsystem.reaction.topology import StateTransitionGraph, Topology

_DOT_HEAD = """digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
"""
_DOT_TAIL = "}\n"
_DOT_RANK_SAME = "    {{ rank=same {} }};\n"
_DOT_DEFAULT_NODE = '    "{}" [shape=none, label="{}"];\n'
_DOT_DEFAULT_EDGE = '    "{}" -> "{}";\n'
_DOT_LABEL_EDGE = '    "{}" -> "{}" [label="{}"];\n'


def embed_dot(func: Callable[[Any], str]) -> Callable[[Any], str]:
    """Add a DOT head and tail to some DOT content."""

    def wrapper(*args, **kwargs):  # type: ignore
        dot_source = _DOT_HEAD
        dot_source += func(*args, **kwargs)
        dot_source += _DOT_TAIL
        return dot_source

    return wrapper


@embed_dot
def graph_list_to_dot(graphs: List[StateTransitionGraph]) -> str:
    dot_source = ""
    for i, graph in enumerate(reversed(graphs)):
        dot_source += __graph_to_dot_content(graph, prefix=f"g{i}_")
    return dot_source


@embed_dot
def graph_to_dot(graph: StateTransitionGraph) -> str:
    return __graph_to_dot_content(graph)


def __graph_to_dot_content(
    graph: Union[StateTransitionGraph, Topology], prefix: str = ""
) -> str:
    dot_source = ""
    if isinstance(graph, StateTransitionGraph):
        topology = graph.topology
    elif isinstance(graph, Topology):
        topology = graph
    else:
        raise NotImplementedError
    top = topology.incoming_edge_ids
    outs = topology.outgoing_edge_ids
    for edge_id in top | outs:
        dot_source += _DOT_DEFAULT_NODE.format(
            prefix + __node_name(edge_id), __get_edge_label(graph, edge_id)
        )
    dot_source += __rank_string(top, prefix)
    dot_source += __rank_string(outs, prefix)
    for i, edge in topology.edges.items():
        j, k = edge.ending_node_id, edge.originating_node_id
        if j is None or k is None:
            dot_source += _DOT_DEFAULT_EDGE.format(
                prefix + __node_name(i, k), prefix + __node_name(i, j)
            )
        else:
            dot_source += _DOT_LABEL_EDGE.format(
                prefix + __node_name(i, k),
                prefix + __node_name(i, j),
                __get_edge_label(graph, i),
            )
    if isinstance(graph, StateTransitionGraph):
        for node_id in topology.nodes:
            node_prop = graph.get_node_props(node_id)
            node_label = __node_label(node_prop)
            dot_source += _DOT_DEFAULT_NODE.format(
                f"{prefix}node{node_id}", node_label
            )
    return dot_source


def __node_name(edge_id: int, node_id: Optional[int] = None) -> str:
    if node_id is None:
        return f"edge{edge_id}"
    return f"node{node_id}"


def __rank_string(node_edge_ids: Iterable[int], prefix: str = "") -> str:
    name_list = [f'"{prefix}{__node_name(i)}"' for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return _DOT_RANK_SAME.format(name_string)


def __get_edge_label(
    graph: Union[StateTransitionGraph, Topology], edge_id: int
) -> str:
    if isinstance(graph, StateTransitionGraph):
        edge_prop = graph.get_edge_props(edge_id)
        if not edge_prop:
            return str(edge_id)
        return __edge_label(edge_prop)
    if isinstance(graph, Topology):
        return str(edge_id)
    raise NotImplementedError


def __edge_label(
    edge_prop: Union[ParticleCollection, Particle, ParticleWithSpin]
) -> str:
    if isinstance(edge_prop, Particle):
        return edge_prop.name
    if isinstance(edge_prop, tuple):
        particle, projection = edge_prop
        spin_projection = float(projection)
        if spin_projection.is_integer():
            spin_projection = int(spin_projection)
        label = particle.name
        if spin_projection is not None:
            label += f"[{projection}]"
        return label
    if isinstance(edge_prop, ParticleCollection):
        return "\n".join(sorted(edge_prop.names))
    raise NotImplementedError


def __node_label(node_prop: Union[InteractionProperties]) -> str:
    if isinstance(node_prop, InteractionProperties):
        output = ""
        if node_prop.l_magnitude is not None:
            l_label = str(node_prop.l_magnitude)
            if node_prop.l_projection is not None:
                l_label = f"{(node_prop.l_magnitude, node_prop.l_projection)}"
            output += f"l={l_label}\n"
        if node_prop.s_magnitude is not None:
            s_label = str(node_prop.s_magnitude)
            if node_prop.s_projection is not None:
                s_label = f"{(node_prop.s_magnitude, node_prop.s_projection)}"
            output += f"s={s_label}\n"
        if node_prop.parity_prefactor is not None:
            output += f"P={node_prop.parity_prefactor}"
        return output
    raise NotImplementedError
