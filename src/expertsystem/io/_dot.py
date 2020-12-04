"""Generate dot sources.

See :doc:`/usage/visualization` for more info.
"""

from typing import Any, Callable, List, Optional

from expertsystem.particle import Particle, ParticleCollection
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
    graph: StateTransitionGraph, prefix: str = ""
) -> str:
    dot_source = ""
    top = graph.get_initial_state_edge_ids()
    outs = graph.get_final_state_edge_ids()
    for edge_id in top + outs:
        dot_source += _DOT_DEFAULT_NODE.format(
            prefix + __node_name(edge_id), __edge_label(graph, edge_id)
        )
    dot_source += __rank_string(top, prefix)
    dot_source += __rank_string(outs, prefix)
    for i, edge in graph.edges.items():
        j, k = edge.ending_node_id, edge.originating_node_id
        if j is None or k is None:
            dot_source += _DOT_DEFAULT_EDGE.format(
                prefix + __node_name(i, k), prefix + __node_name(i, j)
            )
        else:
            dot_source += _DOT_LABEL_EDGE.format(
                prefix + __node_name(i, k),
                prefix + __node_name(i, j),
                __edge_label(graph, i),
            )
    return dot_source


def __node_name(edge_id: int, node_id: Optional[int] = None) -> str:
    if node_id is None:
        return f"edge{edge_id}"
    return f"node{node_id}"


def __rank_string(node_edge_ids: List[int], prefix: str = "") -> str:
    name_list = [f'"{prefix}{__node_name(i)}"' for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return _DOT_RANK_SAME.format(name_string)


def __edge_label(graph: StateTransitionGraph, edge_id: int) -> str:
    if isinstance(graph, StateTransitionGraph):
        edge_prop = graph.get_edge_props(edge_id)
        if not edge_prop:
            return str(edge_id)
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
    if isinstance(graph, Topology):
        return str(edge_id)
    raise NotImplementedError
