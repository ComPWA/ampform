"""Generate dot sources.

See :doc:`/usage/visualization` for more info.
"""

from typing import (
    Any,
    Callable,
    List,
    Optional,
)

from expertsystem.topology import StateTransitionGraph, Topology


def convert_to_dot(instance: object) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph` or a `list` of `.StateTransitionGraph` instances.
    """
    if isinstance(instance, (StateTransitionGraph, Topology)):
        return __graph_to_dot(instance)
    if isinstance(instance, list):
        return __graph_list_to_dot(instance)
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


def write(instance: object, filename: str) -> None:
    output_str = convert_to_dot(instance)
    with open(filename, "w") as stream:
        stream.write(output_str)


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
def __graph_list_to_dot(graphs: List[StateTransitionGraph]) -> str:
    dot_source = ""
    for i, graph in enumerate(graphs):
        dot_source += __graph_to_dot_content(graph, prefix=f"g{i}_")
    return dot_source


@embed_dot
def __graph_to_dot(graph: StateTransitionGraph) -> str:
    return __graph_to_dot_content(graph)


def __graph_to_dot_content(
    graph: StateTransitionGraph, prefix: str = ""
) -> str:
    dot_source = ""
    top = graph.get_initial_state_edges()
    outs = graph.get_final_state_edges()
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
    if isinstance(graph, StateTransitionGraph) and edge_id in graph.edge_props:
        properties = graph.edge_props[edge_id]
        label = properties.get("Name", edge_id)
        quantum_numbers = properties.get("QuantumNumber", None)
        if quantum_numbers is not None:
            spin_projection_candidates = [
                number.get("Projection", None)
                for number in quantum_numbers
                if number["Type"] == "Spin"
            ]
            if spin_projection_candidates:
                projection = float(spin_projection_candidates[0])
                if projection.is_integer():
                    projection = int(projection)
                label += f"[{projection}]"
    else:
        label = str(edge_id)
    return label
