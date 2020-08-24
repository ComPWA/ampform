"""Generate dot sources.

See :doc:`/usage/visualization` for more info.
"""

from typing import (
    List,
    Optional,
)

from expertsystem.topology import StateTransitionGraph


def convert_to_dot(instance: object) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph`.
    """
    if isinstance(instance, StateTransitionGraph):
        return __graph_to_dot(instance)
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


def write(instance: object, filename: str) -> None:
    output_str = convert_to_dot(instance)
    with open(filename, "w") as stream:
        stream.write(output_str)


_DOT_HEAD = """
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none, labelfloat=true];
    """
_DOT_TAIL = "}\n"
_DOT_RANK_SAME = "    {{ rank=same {} }};\n"
_DOT_DEFAULT_NODE = '    "{}" [shape=none, label="{}"];\n'
_DOT_DEFAULT_EDGE = '    "{}" -> "{}";\n'
_DOT_LABEL_EDGE = '    "{}" -> "{}" [label="{}"];\n'


def __graph_to_dot(graph: StateTransitionGraph) -> str:
    def node_name(edge_id: int, node_id: Optional[int] = None) -> str:
        if node_id is None:
            return f"edge{edge_id}"
        return f"node{node_id}"

    def format_particle(node_edge_ids: List[int]) -> str:
        name_list = [f'"{node_name(i)}"' for i in node_edge_ids]
        return ",".join(name_list)

    dot_source = _DOT_HEAD

    top = graph.get_initial_state_edges()
    outs = graph.get_final_state_edges()
    for i in top:
        dot_source += _DOT_DEFAULT_NODE.format(
            node_name(i), graph.edge_props[i]["Name"]
        )
    for i in outs:
        dot_source += _DOT_DEFAULT_NODE.format(
            node_name(i), graph.edge_props[i]["Name"]
        )

    dot_source += _DOT_RANK_SAME.format(format_particle(top))
    dot_source += _DOT_RANK_SAME.format(format_particle(outs))

    for i, edge in graph.edges.items():
        j, k = edge.ending_node_id, edge.originating_node_id
        if j is None or k is None:
            dot_source += _DOT_DEFAULT_EDGE.format(
                node_name(i, k), node_name(i, j)
            )
        else:
            dot_source += _DOT_LABEL_EDGE.format(
                node_name(i, k), node_name(i, j), graph.edge_props[i]["Name"],
            )

    dot_source += _DOT_TAIL
    return dot_source
