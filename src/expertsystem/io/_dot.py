"""Generate dot sources.

See :doc:`/usage/visualize` for more info.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Union

from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction import (
    InteractionProperties,
    ParticleWithSpin,
    StateTransitionGraph,
    Topology,
)

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


def embed_dot(func: Callable) -> Callable:
    """Add a DOT head and tail to some DOT content."""

    def wrapper(*args, **kwargs):  # type: ignore
        dot = _DOT_HEAD
        dot += func(*args, **kwargs)
        dot += _DOT_TAIL
        return dot

    return wrapper


@embed_dot
def graph_list_to_dot(
    graphs: Sequence[StateTransitionGraph],
    render_edge_id: bool = True,
    render_node: bool = True,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
) -> str:
    if strip_spin and collapse_graphs:
        raise ValueError("Cannot both strip spin and collapse graphs")
    if collapse_graphs:
        graphs = _collapse_graphs(graphs)
    elif strip_spin:
        graphs = _get_particle_graphs(graphs)
    dot = ""
    for i, graph in enumerate(reversed(graphs)):
        dot += __graph_to_dot_content(
            graph,
            prefix=f"g{i}_",
            render_edge_id=render_edge_id,
            render_node=render_node,
        )
    return dot


@embed_dot
def graph_to_dot(
    graph: StateTransitionGraph,
    render_edge_id: bool = True,
    render_node: bool = True,
) -> str:
    return __graph_to_dot_content(
        graph,
        render_edge_id=render_edge_id,
        render_node=render_node,
    )


def __graph_to_dot_content(  # pylint: disable=too-many-locals,too-many-branches
    graph: Union[StateTransitionGraph, Topology],
    prefix: str = "",
    render_edge_id: bool = True,
    render_node: bool = True,
) -> str:
    dot = ""
    if isinstance(graph, StateTransitionGraph):
        topology = graph.topology
    elif isinstance(graph, Topology):
        topology = graph
    else:
        raise NotImplementedError
    top = topology.incoming_edge_ids
    outs = topology.outgoing_edge_ids
    for edge_id in top | outs:
        dot += _DOT_DEFAULT_NODE.format(
            prefix + __node_name(edge_id),
            __get_edge_label(graph, edge_id, render_edge_id),
        )
    dot += __rank_string(top, prefix)
    dot += __rank_string(outs, prefix)
    for i, edge in topology.edges.items():
        j, k = edge.ending_node_id, edge.originating_node_id
        if j is None or k is None:
            dot += _DOT_DEFAULT_EDGE.format(
                prefix + __node_name(i, k), prefix + __node_name(i, j)
            )
        else:
            dot += _DOT_LABEL_EDGE.format(
                prefix + __node_name(i, k),
                prefix + __node_name(i, j),
                __get_edge_label(graph, i, render_edge_id),
            )
    if isinstance(graph, StateTransitionGraph):
        for node_id in topology.nodes:
            node_prop = graph.get_node_props(node_id)
            node_label = ""
            if render_node:
                node_label = __node_label(node_prop)
            dot += _DOT_DEFAULT_NODE.format(
                f"{prefix}node{node_id}", node_label
            )
    if isinstance(graph, Topology):
        if len(topology.nodes) > 1:
            for node_id in topology.nodes:
                node_label = ""
                if render_node:
                    node_label = f"({node_id})"
                dot += _DOT_DEFAULT_NODE.format(
                    f"{prefix}node{node_id}", node_label
                )
    return dot


def __node_name(edge_id: int, node_id: Optional[int] = None) -> str:
    if node_id is None:
        return f"edge{edge_id}"
    return f"node{node_id}"


def __rank_string(node_edge_ids: Iterable[int], prefix: str = "") -> str:
    name_list = [f'"{prefix}{__node_name(i)}"' for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return _DOT_RANK_SAME.format(name_string)


def __get_edge_label(
    graph: Union[StateTransitionGraph, Topology],
    edge_id: int,
    render_edge_id: bool = True,
) -> str:
    if isinstance(graph, StateTransitionGraph):
        edge_prop = graph.get_edge_props(edge_id)
        if not edge_prop:
            return str(edge_id)
        edge_label = __edge_label(edge_prop)
        if not render_edge_id:
            return edge_label
        if "\n" in edge_label:
            return f"{edge_id}:\n{edge_label}"
        return f"{edge_id}: {edge_label}"
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


def _get_particle_graphs(
    graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
) -> List[StateTransitionGraph[Particle]]:
    """Strip `list` of `.StateTransitionGraph` s of the spin projections.

    Extract a `list` of `.StateTransitionGraph` instances with only
    particles on the edges.

    .. seealso:: :doc:`/usage/visualize`
    """
    inventory: List[StateTransitionGraph[Particle]] = list()
    for transition in graphs:
        if any(
            transition.compare(
                other, edge_comparator=lambda e1, e2: e1[0] == e2
            )
            for other in inventory
        ):
            continue
        new_edge_props = dict()
        for edge_id in transition.topology.edges:
            edge_props = transition.get_edge_props(edge_id)
            if edge_props:
                new_edge_props[edge_id] = edge_props[0]
        inventory.append(
            StateTransitionGraph[Particle](
                topology=transition.topology,
                node_props={
                    i: node_props
                    for i, node_props in zip(
                        transition.topology.nodes,
                        map(
                            transition.get_node_props,
                            transition.topology.nodes,
                        ),
                    )
                    if node_props
                },
                edge_props=new_edge_props,
            )
        )
    inventory = sorted(
        inventory,
        key=lambda g: [
            g.get_edge_props(i).mass for i in g.topology.intermediate_edge_ids
        ],
    )
    return inventory


def _collapse_graphs(
    graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
) -> List[StateTransitionGraph[ParticleCollection]]:
    def merge_into(
        graph: StateTransitionGraph[Particle],
        merged_graph: StateTransitionGraph[ParticleCollection],
    ) -> None:
        if (
            graph.topology.intermediate_edge_ids
            != merged_graph.topology.intermediate_edge_ids
        ):
            raise ValueError(
                "Cannot merge graphs that don't have the same edge IDs"
            )
        for i in graph.topology.edges:
            particle = graph.get_edge_props(i)
            other_particles = merged_graph.get_edge_props(i)
            if particle not in other_particles:
                other_particles += particle

    def is_same_shape(
        graph: StateTransitionGraph[Particle],
        merged_graph: StateTransitionGraph[ParticleCollection],
    ) -> bool:
        if graph.topology.edges != merged_graph.topology.edges:
            return False
        for edge_id in (
            graph.topology.incoming_edge_ids | graph.topology.outgoing_edge_ids
        ):
            edge_prop = merged_graph.get_edge_props(edge_id)
            if len(edge_prop) != 1:
                return False
            other_particle = next(iter(edge_prop))
            if other_particle != graph.get_edge_props(edge_id):
                return False
        return True

    particle_graphs = _get_particle_graphs(graphs)
    inventory: List[StateTransitionGraph[ParticleCollection]] = list()
    for graph in particle_graphs:
        append_to_inventory = True
        for merged_graph in inventory:
            if is_same_shape(graph, merged_graph):
                merge_into(graph, merged_graph)
                append_to_inventory = False
                break
        if append_to_inventory:
            new_edge_props = {
                edge_id: ParticleCollection({graph.get_edge_props(edge_id)})
                for edge_id in graph.topology.edges
            }
            inventory.append(
                StateTransitionGraph[ParticleCollection](
                    topology=graph.topology,
                    node_props={
                        i: graph.get_node_props(i)
                        for i in graph.topology.nodes
                    },
                    edge_props=new_edge_props,
                )
            )
    return inventory
