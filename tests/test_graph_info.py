from qrules.topology import create_isobar_topologies

from ampform._graph_info import determine_attached_final_state


def test_determine_attached_final_state():
    topologies = create_isobar_topologies(4)
    # outer edges
    for topology in topologies:
        for i in topology.outgoing_edge_ids:
            assert determine_attached_final_state(topology, edge_id=i) == [i]
        for i in topology.incoming_edge_ids:
            assert determine_attached_final_state(topology, edge_id=i) == list(
                topology.outgoing_edge_ids
            )
    # intermediate edges
    topology = topologies[0]
    assert determine_attached_final_state(topology, edge_id=4) == [0, 1]
    assert determine_attached_final_state(topology, edge_id=5) == [2, 3]
    topology = topologies[1]
    assert determine_attached_final_state(topology, edge_id=4) == [1, 2, 3]
    assert determine_attached_final_state(topology, edge_id=5) == [2, 3]
