from itertools import (product)

from expertsystem.ui.system_control import (filter_solutions,
                                            compare_graph_element_properties)
from expertsystem.topology.graph import StateTransitionGraph


def make_test_graph(angular_momentum_magnitude, coupled_spin_magnitude):
    graph = StateTransitionGraph()
    graph.set_graph_element_properties_comparator(
        compare_graph_element_properties)
    graph.node_props[0] = {'QuantumNumber': [
        {'@Value': str(coupled_spin_magnitude), '@Type': 'S',
         '@Projection': '0.0', '@Class': 'Spin'},
        {'@Value': str(angular_momentum_magnitude), '@Type': 'L',
         '@Projection': '0.0', '@Class': 'Spin'},
    ]}
    return graph


def make_test_graph_scrambled(angular_momentum_magnitude,
                              coupled_spin_magnitude):
    graph = StateTransitionGraph()
    graph.set_graph_element_properties_comparator(
        compare_graph_element_properties)
    graph.node_props[0] = {'QuantumNumber': [
        {'@Class': 'Spin', '@Value': str(angular_momentum_magnitude),
         '@Type': 'L', '@Projection': '0.0'},
        {'@Projection': '0.0', '@Class': 'Spin',
         '@Value': str(coupled_spin_magnitude), '@Type': 'S'}
    ]}
    return graph


class TestSolutionFilter(object):
    def test_interaction_qns_different(self):
        graphs = {'test': []}
        graphs['test'].append(([make_test_graph(1, 0)], []))
        graphs['test'].append(([make_test_graph(1, 1)], []))

        results = filter_solutions(graphs, [], [])
        num_solutions = [len(x[0]) for x in results['test']]
        assert sum(num_solutions) == 2

        graphs['test'].append(([make_test_graph_scrambled(1, 0)], []))
        results = filter_solutions(graphs, [], [])
        num_solutions = [len(x[0]) for x in results['test']]
        assert sum(num_solutions) == 2

    def test_interaction_qns_same(self):
        graphs = {'test': []}
        graphs['test'].append(([make_test_graph(1, 0)], []))
        graphs['test'].append(([make_test_graph(1, 0)], []))

        results = filter_solutions(graphs, [], [])
        num_solutions = [len(x[0]) for x in results['test']]
        assert sum(num_solutions) == 1

        graphs['test'].append(([make_test_graph_scrambled(1, 0)], []))
        results = filter_solutions(graphs, [], [])
        num_solutions = [len(x[0]) for x in results['test']]
        assert sum(num_solutions) == 1
