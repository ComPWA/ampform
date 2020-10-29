# pylint: disable=protected-access

import pydot
import pytest

from expertsystem import io
from expertsystem.reaction.solving import Result
from expertsystem.reaction.topology import Edge, Topology


def test_dot_syntax(jpsi_to_gamma_pi_pi_helicity_solutions: Result):
    result = jpsi_to_gamma_pi_pi_helicity_solutions
    for graph in result.solutions:
        dot_data = io.convert_to_dot(graph)
        assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.solutions)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.get_particle_graphs())
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.collapse_graphs())
    assert pydot.graph_from_dot_data(dot_data) is not None


class TestWrite:
    @staticmethod
    def test_exception():
        with pytest.raises(NotImplementedError):
            io.write(
                instance="nope, can't write a str",
                filename="dummy_file.gv",
            )

    @staticmethod
    def test_write_topology():
        output_file = "two_body_decay_topology.gv"
        topology = Topology(
            nodes={0},
            edges={0: Edge(0, None), 1: Edge(None, 0), 2: Edge(None, 0)},
        )
        io.write(
            instance=topology,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_single_graph(
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
    ):
        output_file = "test_single_graph.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions.solutions[0],
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_graph_list(jpsi_to_gamma_pi_pi_helicity_solutions: Result):
        output_file = "test_graph_list.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions.solutions,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_particle_graphs(
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
    ):
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        particle_graphs = result.get_particle_graphs()
        output_file = "test_particle_graphs.gv"
        io.write(
            instance=particle_graphs,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None
