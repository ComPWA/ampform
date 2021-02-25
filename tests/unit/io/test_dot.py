import pydot
import pytest

from expertsystem import io
from expertsystem.reaction import Result
from expertsystem.reaction.topology import (
    Edge,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)


def test_convert_to_dot(jpsi_to_gamma_pi_pi_helicity_solutions: Result):
    result = jpsi_to_gamma_pi_pi_helicity_solutions
    for transition in result.transitions:
        dot_data = io.convert_to_dot(transition)
        assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.transitions)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.get_particle_graphs())
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(result.collapse_graphs())
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(create_n_body_topology(3, 4))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(create_isobar_topologies(2))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(create_isobar_topologies(3))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.convert_to_dot(create_isobar_topologies(4))
    assert pydot.graph_from_dot_data(dot_data) is not None


class TestWrite:
    @staticmethod
    def test_exception(output_dir):
        with pytest.raises(NotImplementedError):
            io.write(
                instance="nope, can't write a str",
                filename=output_dir + "dummy_file.gv",
            )

    @staticmethod
    def test_write_topology(output_dir):
        output_file = output_dir + "two_body_decay_topology.gv"
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
        output_dir,
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
    ):
        output_file = output_dir + "test_single_graph.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions.transitions[0],
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_graph_list(
        output_dir, jpsi_to_gamma_pi_pi_helicity_solutions: Result
    ):
        output_file = output_dir + "test_graph_list.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions.transitions,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_particle_graphs(
        output_dir,
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
    ):
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        particle_graphs = result.get_particle_graphs()
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=particle_graphs,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None
