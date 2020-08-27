import pydot

import pytest

from expertsystem import io


def test_dot_syntax(jpsi_to_gamma_pi_pi_helicity_solutions):
    for i in jpsi_to_gamma_pi_pi_helicity_solutions:
        dot_data = io.dot.convert_to_dot(i)
        assert pydot.graph_from_dot_data(dot_data) is not None


class TestWrite:
    @staticmethod
    def test_exception():
        with pytest.raises(NotImplementedError):
            io.write(
                instance="nope, can't write a str", filename="dummy_file.gv",
            )

    @staticmethod
    def test_write_single_graph(jpsi_to_gamma_pi_pi_helicity_solutions):
        output_file = "test_single_graph.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions[0],
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    @staticmethod
    def test_write_graph_list(jpsi_to_gamma_pi_pi_helicity_solutions):
        output_file = "test_graph_list.gv"
        io.write(
            instance=jpsi_to_gamma_pi_pi_helicity_solutions,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None
