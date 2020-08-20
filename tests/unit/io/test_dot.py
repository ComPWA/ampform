import pydot

import pytest

from expertsystem import io


def test_dot_syntax(jpsi_to_gamma_pi_pi_helicity_solutions):
    for i in jpsi_to_gamma_pi_pi_helicity_solutions:
        dot_data = io.dot.convert_to_dot(i)
        assert pydot.graph_from_dot_data(dot_data) is not None


def test_write_dot(jpsi_to_gamma_pi_pi_helicity_solutions):
    output_filename = "test_write_dot.gv"
    with pytest.raises(NotImplementedError):
        io.write(
            instance="nope, can't write a str", filename=output_filename,
        )
    io.write(
        instance=jpsi_to_gamma_pi_pi_helicity_solutions[0],
        filename=output_filename,
    )
    with open(output_filename, "r") as stream:
        dot_data = stream.read()
    assert pydot.graph_from_dot_data(dot_data) is not None
