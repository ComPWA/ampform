import pytest

from expertsystem.amplitude import get_builder
from expertsystem.reaction import Result


@pytest.mark.parametrize(
    "formalism, n_amplitudes", [("canonical", 16), ("helicity", 8)]
)
def test_generate(
    formalism: str,
    n_amplitudes: int,
    jpsi_to_gamma_pi_pi_canonical_solutions: Result,
    jpsi_to_gamma_pi_pi_helicity_solutions: Result,
):
    if formalism == "canonical":
        result = jpsi_to_gamma_pi_pi_canonical_solutions
    elif formalism == "helicity":
        result = jpsi_to_gamma_pi_pi_helicity_solutions
    else:
        raise NotImplementedError
    sympy_model = get_builder(result).generate()
    assert len(sympy_model.parameters) == 2
    assert len(sympy_model.components) == 4 + n_amplitudes
