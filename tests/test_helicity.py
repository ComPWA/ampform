import pytest
from qrules import Result

from ampform import get_builder


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
    assert len(sympy_model.parameter_defaults) == 2
    assert len(sympy_model.components) == 4 + n_amplitudes

    free_symbol_names = {s.name for s in sympy_model.expression.free_symbols}
    assert free_symbol_names == {
        R"C[J/\psi(1S) \to f_{0}(1500)_{0} \gamma_{+1};f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}]",
        R"C[J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1};f_{0}(980) \to \pi^{0}_{0} \pi^{0}_{0}]",
        "phi_1+2",
        "phi_1,1+2",
        "theta_1+2",
        "theta_1,1+2",
    }
