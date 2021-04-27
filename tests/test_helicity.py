import pytest
import sympy as sp
from qrules import Result
from sympy import cos, sin, sqrt

from ampform import get_builder


@pytest.mark.parametrize(
    ("formalism", "n_amplitudes"),
    [
        ("canonical", 16),
        ("helicity", 8),
    ],
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
    model = get_builder(result).generate()
    assert len(model.parameter_defaults) == 2
    assert len(model.components) == 4 + n_amplitudes
    assert len(model.expression.free_symbols) == 6

    no_dynamics: sp.Expr = model.expression.doit()
    no_dynamics = no_dynamics.subs(model.parameter_defaults)
    assert len(no_dynamics.free_symbols) == 1

    existing_theta = next(iter(no_dynamics.free_symbols))
    theta = sp.Symbol("theta", real=True)
    no_dynamics = no_dynamics.subs({existing_theta: theta})
    no_dynamics = no_dynamics.trigsimp()
    if formalism == "canonical":
        assert (
            no_dynamics
            == 0.8 * sqrt(10) * cos(theta) ** 2
            + 4.4 * cos(theta) ** 2
            + 0.8 * sqrt(10)
            + 4.4
        )
    elif formalism == "helicity":
        assert no_dynamics == 8.0 - 4.0 * sin(theta) ** 2
    else:
        raise NotImplementedError
