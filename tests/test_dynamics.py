import pytest
import qrules as q
import sympy as sp
from sympy import preorder_traversal

from ampform.helicity import HelicityModel


@pytest.mark.parametrize(
    "formalism, n_amplitudes", [("canonical", 16), ("helicity", 8)]
)
def test_generate(
    formalism: str,
    n_amplitudes: int,
    jpsi_to_gamma_pi_pi_canonical_amplitude_model: HelicityModel,
    jpsi_to_gamma_pi_pi_helicity_amplitude_model: HelicityModel,
    particle_database: q.ParticleCollection,
):
    if formalism == "canonical":
        model = jpsi_to_gamma_pi_pi_canonical_amplitude_model
    elif formalism == "helicity":
        model = jpsi_to_gamma_pi_pi_helicity_amplitude_model
    else:
        raise NotImplementedError
    assert len(model.parameter_defaults) == 8
    assert len(model.components) == 4 + n_amplitudes
    assert len(model.expression.free_symbols) == 15

    expression: sp.Expr = model.expression.doit()
    expression = expression.subs(model.parameter_defaults)
    assert len(expression.free_symbols) == 5

    angle_value = 0
    angle_substitutions = {
        s: angle_value
        for s in expression.free_symbols
        if s.name.startswith("phi") or s.name.startswith("theta")
    }
    expression = expression.subs(angle_substitutions)
    assert len(expression.free_symbols) == 3

    pi0 = particle_database["pi0"]
    expression = expression.subs(
        {
            sp.Symbol("m_1", real=True): pi0.mass,
            sp.Symbol("m_2", real=True): pi0.mass,
        },
        simultaneous=True,
    )
    assert len(expression.free_symbols) == 1

    existing_symbol = next(iter(expression.free_symbols))
    m = sp.Symbol("m", real=True)  # pylint: disable=invalid-name
    expression = expression.subs({existing_symbol: m})

    expression = round_nested(expression, n_decimals=2)

    assert expression.args[0] == 2
    assert isinstance(expression.args[1], sp.Pow)
    expression = expression.args[1]

    assert expression.args[1] == 2
    assert isinstance(expression.args[0], sp.Abs)
    expression = expression.args[0]

    assert isinstance(expression.args[0], sp.Add)
    expression = expression.args[0]
    assert len(expression.args) == 2

    expression = round_nested(expression, n_decimals=2)
    expression = round_nested(expression, n_decimals=2)

    if formalism == "canonical":
        assert tuple(map(str, expression.args)) == (
            "0.08/(-m**2 + 0.98 - 0.06*I*sqrt(m**2 - 0.07)/m)",
            "0.23/(-m**2 + 2.27 - 0.17*I*sqrt(m**2 - 0.07)/m)",
        )
    elif formalism == "helicity":
        assert tuple(map(str, expression.args)) == (
            "0.17/(-m**2 + 2.27 - 0.17*I*sqrt(m**2 - 0.07)/m)",
            "0.06/(-m**2 + 0.98 - 0.06*I*sqrt(m**2 - 0.07)/m)",
        )
    else:
        raise NotImplementedError


def round_nested(expression: sp.Expr, n_decimals: int) -> sp.Expr:
    for node in preorder_traversal(expression):
        if node.free_symbols:
            continue
        if isinstance(node, (float, sp.Float)):
            expression = expression.subs(node, round(node, n_decimals))
        if isinstance(node, sp.Pow) and node.args[1] == 1 / 2:
            expression = expression.subs(node, round(node.n(), n_decimals))
    return expression
