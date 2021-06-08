# pylint: disable=no-self-use
import numpy as np
import pytest
import qrules as q
import sympy as sp
from sympy import preorder_traversal

from ampform.dynamics import ComplexSqrt
from ampform.helicity import HelicityModel


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
    m = sp.Symbol("m", real=True)
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

    expression = sp.piecewise_fold(expression)
    assert isinstance(expression, sp.Add)
    a1, a2 = tuple(map(str, expression.args))
    a1 = a1.replace("ComplexSqrt", "sqrt")
    a2 = a2.replace("ComplexSqrt", "sqrt")
    if formalism == "canonical":
        assert a1 == "0.08/(-m**2 + 0.98 - 0.12*I*sqrt(m**2/4 - 0.02)/Abs(m))"
        assert a2 == "0.23/(-m**2 + 2.27 - 0.34*I*sqrt(m**2/4 - 0.02)/Abs(m))"
    elif formalism == "helicity":
        assert a1 == "0.17/(-m**2 + 2.27 - 0.34*I*sqrt(m**2/4 - 0.02)/Abs(m))"
        assert a2 == "0.06/(-m**2 + 0.98 - 0.12*I*sqrt(m**2/4 - 0.02)/Abs(m))"
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


class TestComplexSqrt:
    @pytest.mark.parametrize("real", [False, True])
    def test_evaluate(self, real: bool):
        x = sp.Symbol("x", real=real)
        expr = ComplexSqrt(x).evaluate()
        if real:
            assert expr == sp.Piecewise(
                (sp.I * sp.sqrt(-x), x < 0),
                (sp.sqrt(x), True),
            )
        else:
            assert expr == sp.sqrt(x)

    def test_latex(self):
        x = sp.Symbol("x")
        expr = ComplexSqrt(x)
        assert sp.latex(expr) == R"\sqrt[\mathrm{c}]{x}"

    @pytest.mark.parametrize("real", [False, True])
    @pytest.mark.parametrize("backend", ["math", "numpy"])
    def test_lambdify(self, backend: str, real: bool):
        x = sp.Symbol("x", real=real)
        expression = ComplexSqrt(x)
        lambdified = sp.lambdify(x, expression, backend)
        assert lambdified(np.array(-1)) == 1j

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (sp.Symbol("x", real=True), "ComplexSqrt(x)"),
            (sp.Symbol("x"), "ComplexSqrt(x)"),
            (+4, "2"),
            (-4, "2*I"),
        ],
    )
    def test_new(self, input_value, expected: str):
        expr = ComplexSqrt(input_value)
        assert str(expr) == expected
