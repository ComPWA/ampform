# pylint: disable=no-self-use, too-many-arguments
from typing import Tuple

import numpy as np
import pytest
import sympy as sp
from qrules import ParticleCollection
from sympy import preorder_traversal

from ampform.dynamics import ComplexSqrt
from ampform.helicity import HelicityModel


def test_generate(
    amplitude_model: Tuple[str, HelicityModel],
    particle_database: ParticleCollection,
):
    formalism, model = amplitude_model
    if formalism == "canonical-helicity":
        n_amplitudes = 16
        n_parameters = 10
    else:
        n_amplitudes = 8
        n_parameters = 8
    assert len(model.parameter_defaults) == n_parameters
    assert len(model.components) == 4 + n_amplitudes
    assert len(model.expression.free_symbols) == 7 + n_parameters

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
    if formalism == "canonical-helicity":
        assert a1 == "0.08/(-m**2 - 0.06*I*sqrt(m**2 - 0.07)/Abs(m) + 0.98)"
        assert a2 == "0.23/(-m**2 - 0.17*I*sqrt(m**2 - 0.07)/Abs(m) + 2.27)"
    else:
        assert a1 == "0.17/(-m**2 - 0.17*I*sqrt(m**2 - 0.07)/Abs(m) + 2.27)"
        assert a2 == "0.06/(-m**2 - 0.06*I*sqrt(m**2 - 0.07)/Abs(m) + 0.98)"


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
