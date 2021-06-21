import sympy as sp
from qrules import ReactionInfo
from sympy import cos, sin, sqrt

from ampform import get_builder


def test_generate(reaction: ReactionInfo):
    if reaction.formalism == "canonical-helicity":
        n_amplitudes = 16
        n_parameters = 4
    else:
        n_amplitudes = 8
        n_parameters = 2

    model = get_builder(reaction).generate()
    assert len(model.parameter_defaults) == n_parameters
    assert len(model.components) == 4 + n_amplitudes
    assert len(model.expression.free_symbols) == 4 + n_parameters

    no_dynamics: sp.Expr = model.expression.doit()
    no_dynamics = no_dynamics.subs(model.parameter_defaults)
    assert len(no_dynamics.free_symbols) == 1

    existing_theta = next(iter(no_dynamics.free_symbols))
    theta = sp.Symbol("theta", real=True)
    no_dynamics = no_dynamics.subs({existing_theta: theta})
    no_dynamics = no_dynamics.trigsimp()

    if reaction.formalism == "canonical-helicity":
        assert (
            no_dynamics
            == 0.8 * sqrt(10) * cos(theta) ** 2
            + 4.4 * cos(theta) ** 2
            + 0.8 * sqrt(10)
            + 4.4
        )
    else:
        assert no_dynamics == 8.0 - 4.0 * sin(theta) ** 2
