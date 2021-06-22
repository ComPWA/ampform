# pylint: disable=no-member, no-self-use
import pytest
import sympy as sp
from qrules import ReactionInfo
from sympy import cos, sin, sqrt

from ampform import get_builder
from ampform.helicity import (
    _generate_kinematic_variables,
    formulate_wigner_d,
    group_transitions,
)


class TestAmplitudeBuilder:
    def test_formulate(self, reaction: ReactionInfo):
        if reaction.formalism == "canonical-helicity":
            n_amplitudes = 16
            n_parameters = 4
        else:
            n_amplitudes = 8
            n_parameters = 2

        model_builder = get_builder(reaction)
        model = model_builder.formulate()
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


@pytest.mark.parametrize(
    ("node_id", "mass", "phi", "theta"),
    [
        (0, "m_012", "phi_1+2", "theta_1+2"),
        (1, "m_12", "phi_1,1+2", "theta_1,1+2"),
    ],
)
def test_generate_kinematic_variables(
    reaction: ReactionInfo,
    node_id: int,
    mass: str,
    phi: str,
    theta: str,
):
    for transition in reaction.transitions:
        variables = _generate_kinematic_variables(transition, node_id)
        assert variables[0].name == mass
        assert variables[1].name == phi
        assert variables[2].name == theta


@pytest.mark.parametrize(
    ("transition", "node_id", "expected"),
    [
        (0, 0, "WignerD(1, -1, 1, -phi_1+2, theta_1+2, 0)"),
        (0, 1, "WignerD(0, 0, 0, -phi_1,1+2, theta_1,1+2, 0)"),
        (1, 0, "WignerD(1, -1, -1, -phi_1+2, theta_1+2, 0)"),
        (1, 1, "WignerD(0, 0, 0, -phi_1,1+2, theta_1,1+2, 0)"),
        (2, 0, "WignerD(1, 1, 1, -phi_1+2, theta_1+2, 0)"),
        (2, 1, "WignerD(0, 0, 0, -phi_1,1+2, theta_1,1+2, 0)"),
    ],
)
def test_formulate_wigner_d(
    reaction: ReactionInfo, transition: int, node_id: int, expected: str
):
    if reaction.formalism == "canonical-helicity":
        transition *= 2
    transitions = [
        t
        for t in reaction.transitions
        if t.states[3].particle.name == "f(0)(980)"
    ]
    some_transition = transitions[transition]
    wigner_d = formulate_wigner_d(some_transition, node_id)
    assert str(wigner_d) == expected


def test_group_transitions(reaction: ReactionInfo):
    transition_groups = group_transitions(reaction.transitions)
    assert len(transition_groups) == 4
    for group in transition_groups:
        transition_iter = iter(group)
        first_transition = next(transition_iter)
        for transition in transition_iter:
            assert transition.initial_states == first_transition.initial_states
            assert transition.final_states == first_transition.final_states
