# pylint: disable=no-member, no-self-use
from typing import Optional

import pytest
import sympy as sp
from qrules import ReactionInfo
from qrules.particle import Particle
from qrules.quantum_numbers import InteractionProperties
from sympy import cos, sin, sqrt

from ampform import get_builder
from ampform.helicity import (
    StateWithID,
    TwoBodyDecay,
    generate_kinematic_variables,
    generate_wigner_d,
)


class TestAmplitudeBuilder:
    def test_generate(self, reaction: ReactionInfo):
        if reaction.formalism == "canonical-helicity":
            n_amplitudes = 16
            n_parameters = 4
        else:
            n_amplitudes = 8
            n_parameters = 2

        model_builder = get_builder(reaction)
        model = model_builder.generate()
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


def _create_dummy_decay(
    l_magnitude: Optional[int], spin_magnitude: float
) -> TwoBodyDecay:
    dummy = Particle(name="dummy", pid=123, spin=spin_magnitude, mass=1.0)
    return TwoBodyDecay(
        parent=StateWithID(
            id=0, particle=dummy, spin_projection=spin_magnitude
        ),
        children=(
            StateWithID(id=1, particle=dummy, spin_projection=0.0),
            StateWithID(id=2, particle=dummy, spin_projection=0.0),
        ),
        interaction=InteractionProperties(l_magnitude=l_magnitude),
    )


class TestTwoBodyDecay:
    @pytest.mark.parametrize(
        ("decay", "expected_l"),
        [
            (_create_dummy_decay(1, 0.5), 1),
            (_create_dummy_decay(0, 1.0), 0),
            (_create_dummy_decay(2, 1.0), 2),
            (_create_dummy_decay(None, 0.0), 0),
            (_create_dummy_decay(None, 1.0), 1),
        ],
    )
    def test_extract_angular_momentum(
        self, decay: TwoBodyDecay, expected_l: int
    ):
        assert expected_l == decay.extract_angular_momentum()

    @pytest.mark.parametrize(
        "decay",
        [
            _create_dummy_decay(None, 0.5),
            _create_dummy_decay(None, 1.5),
        ],
    )
    def test_invalid_angular_momentum(self, decay: TwoBodyDecay):
        with pytest.raises(ValueError, match="not integral"):
            decay.extract_angular_momentum()


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
        variables = generate_kinematic_variables(transition, node_id)
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
def test_generate_wigner_d(
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
    wigner_d = generate_wigner_d(some_transition, node_id)
    assert str(wigner_d) == expected
