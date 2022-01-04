# pylint: disable=no-member, no-self-use
from typing import Tuple

import pytest
import sympy as sp
from qrules import ReactionInfo

from ampform import get_builder
from ampform.helicity import (
    HelicityAmplitudeBuilder,
    HelicityModel,
    _generate_kinematic_variables,
    formulate_wigner_d,
    group_transitions,
)


class TestHelicityAmplitudeBuilder:
    @pytest.mark.parametrize("permutate_topologies", [False, True])
    @pytest.mark.parametrize(
        "stable_final_state_ids", [None, (1, 2), (0, 1, 2)]
    )
    def test_formulate(
        self,
        permutate_topologies,
        reaction: ReactionInfo,
        stable_final_state_ids,
    ):
        # pylint: disable=too-many-locals
        if reaction.formalism == "canonical-helicity":
            n_amplitudes = 16
            n_parameters = 4
        else:
            n_amplitudes = 8
            n_parameters = 2
        n_kinematic_variables = 9
        n_symbols = 4 + n_parameters
        if stable_final_state_ids is not None:
            n_parameters += len(stable_final_state_ids)
            n_kinematic_variables -= len(stable_final_state_ids)

        model_builder: HelicityAmplitudeBuilder = get_builder(reaction)
        model_builder.stable_final_state_ids = stable_final_state_ids
        if permutate_topologies:
            model_builder.adapter.permutate_registered_topologies()
            n_kinematic_variables += 10

        model = model_builder.formulate()
        assert len(model.parameter_defaults) == n_parameters
        assert len(model.components) == 4 + n_amplitudes
        assert len(model.expression.free_symbols) == n_symbols
        assert len(model.kinematic_variables) == n_kinematic_variables

        variables = set(model.kinematic_variables)
        paremeters = set(model.parameter_defaults)
        assert model.expression.free_symbols <= variables | paremeters

        final_state_masses = set(sp.symbols("m_(0:3)", real=True))
        stable_final_state_masses = set()
        if stable_final_state_ids is not None:
            stable_final_state_masses = {
                sp.Symbol(f"m_{i}", real=True) for i in stable_final_state_ids
            }
        unstable_final_state_masses = (
            final_state_masses - stable_final_state_masses
        )
        assert stable_final_state_masses <= paremeters
        assert unstable_final_state_masses <= variables

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
                == 0.8 * sp.sqrt(10) * sp.cos(theta) ** 2
                + 4.4 * sp.cos(theta) ** 2
                + 0.8 * sp.sqrt(10)
                + 4.4
            )
        else:
            assert no_dynamics == 8.0 - 4.0 * sp.sin(theta) ** 2

    def test_stable_final_state_ids(self, reaction: ReactionInfo):
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        assert builder.stable_final_state_ids is None
        builder.stable_final_state_ids = (1, 2)  # type: ignore[assignment]
        assert builder.stable_final_state_ids == {1, 2}
        with pytest.raises(
            ValueError, match=r"^Final state IDs are \[0, 1, 2\].*"
        ):
            builder.stable_final_state_ids = [1, 2, 3]  # type: ignore[assignment]


class TestHelicityModel:
    def test_sum_components(self, amplitude_model: Tuple[str, HelicityModel]):
        # pylint: disable=cell-var-from-loop, line-too-long
        _, model = amplitude_model
        from_intensities = model.sum_components(
            components=filter(lambda c: c.startswith("I"), model.components),
        )
        assert from_intensities == model.expression
        for spin_jpsi in ["-1", "+1"]:
            for spin_gamma in ["-1", "+1"]:
                jpsi_with_spin = fR"J/\psi(1S)_{{{spin_jpsi}}}"
                gamma_with_spin = fR"\gamma_{{{spin_gamma}}}"
                from_amplitudes = model.sum_components(
                    components=filter(
                        lambda c: c.startswith("A")
                        and jpsi_with_spin in c
                        and gamma_with_spin in c,
                        model.components,
                    )
                )
                selected_intensities = filter(
                    lambda c: c.startswith("I")
                    and jpsi_with_spin in c
                    and gamma_with_spin in c,
                    model.components,
                )
                selected_intensity = next(selected_intensities)
                assert from_amplitudes == model.components[selected_intensity]


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
