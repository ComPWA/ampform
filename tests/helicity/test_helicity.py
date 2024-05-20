from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import qrules
import sympy as sp
from qrules import ReactionInfo

from ampform import get_builder
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.helicity import (
    HelicityAmplitudeBuilder,
    HelicityModel,
    ParameterValue,
    ParameterValues,
    _generate_kinematic_variables,
    formulate_isobar_wigner_d,
    group_by_spin_projection,
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from qrules.transition import SpinFormalism


class TestHelicityAmplitudeBuilder:
    @pytest.mark.parametrize("permutate_topologies", [False, True])
    @pytest.mark.parametrize("stable_final_state_ids", [None, (1, 2), (0, 1, 2)])
    def test_formulate(
        self,
        permutate_topologies,
        reaction: ReactionInfo,
        stable_final_state_ids,
    ):
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
        model_builder.config.stable_final_state_ids = stable_final_state_ids
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
        free_symbols = model.expression.free_symbols
        undefined_symbols = free_symbols - paremeters - variables
        assert not undefined_symbols

        final_state_masses = set(sp.symbols("m_(0:3)", nonnegative=True))
        stable_final_state_masses = set()
        if stable_final_state_ids is not None:
            stable_final_state_masses = {
                sp.Symbol(f"m_{i}", nonnegative=True) for i in stable_final_state_ids
            }
        unstable_final_state_masses = final_state_masses - stable_final_state_masses
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
        assert builder.config.stable_final_state_ids is None
        builder.config.stable_final_state_ids = (1, 2)  # type: ignore[assignment]
        assert builder.config.stable_final_state_ids == {1, 2}

    def test_scalar_initial_state(self, reaction: ReactionInfo):
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        assert builder.config.scalar_initial_state_mass is False
        initial_state_mass = sp.Symbol("m_012", nonnegative=True)

        model = builder.formulate()
        assert initial_state_mass in model.kinematic_variables
        assert initial_state_mass not in model.parameter_defaults

        builder.config.scalar_initial_state_mass = True
        model = builder.formulate()
        assert initial_state_mass not in model.kinematic_variables
        assert initial_state_mass in model.parameter_defaults

    def test_use_helicity_couplings(self, reaction: ReactionInfo):
        # cspell:ignore coeff
        builder: HelicityAmplitudeBuilder = get_builder(reaction)
        builder.config.use_helicity_couplings = False
        coeff_model = builder.formulate()
        builder.config.use_helicity_couplings = True
        coupling_model = builder.formulate()

        coefficient_names = {str(p) for p in coeff_model.parameter_defaults}
        coupling_names = {str(p) for p in coupling_model.parameter_defaults}
        if reaction.formalism == "canonical-helicity":
            assert len(coefficient_names) == 4
            assert coefficient_names == {
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma;"
                R" f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(1500) \gamma;"
                R" f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(1500) \gamma;"
                R" f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma;"
                R" f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
            }
            assert len(coupling_names) == 6
            assert coupling_names == {
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(1500) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(1500) \gamma}",
                R"H_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma}",
                R"H_{f_{0}(1500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"H_{f_{0}(980) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
            }
        else:
            assert len(coefficient_names) == 2
            assert coefficient_names == {
                R"C_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{+1}; f_{0}(980)"
                R" \to \pi^{0}_{0} \pi^{0}_{0}}",
                R"C_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1};"
                R" f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}",
            }
            assert len(coupling_names) == 6
            assert coupling_names == {
                R"H_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{+1}}",
                R"H_{J/\psi(1S) \to {f_{0}(1500)}_{0} \gamma_{-1}}",
                R"H_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{+1}}",
                R"H_{J/\psi(1S) \to {f_{0}(980)}_{0} \gamma_{-1}}",
                R"H_{f_{0}(1500) \to \pi^{0}_{0} \pi^{0}_{0}}",
                R"H_{f_{0}(980) \to \pi^{0}_{0} \pi^{0}_{0}}",
            }


class TestHelicityModel:
    def test_parameter_defaults_item_types(
        self, amplitude_model: tuple[str, HelicityModel]
    ):
        _, model = amplitude_model
        for symbol, value in model.parameter_defaults.items():
            assert isinstance(symbol, sp.Symbol)
            assert isinstance(value, ParameterValue.__args__)  # type: ignore[attr-defined]

    def test_rename_symbols_no_renames(
        self, amplitude_model: tuple[str, HelicityModel]
    ):
        _, model = amplitude_model
        new_model = model.rename_symbols({})
        assert new_model == model

    def test_rename_parameters(self, amplitude_model: tuple[str, HelicityModel]):
        _, model = amplitude_model
        d1, d2 = sp.symbols("d_{f_{0}(980)} d_{f_{0}(1500)}", positive=True)
        assert {d1, d2} <= set(model.parameter_defaults)
        assert {d1, d2} <= model.expression.free_symbols

        new_d = sp.Symbol("d", positive=True)
        new_model = model.rename_symbols({
            d1.name: new_d.name,
            d2.name: new_d.name,
        })
        assert not {d1, d2} & new_model.expression.free_symbols
        assert not {d1, d2} & set(new_model.parameter_defaults)
        assert new_d in new_model.parameter_defaults
        assert new_d in new_model.expression.free_symbols
        assert (
            len(new_model.expression.free_symbols)
            == len(model.expression.free_symbols) - 1
        )
        assert len(new_model.parameter_defaults) == len(model.parameter_defaults) - 1
        assert model.expression.xreplace({d1: new_d, d2: new_d}) == new_model.expression

    @pytest.mark.parametrize("stable_final_states", [False, True])
    def test_rename_all_parameters_with_stable_final_state(
        self,
        reaction: ReactionInfo,
        stable_final_states: bool,
    ):
        builder = get_builder(reaction)
        for name in reaction.get_intermediate_particles().names:
            builder.dynamics.assign(name, create_relativistic_breit_wigner_with_ff)
        if stable_final_states:
            builder.config.stable_final_state_ids = set(reaction.final_state)
        original_model = builder.formulate()
        renames = {
            str(par): Rf"{{{par!s}}}_\mathrm{{renamed}}"
            for par in original_model.parameter_defaults
        }
        new_model = original_model.rename_symbols(renames)
        for par in original_model.parameter_defaults:
            if str(par).startswith("m_") and str(par)[-1] in {"0", "1", "2"}:
                continue
            assert par not in new_model.parameter_defaults
        for par in new_model.parameter_defaults:
            if str(par).startswith("m_") and str(par)[-1] in {"0", "1", "2"}:
                continue
            assert str(par).endswith(R"_\mathrm{renamed}")

    def test_rename_variables(self, amplitude_model: tuple[str, HelicityModel]):
        _, model = amplitude_model
        old_symbol = sp.Symbol("m_12", nonnegative=True)
        assert old_symbol in model.kinematic_variables
        assert old_symbol in model.expression.free_symbols

        new_symbol = sp.Symbol("m_{f_0}", nonnegative=True)
        new_model = model.rename_symbols({old_symbol.name: new_symbol.name})
        assert old_symbol not in new_model.kinematic_variables
        assert old_symbol not in new_model.expression.free_symbols
        assert new_symbol in new_model.kinematic_variables
        assert new_symbol in new_model.expression.free_symbols
        assert (
            model.expression.xreplace({old_symbol: new_symbol}) == new_model.expression
        )

    def test_assumptions_after_rename(self, amplitude_model: tuple[str, HelicityModel]):
        _, model = amplitude_model
        old = "m_{f_{0}(980)}"
        new = "m"
        new_model = model.rename_symbols({old: new})
        assert (
            new_model.parameter_defaults._get_parameter(new).assumptions0
            == model.parameter_defaults._get_parameter(old).assumptions0
        )

    def test_rename_symbols_warnings(
        self,
        amplitude_model: tuple[str, HelicityModel],
        caplog: LogCaptureFixture,
    ):
        _, model = amplitude_model
        old_name = "non-existent"
        with caplog.at_level(logging.WARNING):
            new_model = model.rename_symbols({old_name: "new name"})
        assert caplog.records
        assert old_name in caplog.records[-1].msg
        assert new_model == model

    @pytest.mark.parametrize("formalism", ["canonical-helicity", "helicity"])
    def test_amplitudes(self, formalism: SpinFormalism):
        reaction = qrules.generate_transitions(
            initial_state=("J/psi(1S)", [-1, +1]),
            final_state=["K0", "Sigma+", "p~"],
            allowed_intermediate_particles=["Sigma(1660)~-"],
            allowed_interaction_types=["strong"],
            formalism=formalism,
        )
        assert len(reaction.get_intermediate_particles()) == 1

        builder = get_builder(reaction)
        helicity_combinations = {
            tuple(
                state.spin_projection
                for state_id, state in transition.states.items()
                if state_id not in transition.intermediate_states
            )
            for transition in reaction.transitions
        }
        assert len(helicity_combinations) == 8

        model = builder.formulate()
        assert len(model.amplitudes) == len(helicity_combinations)
        intensity_terms = model.intensity.evaluate().args
        assert len(intensity_terms) == len(helicity_combinations)


class TestParameterValues:
    @pytest.mark.parametrize("subs_method", ["subs", "xreplace"])
    def test_subs_xreplace(self, subs_method: str):
        base = sp.IndexedBase("b")
        a, x, y = sp.symbols("a x y")
        b: sp.Indexed = base[1, 2]
        expr: sp.Expr = a * x + b * y
        parameters = ParameterValues({a: 2, b: -3})
        if subs_method == "subs":
            expr = expr.subs(parameters)
        elif subs_method == "xreplace":
            expr = expr.xreplace(parameters)
        else:
            raise NotImplementedError
        assert expr == 2 * x - 3 * y


@pytest.mark.parametrize(
    ("node_id", "mass", "phi", "theta"),
    [
        (0, "m_012", "phi_0", "theta_0"),
        (1, "m_12", "phi_1^12", "theta_1^12"),
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
        (0, 0, "WignerD(1, -1, -1, -phi_0, theta_0, 0)"),
        (0, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
        (1, 0, "WignerD(1, -1, 1, -phi_0, theta_0, 0)"),
        (1, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
        (2, 0, "WignerD(1, 1, -1, -phi_0, theta_0, 0)"),
        (2, 1, "WignerD(0, 0, 0, -phi_1^12, theta_1^12, 0)"),
    ],
)
def test_formulate_isobar_wigner_d(
    reaction: ReactionInfo, transition: int, node_id: int, expected: str
):
    if reaction.formalism == "canonical-helicity":
        transition *= 2
    transitions = [
        t for t in reaction.transitions if t.states[3].particle.name == "f(0)(980)"
    ]
    some_transition = transitions[transition]
    wigner_d = formulate_isobar_wigner_d(some_transition, node_id)
    assert str(wigner_d) == expected


def test_group_by_spin_projection(reaction: ReactionInfo):
    transition_groups = group_by_spin_projection(reaction.transitions)
    assert len(transition_groups) == 4
    for group in transition_groups:
        transition_iter = iter(group)
        first_transition = next(transition_iter)
        for transition in transition_iter:
            assert transition.initial_states == first_transition.initial_states
            assert transition.final_states == first_transition.final_states
