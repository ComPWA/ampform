from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules

import ampform
from ampform.helicity.align.dpd import (
    DalitzPlotDecomposition,
    _collect_outer_state_helicities,
    relabel_edge_ids,
)
from ampform.kinematics.lorentz import create_four_momentum_symbol

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from qrules.transition import ReactionInfo, SpinFormalism


class TestDalitzPlotDecomposition:
    @pytest.fixture(scope="session", params=["canonical-helicity", "helicity"])
    def jpsi_to_k0_sigma_pbar(self, request: SubRequest) -> ReactionInfo:
        formalism: SpinFormalism = request.param
        reaction = qrules.generate_transitions(
            initial_state=("J/psi(1S)", [-1, +1]),
            final_state=["K0", "Sigma+", "p~"],
            allowed_intermediate_particles=["Sigma(1660)", "N(1650)"],
            allowed_interaction_types=["strong"],
            formalism=formalism,
        )
        return relabel_edge_ids(reaction)

    @pytest.mark.parametrize("scalar_initial_state_mass", [False, True])
    @pytest.mark.parametrize("stable_final_state_ids", [None, {1, 2}, {1, 2, 3}])
    def test_free_symbols_kinematic_variables(
        self,
        jpsi_to_k0_sigma_pbar: ReactionInfo,
        scalar_initial_state_mass: bool,
        stable_final_state_ids: set[int] | None,
    ):
        builder = ampform.get_builder(jpsi_to_k0_sigma_pbar)
        builder.config.spin_alignment = DalitzPlotDecomposition(reference_subsystem=1)
        builder.config.scalar_initial_state_mass = scalar_initial_state_mass
        builder.config.stable_final_state_ids = stable_final_state_ids
        model = builder.formulate()

        for expr in model.kinematic_variables.values():
            if not scalar_initial_state_mass:
                assert "m_0" not in {str(s) for s in expr.free_symbols}
            expr = expr.xreplace(model.parameter_defaults)
            assert {str(s) for s in expr.free_symbols} <= {"p1", "p2", "p3"}

        str_variables = {str(s) for s in model.kinematic_variables}
        if scalar_initial_state_mass:
            assert "m_0" not in str_variables

    @pytest.mark.slow
    def test_free_symbols_main_expression(self, jpsi_to_k0_sigma_pbar: ReactionInfo):
        builder = ampform.get_builder(jpsi_to_k0_sigma_pbar)
        builder.config.spin_alignment = DalitzPlotDecomposition(reference_subsystem=1)
        model = builder.formulate()
        substituted_expr = model.expression
        substituted_expr = substituted_expr.xreplace(model.amplitudes)
        substituted_expr = substituted_expr.xreplace(model.kinematic_variables)
        substituted_expr = substituted_expr.xreplace(model.parameter_defaults)
        p1, p2, p3 = map(create_four_momentum_symbol, [1, 2, 3])
        sorted_free_symbols = sorted(substituted_expr.free_symbols, key=str)
        assert str(sorted_free_symbols) == str([p1, p2, p3])


def test_collect_outer_state_helicities(reaction: ReactionInfo):
    helicities = _collect_outer_state_helicities(reaction)
    assert helicities == {
        -1: [-1, +1],
        0: [-1, +1],
        1: [0],
        2: [0],
    }


def test_relabel_edge_ids(reaction: ReactionInfo):
    for transition in reaction.transitions:
        assert set(transition.initial_states) == {-1}
        assert set(transition.final_states) == {0, 1, 2}
        assert set(transition.intermediate_states) == {3}
    assert set(reaction.initial_state) == {-1}
    assert set(reaction.final_state) == {0, 1, 2}

    relabeled_reaction = relabel_edge_ids(reaction)
    for transition in relabeled_reaction.transitions:
        assert set(transition.initial_states) == {0}
        assert set(transition.final_states) == {1, 2, 3}
        assert set(transition.intermediate_states) == {4}
    assert set(relabeled_reaction.initial_state) == {0}
    assert set(relabeled_reaction.final_state) == {1, 2, 3}
