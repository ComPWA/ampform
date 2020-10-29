import pytest

import expertsystem as es
from expertsystem.reaction import InteractionTypes as IT  # noqa: N817
from expertsystem.reaction import (
    StateTransitionManager,
    _determine_interaction_types,
)


class TestStateTransitionManager:
    @staticmethod
    def test_allowed_intermediate_particles():
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["p", "p~", "eta"],
        )
        particle_name = "N(753)"
        try:
            stm.set_allowed_intermediate_particles([particle_name])
        except LookupError as exception:
            assert particle_name in exception.args[0]


@pytest.mark.parametrize(
    "description, expected",
    [
        ("all", {IT.Strong, IT.Weak, IT.EM}),
        ("EM", {IT.EM}),
        ("electromagnetic", {IT.EM}),
        ("electro-weak", {IT.EM, IT.Weak}),
        ("ew", {IT.EM, IT.Weak}),
        ("w", {IT.Weak}),
        ("strong", {IT.Strong}),
        ("only strong", {IT.Strong}),
        ("S", {IT.Strong}),
        (["e", "s", "w"], {IT.Strong, IT.Weak, IT.EM}),
        ("S", {IT.Strong}),
        ("strong and EM", {IT.Strong, IT.EM}),
        ("", ValueError),
        ("non-existing", ValueError),
    ],
)
def test_determine_interaction_types(description, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            assert _determine_interaction_types(description)
    else:
        assert _determine_interaction_types(description) == expected


@pytest.mark.parametrize(
    "initial_state, final_state, interaction_types, n_intermediate_states",
    [
        ("pi0", ["gamma", "gamma", "gamma"], "EM", ValueError),
        ("pi0", ["gamma", "gamma"], "EM", 0),
        ("pi0", ["e+", "e-"], "EM", 0),
    ],
)
def test_check_reaction(
    initial_state,
    final_state,
    interaction_types,
    n_intermediate_states,
):
    if n_intermediate_states is ValueError:
        with pytest.raises(ValueError):
            es.reaction.check(initial_state, final_state, interaction_types)
    else:
        assert (
            len(
                es.reaction.check(
                    initial_state, final_state, interaction_types
                )
            )
            == n_intermediate_states
        )
