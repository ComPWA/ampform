# pylint: disable=redefined-outer-name
from typing import Dict, Set, Tuple

import pytest

from expertsystem.reaction.solving import Rule
from expertsystem.ui import (
    InteractionTypes,
    SolvingMode,
    StateTransitionManager,
)


def reduce_violated_rules(
    violated_rules: Dict[int, Set[Tuple[Rule]]]
) -> Set[str]:
    reduced_violations = set()
    for rule_set in violated_rules.values():
        for rule_group in rule_set:
            reduced_violations.update(set(str(x) for x in rule_group))

    return reduced_violations


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            (["p", "p~"], ["pi+", "pi0"]),
            ["ChargeConservation", "IsoSpinConservation"],
        ),
        ((["eta"], ["gamma", "gamma"]), []),
        ((["Sigma0"], ["Lambda", "pi0"]), ["MassConservation"]),
        ((["Sigma-"], ["n", "pi-"]), []),
        ((["e+", "e-"], ["mu+", "mu-"]), []),
        # (
        #     (["mu-"], ["e-", "nu(e)~"]),
        #     ["MuonLNConservation", "SpinConservation"],
        # ),
        # this is just an additional lepton number test
        # (
        #     (["mu-"], ["e-", "nu(e)"]),
        #     [
        #         "ElectronLNConservation",
        #         "MuonLNConservation",
        #         "SpinConservation",
        #     ],
        # ),
        ((["Delta(1232)+"], ["p", "pi0"]), []),
        ((["nu(e)~", "p"], ["n", "e+"]), []),
        # (
        #     (["e-", "p"], ["nu(e)", "pi0"]),
        #     ["BaryonNumberConservation", "SpinConservation"],
        # ),
        ((["p", "p"], ["Sigma+", "n", "K~0", "pi+", "pi0"]), []),
        (
            (["p"], ["e+", "gamma"]),
            ["ElectronLNConservation", "BaryonNumberConservation"],
        ),
        ((["p", "p"], ["p", "p", "p", "p~"]), []),
        ((["n", "n~"], ["pi+", "pi-", "pi0"]), []),
        (
            (["pi+", "n"], ["pi-", "p"]),
            ["ChargeConservation", "IsoSpinConservation"],
        ),
        ((["K-"], ["pi-", "pi0"]), []),
        (
            (["Sigma+", "n"], ["Sigma-", "p"]),
            ["ChargeConservation", "IsoSpinConservation"],
        ),
        ((["Sigma0"], ["Lambda", "gamma"]), []),
        ((["Xi-"], ["Lambda", "pi-"]), []),
        ((["Xi0"], ["p", "pi-"]), []),
        ((["pi-", "p"], ["Lambda", "K~0"]), []),
        ((["pi0"], ["gamma", "gamma"]), []),
        ((["pi0"], ["gamma", "gamma", "gamma"]), []),
        ((["Sigma-"], ["n", "e-", "nu(e)~"]), []),
        # (
        #     (["rho(770)0"], ["pi0", "pi0"]),
        #     [
        #         "IsoSpinConservation",  # Clebsch Gordan coefficient = 0
        #         "CParityConservation",
        #         "IdenticalParticleSymmetrization",
        #     ],
        # ),
        ((["rho(770)0"], ["gamma", "gamma"]), []),
        ((["J/psi(1S)"], ["pi0", "eta"]), []),
        ((["J/psi(1S)"], ["rho(770)0", "rho(770)0"]), []),
        ((["K~0"], ["pi+", "pi-", "pi0"]), []),
    ],
)
def test_general_reaction(test_input, expected):
    # define all of the different decay scenarios
    print("processing case:" + str(test_input))

    stm = StateTransitionManager(
        initial_state=test_input[0],
        final_state=test_input[1],
        formalism_type="canonical",
        topology_building="nbody"
        if len(test_input[0]) > 1 or len(test_input[1]) > 2
        else "isobar",
        solving_mode=SolvingMode.Full,
        number_of_threads=1,
    )

    graph_interaction_settings = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings)

    if len(result.solutions) > 0:
        print("is valid")
        assert len(expected) == 0
    else:
        reduced_violations = reduce_violated_rules(result.violated_rules)
        print("not allowed! violates: " + str(reduced_violations))
        assert reduced_violations == set(expected)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((["f(0)(980)"], ["pi+", "pi-"]), []),
        # (
        #     (["J/psi(1S)"], ["pi0", "f(0)(980)"]),
        #     ["CParityConservation", "SpinConservation"],
        # ),
        ((["pi0"], ["gamma", "gamma"]), []),
        (
            (["pi0"], ["gamma", "gamma", "gamma"]),
            ["CParityConservation"],
        ),
        ((["pi0"], ["e+", "e-", "gamma"]), []),
        ((["pi0"], ["e+", "e-"]), []),
    ],
)
def test_em_reactions(test_input, expected):
    stm = StateTransitionManager(
        initial_state=test_input[0],
        final_state=test_input[1],
        formalism_type="canonical",
        topology_building="nbody"
        if len(test_input[0]) > 1 or len(test_input[1]) > 2
        else "isobar",
        solving_mode=SolvingMode.Full,
        number_of_threads=1,
    )

    stm.set_allowed_interaction_types([InteractionTypes.EM])

    graph_interaction_settings = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings)

    reduced_violations = reduce_violated_rules(result.violated_rules)
    assert reduced_violations == set(expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((["f(0)(980)"], ["pi+", "pi-"]), []),
        # (
        #     (["J/psi(1S)"], ["pi0", "f(0)(980)"]),
        #     ["IsoSpinConservation", "CParityConservation", "SpinConservation"],
        # ),
    ],
)
def test_strong_reactions(test_input, expected):
    stm = StateTransitionManager(
        initial_state=test_input[0],
        final_state=test_input[1],
        formalism_type="canonical",
        topology_building="nbody"
        if len(test_input[0]) > 1 or len(test_input[1]) > 2
        else "isobar",
        solving_mode=SolvingMode.Full,
        number_of_threads=1,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])

    graph_interaction_settings = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings)

    reduced_violations = reduce_violated_rules(result.violated_rules)
    assert set(reduced_violations) == set(expected)
