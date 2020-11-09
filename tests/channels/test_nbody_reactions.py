# pylint: disable=redefined-outer-name
from typing import FrozenSet, Set, Union

import pytest

from expertsystem.reaction import check_reaction_violations


def reduce_violated_rules(
    violated_rules: Set[FrozenSet[str]],
) -> Set[Union[str, FrozenSet[str]]]:
    reduced_violations: Set[Union[str, FrozenSet[str]]] = set()
    for rule_group in violated_rules:
        if len(rule_group) == 1:
            reduced_violations.add(tuple(rule_group)[0])
        else:
            reduced_violations.add(rule_group)

    return reduced_violations


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            (["p", "p~"], ["pi+", "pi0"]),
            {"ChargeConservation", "isospin_conservation"},
        ),
        ((["eta"], ["gamma", "gamma"]), {}),
        (
            (["eta"], ["pi+", "pi-"]),
            {
                (
                    "c_parity_conservation",
                    "parity_conservation",
                    "spin_magnitude_conservation",
                )
            },
        ),
        (
            (["Sigma0"], ["Lambda", "pi0"]),
            {"MassConservation"},
        ),
        (
            (["Sigma-"], ["n", "pi-"]),
            {
                "isospin_conservation",
                "StrangenessConservation",
            },
        ),
        ((["e+", "e-"], ["mu+", "mu-"]), {}),
        (
            (["mu-"], ["e-", "nu(e)~"]),
            {"MuonLNConservation", "spin_magnitude_conservation"},
        ),
        (
            (["mu-"], ["e-", "nu(e)"]),
            {
                "ElectronLNConservation",
                "MuonLNConservation",
                "spin_magnitude_conservation",
            },
        ),
        ((["Delta(1232)+"], ["p", "pi0"]), {}),
        ((["nu(e)~", "p"], ["n", "e+"]), {}),
        (
            (["e-", "p"], ["nu(e)", "pi0"]),
            {"BaryonNumberConservation", "spin_magnitude_conservation"},
        ),
        ((["f(0)(980)"], ["pi+", "pi-"]), {}),
        ((["pi0"], ["gamma", "gamma"]), {}),
        (
            (["pi0"], ["gamma", "gamma", "gamma"]),
            {"c_parity_conservation"},
        ),
        ((["pi0"], ["e+", "e-", "gamma"]), {}),
        ((["pi0"], ["e+", "e-"]), {}),
        (
            (["J/psi(1S)"], ["pi0", "f(0)(980)"]),
            {
                "isospin_conservation",
                "c_parity_conservation",
                ("parity_conservation", "spin_magnitude_conservation"),
            },
        ),
        ((["p", "p"], ["Sigma+", "n", "K0", "pi+", "pi0"]), {}),
        (
            (["p", "p"], ["Sigma+", "n", "K~0", "pi+", "pi0"]),
            {"StrangenessConservation", "isospin_conservation"},
        ),
        (
            (["p"], ["e+", "gamma"]),
            {"ElectronLNConservation", "BaryonNumberConservation"},
        ),
        ((["p", "p"], ["p", "p", "p", "p~"]), {}),
        ((["n", "n~"], ["pi+", "pi-", "pi0"]), {}),
        (
            (["pi+", "n"], ["pi-", "p"]),
            {"ChargeConservation", "isospin_conservation"},
        ),
        (
            (["K-"], ["pi-", "pi0"]),
            {
                "isospin_conservation",
                "StrangenessConservation",
                ("parity_conservation", "spin_magnitude_conservation"),
            },
        ),
        (
            (["Sigma+", "n"], ["Sigma-", "p"]),
            {"ChargeConservation", "isospin_conservation"},
        ),
        ((["Sigma0"], ["Lambda", "gamma"]), []),
        (
            (["Xi-"], ["Lambda", "pi-"]),
            {"StrangenessConservation", "isospin_conservation"},
        ),
        (
            (["Xi0"], ["p", "pi-"]),
            {"StrangenessConservation", "isospin_conservation"},
        ),
        ((["pi-", "p"], ["Lambda", "K0"]), {}),
        ((["pi0"], ["gamma", "gamma"]), {}),
        ((["pi0"], ["gamma", "gamma", "gamma"]), {"c_parity_conservation"}),
        ((["Sigma-"], ["n", "e-", "nu(e)~"]), {"StrangenessConservation"}),
        (
            (["rho(770)0"], ["pi0", "pi0"]),
            {
                "isospin_conservation",  # Clebsch Gordan coefficient = 0
                "c_parity_conservation",
                "identical_particle_symmetrization",
            },
        ),
        ((["rho(770)0"], ["gamma", "gamma"]), {"c_parity_conservation"}),
        ((["rho(770)0"], ["gamma", "gamma", "gamma"]), {}),
        (
            (["J/psi(1S)"], ["pi0", "eta"]),
            {"c_parity_conservation", "isospin_conservation"},
        ),
        (
            (["J/psi(1S)"], ["rho(770)0", "rho(770)0"]),
            {"c_parity_conservation", "g_parity_conservation"},
        ),
        (
            (["K~0"], ["pi+", "pi-", "pi0"]),
            {"isospin_conservation", "StrangenessConservation"},
        ),
    ],
)
def test_nbody_reaction(test_input, expected):
    violations = check_reaction_violations(
        initial_state=test_input[0],
        final_state=test_input[1],
    )

    reduced_violations = reduce_violated_rules(violations)
    assert reduced_violations == {
        frozenset(x) if isinstance(x, tuple) else x for x in expected
    }
