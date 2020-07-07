"""Default configuration for the `expertsystem`."""

import sys
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
)

from expertsystem.state.conservation_rules import (
    AdditiveQuantumNumberConservation,
    CParityConservation,
    ClebschGordanCheckHelicityToCanonical,
    GParityConservation,
    GellMannNishijimaRule,
    HelicityConservation,
    IdenticalParticleSymmetrization,
    MassConservation,
    ParityConservation,
    ParityConservationHelicity,
    SpinConservation,
)
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
    create_spin_domain,
)
from expertsystem.state.propagation import (
    InteractionNodeSettings,
    InteractionTypes,
)


SYSTEM_SEARCH_PATHS = [
    ".",
    "..",
]
SYSTEM_SEARCH_PATHS += sys.path

# If a conservation law is not listed here, a default priority of 1 is assumed.
# Higher number means higher priority
CONSERVATION_LAW_PRIORITIES = {
    "SpinConservation": 8,
    "HelicityConservation": 7,
    "MassConservation": 10,
    "GellMannNishijimaRule": 50,
    "ChargeConservation": 100,
    "ElectronLNConservation": 45,
    "MuonLNConservation": 44,
    "TauLNConservation": 43,
    "BaryonNumberConservation": 90,
    "IdenticalParticleSymmetrization": 2,
    "CharmConservation": 70,
    "StrangenessConservation": 69,
    "ParityConservation": 6,
    "CParityConservation": 5,
    "ParityConservationHelicity": 4,
    "IsoSpinConservation": 60,
    "GParityConservation": 3,
}


def create_default_interaction_settings(
    formalism_type: str, use_mass_conservation: bool = True
) -> Dict[InteractionTypes, InteractionNodeSettings]:
    """Create a container that holds the settings for the various interactions.

    E.g.: strong, em and weak interaction.
    """
    interaction_type_settings = {}
    formalism_conservation_laws = []
    formalism_qn_domains = {}
    if "helicity" in formalism_type:
        formalism_conservation_laws = [
            SpinConservation(StateQuantumNumberNames.Spin, False),
            HelicityConservation(),
        ]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                [0, 1, 2], True
            ),
            InteractionQuantumNumberNames.S: create_spin_domain(
                [0, 0.5, 1, 1.5, 2], True
            ),
        }
    elif formalism_type == "canonical":
        formalism_conservation_laws = [
            SpinConservation(StateQuantumNumberNames.Spin)
        ]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain([0, 1, 2]),
            InteractionQuantumNumberNames.S: create_spin_domain(
                [0, 0.5, 1, 2]
            ),
        }
    if formalism_type == "canonical-helicity":
        formalism_conservation_laws.append(
            ClebschGordanCheckHelicityToCanonical()
        )
    if use_mass_conservation:
        formalism_conservation_laws.append(MassConservation())

    weak_settings = InteractionNodeSettings()
    weak_settings.conservation_laws = formalism_conservation_laws
    weak_settings.conservation_laws.extend(
        [
            GellMannNishijimaRule(),
            AdditiveQuantumNumberConservation(StateQuantumNumberNames.Charge),
            AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.ElectronLN
            ),
            AdditiveQuantumNumberConservation(StateQuantumNumberNames.MuonLN),
            AdditiveQuantumNumberConservation(StateQuantumNumberNames.TauLN),
            AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.BaryonNumber
            ),
            IdenticalParticleSymmetrization(),
        ]
    )
    weak_settings.qn_domains = {
        StateQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
        StateQuantumNumberNames.BaryonNumber: [-1, 0, 1],
        StateQuantumNumberNames.ElectronLN: [-1, 0, 1],
        StateQuantumNumberNames.MuonLN: [-1, 0, 1],
        StateQuantumNumberNames.TauLN: [-1, 0, 1],
        StateQuantumNumberNames.Parity: [-1, 1],
        StateQuantumNumberNames.Cparity: [-1, 1, None],
        StateQuantumNumberNames.Gparity: [-1, 1, None],
        StateQuantumNumberNames.Spin: create_spin_domain([0, 0.5, 1, 1.5, 2]),
        StateQuantumNumberNames.IsoSpin: create_spin_domain([0, 0.5, 1, 1.5]),
        StateQuantumNumberNames.Charm: [-1, 0, 1],
        StateQuantumNumberNames.Strangeness: [-1, 0, 1],
    }
    weak_settings.qn_domains.update(formalism_qn_domains)
    weak_settings.interaction_strength = 10 ** (-4)

    interaction_type_settings[InteractionTypes.Weak] = weak_settings

    em_settings = deepcopy(weak_settings)
    em_settings.conservation_laws.extend(
        [
            AdditiveQuantumNumberConservation(StateQuantumNumberNames.Charm),
            AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.Strangeness
            ),
            ParityConservation(),
            CParityConservation(),
        ]
    )
    if "helicity" in formalism_type:
        em_settings.conservation_laws.append(ParityConservationHelicity())
        em_settings.qn_domains.update(
            {InteractionQuantumNumberNames.ParityPrefactor: [-1, 1]}
        )
    em_settings.interaction_strength = 1

    interaction_type_settings[InteractionTypes.EM] = em_settings

    strong_settings = deepcopy(em_settings)
    strong_settings.conservation_laws.extend(
        [
            SpinConservation(StateQuantumNumberNames.IsoSpin),
            GParityConservation(),
        ]
    )
    strong_settings.interaction_strength = 60
    interaction_type_settings[InteractionTypes.Strong] = strong_settings

    # reorder conservation laws according to priority
    weak_settings.conservation_laws = reorder_list_by_priority(
        weak_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )
    em_settings.conservation_laws = reorder_list_by_priority(
        em_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )
    strong_settings.conservation_laws = reorder_list_by_priority(
        strong_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )

    return interaction_type_settings


def reorder_list_by_priority(
    some_list: List[Any], priority_mapping: Dict[str, Any]
) -> List[Any]:
    # first add priorities to the entries
    priority_list = [
        (x, priority_mapping[str(x)]) if str(x) in priority_mapping else (x, 1)
        for x in some_list
    ]
    # then sort according to priority
    sorted_list = sorted(priority_list, key=lambda x: x[1], reverse=True)
    # and strip away the priorities again
    return [x[0] for x in sorted_list]
