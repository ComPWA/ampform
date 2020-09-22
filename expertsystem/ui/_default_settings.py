"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from os.path import (
    dirname,
    join,
    realpath,
)
from typing import (
    Any,
    Dict,
    List,
)

from expertsystem.state.conservation_rules import (
    BaryonNumberConservation,
    CParityConservation,
    ChargeConservation,
    CharmConservation,
    ClebschGordanCheckHelicityToCanonical,
    ElectronLNConservation,
    GParityConservation,
    GellMannNishijimaRule,
    HelicityConservation,
    IdenticalParticleSymmetrization,
    IsoSpinConservation,
    MassConservation,
    MuonLNConservation,
    ParityConservation,
    ParityConservationHelicity,
    SpinConservation,
    StrangenessConservation,
    TauLNConservation,
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


EXPERT_SYSTEM_PATH = dirname(dirname(realpath(__file__)))
DEFAULT_PARTICLE_LIST_FILE = "additional_particle_definitions.yml"
DEFAULT_PARTICLE_LIST_PATH = join(
    EXPERT_SYSTEM_PATH, DEFAULT_PARTICLE_LIST_FILE
)

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


def _get_spin_magnitudes(is_nbody: bool) -> List[float]:
    if is_nbody:
        return [
            0,
        ]
    return [0, 0.5, 1, 1.5, 2]


def _get_ang_mom_magnitudes(is_nbody: bool) -> List[float]:
    if is_nbody:
        return [
            0,
        ]
    return [0, 1, 2]


def create_default_interaction_settings(
    formalism_type: str,
    nbody_topology: bool = False,
    use_mass_conservation: bool = True,
) -> Dict[InteractionTypes, InteractionNodeSettings]:
    """Create a container that holds the settings for the various interactions.

    E.g.: strong, em and weak interaction.
    """
    interaction_type_settings = {}
    formalism_conservation_laws = []
    formalism_qn_domains = {}
    if "helicity" in formalism_type:
        formalism_conservation_laws = [
            SpinConservation(False),
            HelicityConservation(),
        ]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                _get_ang_mom_magnitudes(nbody_topology), True
            ),
            InteractionQuantumNumberNames.S: create_spin_domain(
                _get_spin_magnitudes(nbody_topology), True
            ),
        }
    elif formalism_type == "canonical":
        formalism_conservation_laws = [SpinConservation(not nbody_topology)]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                _get_ang_mom_magnitudes(nbody_topology)
            ),
            InteractionQuantumNumberNames.S: create_spin_domain(
                _get_spin_magnitudes(nbody_topology)
            ),
        }
    if formalism_type == "canonical-helicity":
        formalism_conservation_laws.append(
            ClebschGordanCheckHelicityToCanonical()
        )
    if use_mass_conservation:
        formalism_conservation_laws.append(MassConservation(5))

    weak_settings = InteractionNodeSettings()
    weak_settings.conservation_laws = formalism_conservation_laws
    weak_settings.conservation_laws.extend(
        [
            GellMannNishijimaRule(),
            ChargeConservation(),
            ElectronLNConservation(),
            MuonLNConservation(),
            TauLNConservation(),
            BaryonNumberConservation(),
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
        StateQuantumNumberNames.CParity: [-1, 1, None],
        StateQuantumNumberNames.GParity: [-1, 1, None],
        StateQuantumNumberNames.Spin: create_spin_domain([0, 0.5, 1, 1.5, 2]),
        StateQuantumNumberNames.IsoSpin: create_spin_domain([0, 0.5, 1, 1.5]),
        StateQuantumNumberNames.Charmness: [-1, 0, 1],
        StateQuantumNumberNames.Strangeness: [-1, 0, 1],
    }
    weak_settings.qn_domains.update(formalism_qn_domains)
    weak_settings.interaction_strength = 10 ** (-4)

    interaction_type_settings[InteractionTypes.Weak] = weak_settings

    em_settings = deepcopy(weak_settings)
    em_settings.conservation_laws.extend(
        [
            CharmConservation(),
            StrangenessConservation(),
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
        [IsoSpinConservation(), GParityConservation(),]
    )
    strong_settings.interaction_strength = 60
    interaction_type_settings[InteractionTypes.Strong] = strong_settings

    # reorder conservation laws according to priority
    weak_settings.conservation_laws = _reorder_list_by_priority(
        weak_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )
    em_settings.conservation_laws = _reorder_list_by_priority(
        em_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )
    strong_settings.conservation_laws = _reorder_list_by_priority(
        strong_settings.conservation_laws, CONSERVATION_LAW_PRIORITIES
    )

    return interaction_type_settings


def _reorder_list_by_priority(
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
