"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from os.path import (
    dirname,
    join,
    realpath,
)
from typing import Dict, List, Tuple

from expertsystem.nested_dicts import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
)
from expertsystem.solving import (
    EdgeSettings,
    InteractionTypes,
    NodeSettings,
)
from expertsystem.solving.conservation_rules import (
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
from expertsystem.state.properties import create_spin_domain


EXPERT_SYSTEM_PATH = dirname(dirname(realpath(__file__)))
DEFAULT_PARTICLE_LIST_FILE = "additional_particle_definitions.yml"
DEFAULT_PARTICLE_LIST_PATH = join(
    EXPERT_SYSTEM_PATH, DEFAULT_PARTICLE_LIST_FILE
)

# If a conservation law is not listed here, a default priority of 1 is assumed.
# Higher number means higher priority
CONSERVATION_LAW_PRIORITIES = {
    SpinConservation: 8,
    HelicityConservation: 7,
    MassConservation: 10,
    GellMannNishijimaRule: 50,
    ChargeConservation: 100,
    ElectronLNConservation: 45,
    MuonLNConservation: 44,
    TauLNConservation: 43,
    BaryonNumberConservation: 90,
    IdenticalParticleSymmetrization: 2,
    CharmConservation: 70,
    StrangenessConservation: 69,
    ParityConservation: 6,
    CParityConservation: 5,
    ParityConservationHelicity: 4,
    IsoSpinConservation: 60,
    GParityConservation: 3,
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
) -> Dict[InteractionTypes, Tuple[EdgeSettings, NodeSettings]]:
    """Create a container that holds the settings for the various interactions.

    E.g.: strong, em and weak interaction.
    """
    interaction_type_settings = {}
    formalism_edge_settings = EdgeSettings()
    formalism_node_settings = NodeSettings(
        rule_priorities=CONSERVATION_LAW_PRIORITIES
    )

    if "helicity" in formalism_type:
        formalism_node_settings.conservation_rules = {
            SpinConservation(False),
            HelicityConservation(),
        }
        formalism_node_settings.qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                _get_ang_mom_magnitudes(nbody_topology), True
            ),
            InteractionQuantumNumberNames.S: create_spin_domain(
                _get_spin_magnitudes(nbody_topology), True
            ),
        }
    elif formalism_type == "canonical":
        formalism_node_settings.conservation_rules = {
            SpinConservation(not nbody_topology),
        }
        formalism_node_settings.qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                _get_ang_mom_magnitudes(nbody_topology)
            ),
            InteractionQuantumNumberNames.S: create_spin_domain(
                _get_spin_magnitudes(nbody_topology)
            ),
        }
    if formalism_type == "canonical-helicity":
        formalism_node_settings.conservation_rules.add(
            ClebschGordanCheckHelicityToCanonical()
        )
    if use_mass_conservation:
        formalism_node_settings.conservation_rules.add(MassConservation(5))

    weak_node_settings = deepcopy(formalism_node_settings)
    weak_node_settings.conservation_rules.update(
        [
            ChargeConservation(),
            ElectronLNConservation(),
            MuonLNConservation(),
            TauLNConservation(),
            BaryonNumberConservation(),
            IdenticalParticleSymmetrization(),
            GellMannNishijimaRule(),  # should be changed to a pure edge rule
        ]
    )
    weak_node_settings.interaction_strength = 10 ** (-4)

    weak_edge_settings = deepcopy(formalism_edge_settings)
    weak_edge_settings.qn_domains.update(
        {
            StateQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
            StateQuantumNumberNames.BaryonNumber: [-1, 0, 1],
            StateQuantumNumberNames.ElectronLN: [-1, 0, 1],
            StateQuantumNumberNames.MuonLN: [-1, 0, 1],
            StateQuantumNumberNames.TauLN: [-1, 0, 1],
            StateQuantumNumberNames.Parity: [-1, 1],
            StateQuantumNumberNames.CParity: [-1, 1, None],
            StateQuantumNumberNames.GParity: [-1, 1, None],
            StateQuantumNumberNames.Spin: create_spin_domain(
                [0, 0.5, 1, 1.5, 2]
            ),
            StateQuantumNumberNames.IsoSpin: create_spin_domain(
                [0, 0.5, 1, 1.5]
            ),
            StateQuantumNumberNames.Charmness: [-1, 0, 1],
            StateQuantumNumberNames.Strangeness: [-1, 0, 1],
        },
    )

    interaction_type_settings[InteractionTypes.Weak] = (
        weak_edge_settings,
        weak_node_settings,
    )

    em_node_settings = deepcopy(weak_node_settings)
    em_node_settings.conservation_rules.update(
        {
            CharmConservation(),
            StrangenessConservation(),
            ParityConservation(),
            CParityConservation(),
        }
    )
    if "helicity" in formalism_type:
        em_node_settings.conservation_rules.add(ParityConservationHelicity())
        em_node_settings.qn_domains.update(
            {InteractionQuantumNumberNames.ParityPrefactor: [-1, 1]}
        )
    em_node_settings.interaction_strength = 1

    em_edge_settings = deepcopy(weak_edge_settings)

    interaction_type_settings[InteractionTypes.EM] = (
        em_edge_settings,
        em_node_settings,
    )

    strong_node_settings = deepcopy(em_node_settings)
    strong_node_settings.conservation_rules.update(
        {IsoSpinConservation(), GParityConservation()}
    )
    strong_node_settings.interaction_strength = 60

    strong_edge_settings = deepcopy(em_edge_settings)

    interaction_type_settings[InteractionTypes.Strong] = (
        strong_edge_settings,
        strong_node_settings,
    )

    return interaction_type_settings
