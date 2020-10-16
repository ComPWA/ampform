"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from os.path import dirname, join, realpath
from typing import Dict, List, Tuple, Union

from expertsystem.reaction.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ClebschGordanCheckHelicityToCanonical,
    CParityConservation,
    ElectronLNConservation,
    GellMannNishijimaRule,
    GParityConservation,
    HelicityConservation,
    IdenticalParticleSymmetrization,
    IsoSpinConservation,
    IsoSpinValidity,
    MassConservation,
    MuonLNConservation,
    ParityConservation,
    ParityConservationHelicity,
    SpinConservation,
    SpinConservationMagnitude,
    StrangenessConservation,
    TauLNConservation,
)
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    NodeQuantumNumbers,
)
from expertsystem.reaction.solving import (
    EdgeSettings,
    InteractionTypes,
    NodeSettings,
)

__EXPERT_SYSTEM_PATH = dirname(dirname(realpath(__file__)))
__DEFAULT_PARTICLE_LIST_FILE = "additional_particle_definitions.yml"
DEFAULT_PARTICLE_LIST_PATH = join(
    __EXPERT_SYSTEM_PATH, __DEFAULT_PARTICLE_LIST_FILE
)

# If a conservation law is not listed here, a default priority of 1 is assumed.
# Higher number means higher priority
__CONSERVATION_LAW_PRIORITIES = {
    SpinConservation: 8,
    SpinConservationMagnitude: 8,
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
    IsoSpinValidity: 61,
    GParityConservation: 3,
    BottomnessConservation: 68,
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


def __create_projections(
    magnitudes: List[Union[int, float]]
) -> List[Union[int, float]]:
    return magnitudes + list([-x for x in magnitudes if x > 0])


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
        rule_priorities=__CONSERVATION_LAW_PRIORITIES
    )

    if "helicity" in formalism_type:
        formalism_node_settings.conservation_rules = {
            SpinConservationMagnitude(),
            HelicityConservation(),
        }
        formalism_node_settings.qn_domains = {
            NodeQuantumNumbers.l_magnitude: _get_ang_mom_magnitudes(
                nbody_topology
            ),
            NodeQuantumNumbers.s_magnitude: _get_spin_magnitudes(
                nbody_topology
            ),
        }
    elif formalism_type == "canonical":
        formalism_node_settings.conservation_rules = {
            SpinConservationMagnitude()
            if nbody_topology
            else SpinConservation(),
        }
        formalism_node_settings.qn_domains = {
            NodeQuantumNumbers.l_magnitude: _get_ang_mom_magnitudes(
                nbody_topology
            ),
            NodeQuantumNumbers.l_projection: __create_projections(
                _get_ang_mom_magnitudes(nbody_topology)
            ),
            NodeQuantumNumbers.s_magnitude: _get_spin_magnitudes(
                nbody_topology
            ),
            NodeQuantumNumbers.s_projection: __create_projections(
                _get_spin_magnitudes(nbody_topology)
            ),
        }
    if formalism_type == "canonical-helicity":
        formalism_node_settings.conservation_rules.add(
            ClebschGordanCheckHelicityToCanonical()
        )
        formalism_node_settings.qn_domains.update(
            {
                NodeQuantumNumbers.l_projection: [0],
                NodeQuantumNumbers.s_projection: __create_projections(
                    _get_spin_magnitudes(nbody_topology)
                ),
            }
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
            IsoSpinValidity(),  # should be changed to a pure edge rule
            IdenticalParticleSymmetrization(),
            GellMannNishijimaRule(),  # should be changed to a pure edge rule
        ]
    )
    weak_node_settings.interaction_strength = 10 ** (-4)

    weak_edge_settings = deepcopy(formalism_edge_settings)
    weak_edge_settings.qn_domains.update(
        {
            EdgeQuantumNumbers.charge: [-2, -1, 0, 1, 2],
            EdgeQuantumNumbers.baryon_number: [-1, 0, 1],
            EdgeQuantumNumbers.electron_lepton_number: [-1, 0, 1],
            EdgeQuantumNumbers.muon_lepton_number: [-1, 0, 1],
            EdgeQuantumNumbers.tau_lepton_number: [-1, 0, 1],
            EdgeQuantumNumbers.parity: [-1, 1],
            EdgeQuantumNumbers.c_parity: [-1, 1, None],
            EdgeQuantumNumbers.g_parity: [-1, 1, None],
            EdgeQuantumNumbers.spin_magnitude: [0, 0.5, 1, 1.5, 2],
            EdgeQuantumNumbers.spin_projection: __create_projections(
                [0, 0.5, 1, 1.5, 2]
            ),
            EdgeQuantumNumbers.isospin_magnitude: [0, 0.5, 1, 1.5],
            EdgeQuantumNumbers.isospin_projection: __create_projections(
                [0, 0.5, 1, 1.5]
            ),
            EdgeQuantumNumbers.charmness: [-1, 0, 1],
            EdgeQuantumNumbers.strangeness: [-1, 0, 1],
            EdgeQuantumNumbers.bottomness: [-1, 0, 1],
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
            BottomnessConservation(),
            ParityConservation(),
            CParityConservation(),
        }
    )
    if "helicity" in formalism_type:
        em_node_settings.conservation_rules.add(ParityConservationHelicity())
        em_node_settings.qn_domains.update(
            {NodeQuantumNumbers.parity_prefactor: [-1, 1]}
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
