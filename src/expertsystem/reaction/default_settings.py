"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from enum import Enum, auto
from os.path import dirname, join, realpath
from typing import Dict, List, Optional, Tuple, Union

from expertsystem.reaction.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ConservationRule,
    EdgeQNConservationRule,
    ElectronLNConservation,
    GraphElementRule,
    MassConservation,
    MuonLNConservation,
    StrangenessConservation,
    TauLNConservation,
    c_parity_conservation,
    clebsch_gordan_helicity_to_canonical,
    g_parity_conservation,
    gellmann_nishijima,
    helicity_conservation,
    identical_particle_symmetrization,
    isospin_conservation,
    isospin_validity,
    ls_spin_validity,
    parity_conservation,
    parity_conservation_helicity,
    spin_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    NodeQuantumNumbers,
)
from expertsystem.reaction.solving import EdgeSettings, NodeSettings

__EXPERT_SYSTEM_PATH = dirname(dirname(realpath(__file__)))
__DEFAULT_PARTICLE_LIST_FILE = "particle/additional_definitions.yml"
ADDITIONAL_PARTICLES_DEFINITIONS_PATH = join(
    __EXPERT_SYSTEM_PATH, __DEFAULT_PARTICLE_LIST_FILE
)

# If a conservation law is not listed here, a default priority of 1 is assumed.
# Higher number means higher priority
__CONSERVATION_LAW_PRIORITIES: Dict[
    Union[GraphElementRule, EdgeQNConservationRule, ConservationRule], int
] = {
    spin_conservation: 8,
    ls_spin_validity: 89,
    spin_magnitude_conservation: 8,
    helicity_conservation: 7,
    MassConservation: 10,
    ChargeConservation: 100,
    ElectronLNConservation: 45,
    MuonLNConservation: 44,
    TauLNConservation: 43,
    BaryonNumberConservation: 90,
    identical_particle_symmetrization: 2,
    CharmConservation: 70,
    StrangenessConservation: 69,
    parity_conservation: 6,
    c_parity_conservation: 5,
    parity_conservation_helicity: 4,
    isospin_conservation: 60,
    g_parity_conservation: 3,
    BottomnessConservation: 68,
}


__EDGE_RULE_PRIORITIES: Dict[GraphElementRule, int] = {
    gellmann_nishijima: 50,
    isospin_validity: 61,
    spin_validity: 62,
}


class InteractionTypes(Enum):
    """Types of interactions in the form of an enumerate."""

    STRONG = auto()
    EM = auto()
    WEAK = auto()


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
    return magnitudes + [-x for x in magnitudes if x > 0]


def create_default_interaction_settings(
    formalism_type: str,
    nbody_topology: bool = False,
    mass_conservation_factor: Optional[float] = 3.0,
) -> Dict[InteractionTypes, Tuple[EdgeSettings, NodeSettings]]:
    """Create a container that holds the settings for the various interactions.

    E.g.: strong, em and weak interaction.
    """
    interaction_type_settings = {}
    formalism_edge_settings = EdgeSettings(
        conservation_rules={
            isospin_validity,
            gellmann_nishijima,
            spin_validity,
        },
        rule_priorities=__EDGE_RULE_PRIORITIES,
        qn_domains={
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
    formalism_node_settings = NodeSettings(
        rule_priorities=__CONSERVATION_LAW_PRIORITIES
    )

    if "helicity" in formalism_type:
        formalism_node_settings.conservation_rules = {
            spin_magnitude_conservation,
            helicity_conservation,
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
            spin_magnitude_conservation
        }
        if nbody_topology:
            formalism_node_settings.conservation_rules = {
                spin_conservation,
                ls_spin_validity,
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
        formalism_node_settings.conservation_rules.update(
            {
                clebsch_gordan_helicity_to_canonical,
                ls_spin_validity,
            }
        )
        formalism_node_settings.qn_domains.update(
            {
                NodeQuantumNumbers.l_projection: [0],
                NodeQuantumNumbers.s_projection: __create_projections(
                    _get_spin_magnitudes(nbody_topology)
                ),
            }
        )
    if mass_conservation_factor is not None:
        formalism_node_settings.conservation_rules.add(
            MassConservation(mass_conservation_factor)
        )

    weak_node_settings = deepcopy(formalism_node_settings)
    weak_node_settings.conservation_rules.update(
        [
            ChargeConservation(),
            ElectronLNConservation(),
            MuonLNConservation(),
            TauLNConservation(),
            BaryonNumberConservation(),
            identical_particle_symmetrization,
        ]
    )
    weak_node_settings.interaction_strength = 10 ** (-4)
    weak_edge_settings = deepcopy(formalism_edge_settings)

    interaction_type_settings[InteractionTypes.WEAK] = (
        weak_edge_settings,
        weak_node_settings,
    )

    em_node_settings = deepcopy(weak_node_settings)
    em_node_settings.conservation_rules.update(
        {
            CharmConservation(),
            StrangenessConservation(),
            BottomnessConservation(),
            parity_conservation,
            c_parity_conservation,
        }
    )
    if "helicity" in formalism_type:
        em_node_settings.conservation_rules.add(parity_conservation_helicity)
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
        {isospin_conservation, g_parity_conservation}
    )
    strong_node_settings.interaction_strength = 60

    strong_edge_settings = deepcopy(em_edge_settings)

    interaction_type_settings[InteractionTypes.STRONG] = (
        strong_edge_settings,
        strong_node_settings,
    )

    return interaction_type_settings
