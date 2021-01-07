"""Implementation of the helicity formalism for amplitude model generation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

from expertsystem.particle import ParticleCollection, Spin
from expertsystem.reaction import Result
from expertsystem.reaction.combinatorics import (
    perform_external_edge_identical_particle_combinatorics,
)
from expertsystem.reaction.quantum_numbers import ParticleWithSpin
from expertsystem.reaction.topology import StateTransitionGraph

from .model import (
    AmplitudeModel,
    AmplitudeNode,
    CoefficientAmplitude,
    CoherentIntensity,
    DecayNode,
    DecayProduct,
    FitParameter,
    FitParameters,
    HelicityDecay,
    HelicityParticle,
    IncoherentIntensity,
    IntensityNode,
    Kinematics,
    ParticleDynamics,
    RecoilSystem,
    SequentialAmplitude,
)


def _group_graphs_same_initial_and_final(
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
) -> List[List[StateTransitionGraph[ParticleWithSpin]]]:
    """Match final and initial states in groups.

    Each graph corresponds to a specific state transition amplitude.
    This function groups together graphs, which have the same initial and final
    state (including spin). This is needed to determine the coherency of the
    individual amplitude parts.
    """
    graph_groups: Dict[
        Tuple[tuple, tuple], List[StateTransitionGraph[ParticleWithSpin]]
    ] = dict()
    for graph in graphs:
        ise = graph.get_final_state_edge_ids()
        fse = graph.get_initial_state_edge_ids()
        graph_group = (
            tuple(
                sorted(
                    [
                        (
                            graph.get_edge_props(x)[0].name,
                            graph.get_edge_props(x)[1],
                        )
                        for x in ise
                    ]
                )
            ),
            tuple(
                sorted(
                    [
                        (
                            graph.get_edge_props(x)[0].name,
                            graph.get_edge_props(x)[1],
                        )
                        for x in fse
                    ]
                )
            ),
        )
        if graph_group not in graph_groups:
            graph_groups[graph_group] = []
        graph_groups[graph_group].append(graph)

    graph_group_list = list(graph_groups.values())
    return graph_group_list


def _get_graph_group_unique_label(
    graph_group: List[StateTransitionGraph[ParticleWithSpin]],
) -> str:
    label = ""
    if graph_group:
        ise = graph_group[0].get_initial_state_edge_ids()
        fse = graph_group[0].get_final_state_edge_ids()
        is_names = _get_name_hel_list(graph_group[0], ise)
        fs_names = _get_name_hel_list(graph_group[0], fse)
        label += (
            _generate_particles_string(is_names)
            + "_to_"
            + _generate_particles_string(fs_names)
        )
    return label


def _determine_attached_final_state(
    graph: StateTransitionGraph[ParticleWithSpin], edge_id: int
) -> List[int]:
    """Determine all final state particles of a graph.

    These are attached downward (forward in time) for a given edge (resembling
    the root).
    """
    final_state_edge_ids = []
    all_final_state_edges = graph.get_final_state_edge_ids()
    current_edges = [edge_id]
    while current_edges:
        temp_current_edges = current_edges
        current_edges = []
        for current_edge in temp_current_edges:
            if current_edge in all_final_state_edges:
                final_state_edge_ids.append(current_edge)
            else:
                node_id = graph.edges[current_edge].ending_node_id
                if node_id:
                    current_edges.extend(
                        graph.get_edge_ids_outgoing_from_node(node_id)
                    )
    return final_state_edge_ids


def _get_recoil_edge(
    graph: StateTransitionGraph[ParticleWithSpin], edge_id: int
) -> Optional[int]:
    """Determine the id of the recoil edge for the specified edge of a graph."""
    node_id = graph.edges[edge_id].originating_node_id
    if node_id is None:
        return None
    outgoing_edges = graph.get_edge_ids_outgoing_from_node(node_id)
    outgoing_edges.remove(edge_id)
    if len(outgoing_edges) != 1:
        raise ValueError(
            f"The node with id {node_id} has more than 2 outgoing edges:\n"
            + str(graph)
        )
    return outgoing_edges[0]


def _get_parent_recoil_edge(
    graph: StateTransitionGraph[ParticleWithSpin], edge_id: int
) -> Optional[int]:
    """Determine the id of the recoil edge of the parent edge."""
    node_id = graph.edges[edge_id].originating_node_id
    if node_id is None:
        return None
    ingoing_edges = graph.get_edge_ids_ingoing_to_node(node_id)
    if len(ingoing_edges) != 1:
        raise ValueError(
            f"The node with id {node_id} does not have a single ingoing edge!\n"
            + str(graph)
        )
    return _get_recoil_edge(graph, ingoing_edges[0])


def _get_prefactor(
    graph: StateTransitionGraph[ParticleWithSpin],
) -> float:
    """Calculate the product of all prefactors defined in this graph."""
    prefactor = 1.0
    for node_id in graph.nodes:
        node_props = graph.get_node_props(node_id)
        if node_props:
            temp_prefactor = __validate_float_type(node_props.parity_prefactor)
            if temp_prefactor is not None:
                prefactor *= temp_prefactor
    return prefactor


def _generate_particle_collection(
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
) -> ParticleCollection:
    particles = ParticleCollection()
    for graph in graphs:
        for edge_props in map(graph.get_edge_props, graph.edges):
            particle_name = edge_props[0].name
            if particle_name not in particles:
                particles.add(edge_props[0])
    return particles


def _generate_kinematics(
    result: Result, particles: ParticleCollection
) -> Kinematics:
    kinematics = Kinematics(particles)
    initial_state = [p.name for p in result.get_initial_state()]
    final_state = [p.name for p in result.get_final_state()]
    kinematics.set_reaction(
        initial_state=initial_state,
        final_state=final_state,
        intermediate_states=len(
            result.solutions[0].get_intermediate_state_edge_ids()
        ),
    )
    return kinematics


def _generate_particles_string(
    name_hel_list: List[Tuple[str, float]],
    use_helicity: bool = True,
    make_parity_partner: bool = False,
) -> str:
    string = ""
    for name, hel in name_hel_list:
        string += name
        if use_helicity:
            if make_parity_partner:
                string += "_" + str(-1 * hel)
            else:
                string += "_" + str(hel)
        string += "+"
    return string[:-1]


def _get_name_hel_list(
    graph: StateTransitionGraph[ParticleWithSpin], edge_ids: List[int]
) -> List[Tuple[str, float]]:
    name_hel_list = []
    for i in edge_ids:
        temp_hel = graph.get_edge_props(i)[1]
        # remove .0
        if temp_hel % 1 == 0:
            temp_hel = int(temp_hel)
        name_hel_list.append((graph.get_edge_props(i)[0].name, temp_hel))

    # in order to ensure correct naming of amplitude coefficients the list has
    # to be sorted by name. The same coefficient names have to be created for
    # two graphs that only differ from a kinematic standpoint
    # (swapped external edges)
    return sorted(name_hel_list, key=lambda entry: entry[0])


class _HelicityAmplitudeNameGenerator:
    """Parameter name generator for the helicity formalism."""

    def __init__(self) -> None:
        self.parity_partner_coefficient_mapping: Dict[str, str] = {}

    def _generate_amplitude_coefficient_couple(
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> Tuple[str, str, str]:
        (in_hel_info, out_hel_info) = self._retrieve_helicity_info(
            graph, node_id
        )
        par_name_suffix = self.generate_amplitude_coefficient_name(
            graph, node_id
        )

        pp_par_name_suffix = (
            _generate_particles_string(in_hel_info, False)
            + "_to_"
            + _generate_particles_string(
                out_hel_info, make_parity_partner=True
            )
        )

        priority_name_suffix = par_name_suffix
        if out_hel_info[0][1] < 0 or (
            out_hel_info[0][1] == 0 and out_hel_info[1][1] < 0
        ):
            priority_name_suffix = pp_par_name_suffix

        return (par_name_suffix, pp_par_name_suffix, priority_name_suffix)

    def register_amplitude_coefficient_name(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> None:
        for node_id in graph.nodes:
            (
                coefficient_suffix,
                parity_partner_coefficient_suffix,
                priority_partner_coefficient_suffix,
            ) = self._generate_amplitude_coefficient_couple(graph, node_id)

            if graph.get_node_props(node_id).parity_prefactor is None:
                continue

            if (
                coefficient_suffix
                not in self.parity_partner_coefficient_mapping
            ):
                if (
                    parity_partner_coefficient_suffix
                    in self.parity_partner_coefficient_mapping
                ):
                    if (
                        parity_partner_coefficient_suffix
                        == priority_partner_coefficient_suffix
                    ):
                        self.parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = parity_partner_coefficient_suffix
                    else:
                        self.parity_partner_coefficient_mapping[
                            parity_partner_coefficient_suffix
                        ] = coefficient_suffix
                        self.parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = coefficient_suffix

                else:
                    # if neither this coefficient nor its partner are registered just add it
                    self.parity_partner_coefficient_mapping[
                        coefficient_suffix
                    ] = coefficient_suffix

    def generate_unique_amplitude_name(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        node_id: Optional[int] = None,
    ) -> str:
        """Generates a unique name for the amplitude corresponding.

        That is, corresponging to the given :class:`StateTransitionGraph`. If
        ``node_id`` is given, it generates a unique name for the partial
        amplitude corresponding to the interaction node of the given
        :class:`StateTransitionGraph`.
        """
        name = ""
        if isinstance(node_id, int):
            nodelist = frozenset({node_id})
        else:
            nodelist = graph.nodes
        for node in nodelist:
            (in_hel_info, out_hel_info) = self._retrieve_helicity_info(
                graph, node
            )

            name += (
                _generate_particles_string(in_hel_info)
                + "_to_"
                + _generate_particles_string(out_hel_info)
                + ";"
            )
        return name

    @staticmethod
    def _retrieve_helicity_info(
        graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        in_edges = graph.get_edge_ids_ingoing_to_node(node_id)
        out_edges = graph.get_edge_ids_outgoing_from_node(node_id)

        in_names_hel_list = _get_name_hel_list(graph, in_edges)
        out_names_hel_list = _get_name_hel_list(graph, out_edges)

        return (in_names_hel_list, out_names_hel_list)

    def generate_amplitude_coefficient_name(
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> str:
        """Generate partial amplitude coefficient name suffix."""
        in_hel_info, out_hel_info = self._retrieve_helicity_info(
            graph, node_id
        )
        return (
            _generate_particles_string(in_hel_info, False)
            + "_to_"
            + _generate_particles_string(out_hel_info)
        )

    def generate_sequential_amplitude_suffix(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> str:
        """Generate unique suffix for a sequential amplitude graph."""
        output_suffix = ""
        for node_id in graph.nodes:
            suffix = self.generate_amplitude_coefficient_name(graph, node_id)
            if suffix in self.parity_partner_coefficient_mapping:
                suffix = self.parity_partner_coefficient_mapping[suffix]
            output_suffix += suffix + ";"
        return output_suffix


class HelicityAmplitudeGenerator:
    """Amplitude model generator for the helicity formalism."""

    def __init__(
        self,
        top_node_no_dynamics: bool = True,
    ) -> None:
        self.top_node_no_dynamics = top_node_no_dynamics
        self.name_generator = _HelicityAmplitudeNameGenerator()
        self.particles: Optional[ParticleCollection] = None
        self.kinematics: Optional[Kinematics] = None
        self.dynamics: Optional[ParticleDynamics] = None
        self.intensities: Optional[IntensityNode] = None
        self.fit_parameters: FitParameters = FitParameters()

    def generate(self, reaction_result: Result) -> AmplitudeModel:
        graphs = reaction_result.solutions
        if len(graphs) < 1:
            raise ValueError(
                f"At least one {StateTransitionGraph.__name__} required to"
                " genenerate an amplitude model!"
            )

        get_initial_state = reaction_result.get_initial_state()
        if len(get_initial_state) != 1:
            raise ValueError(
                "Helicity amplitude model requires exactly one initial state"
            )
        initial_state = get_initial_state[0].name

        self.particles = _generate_particle_collection(graphs)
        self.kinematics = _generate_kinematics(reaction_result, self.particles)
        self.dynamics = ParticleDynamics(self.particles, self.fit_parameters)
        if self.top_node_no_dynamics:
            self.dynamics.set_non_dynamic(initial_state)
        self.intensities = self.__generate_intensities(graphs)

        return AmplitudeModel(
            particles=self.particles,
            kinematics=self.kinematics,
            parameters=self.fit_parameters,
            intensity=self.intensities,
            dynamics=self.dynamics,
        )

    def __generate_intensities(
        self, graphs: List[StateTransitionGraph[ParticleWithSpin]]
    ) -> IntensityNode:
        graph_groups = _group_graphs_same_initial_and_final(graphs)
        logging.debug("There are %d graph groups", len(graph_groups))

        self.__create_parameter_couplings(graph_groups)
        incoherent_intensity = IncoherentIntensity()
        for graph_group in graph_groups:
            coherent_intensity = self.__generate_coherent_intensity(
                graph_group
            )
            incoherent_intensity.intensities.append(coherent_intensity)
        if len(incoherent_intensity.intensities) == 0:
            raise ValueError("List of incoherent intensities cannot be empty")
        if len(incoherent_intensity.intensities) == 1:
            return incoherent_intensity.intensities[0]
        return incoherent_intensity

    def __create_parameter_couplings(
        self, graph_groups: List[List[StateTransitionGraph[ParticleWithSpin]]]
    ) -> None:
        for graph_group in graph_groups:
            for graph in graph_group:
                self.name_generator.register_amplitude_coefficient_name(graph)

    def __generate_coherent_intensity(
        self,
        graph_group: List[StateTransitionGraph[ParticleWithSpin]],
    ) -> CoherentIntensity:
        coherent_amp_name = "coherent_" + _get_graph_group_unique_label(
            graph_group
        )
        coherent_intensity = CoherentIntensity(coherent_amp_name)
        for graph in graph_group:
            sequential_graphs = (
                perform_external_edge_identical_particle_combinatorics(graph)
            )
            for seq_graph in sequential_graphs:
                coherent_intensity.amplitudes.append(
                    self.__generate_sequential_decay(seq_graph)
                )
        return coherent_intensity

    def __generate_sequential_decay(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> AmplitudeNode:
        partial_decays: List[AmplitudeNode] = [
            self._generate_partial_decay(graph, node_id)
            for node_id in graph.nodes
        ]
        sequential_amplitudes = SequentialAmplitude(partial_decays)

        amp_name = self.name_generator.generate_unique_amplitude_name(graph)
        magnitude, phase = self.__generate_amplitude_coefficient(graph)
        prefactor = self.__generate_amplitude_prefactor(graph)
        return CoefficientAmplitude(
            component=amp_name,
            magnitude=magnitude,
            phase=phase,
            amplitude=sequential_amplitudes,
            prefactor=prefactor,
        )

    def _generate_partial_decay(
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> DecayNode:
        def create_helicity_particle(
            edge_props: ParticleWithSpin,
        ) -> HelicityParticle:
            if self.particles is None:
                raise ValueError(
                    f"{ParticleCollection.__name__} not yet initialized!"
                )
            particle = edge_props[0]
            helicity = edge_props[1]
            return HelicityParticle(particle, helicity)

        decay_products: List[DecayProduct] = list()
        for out_edge_id in graph.get_edge_ids_outgoing_from_node(node_id):
            edge_props = graph.get_edge_props(out_edge_id)
            helicity_particle = create_helicity_particle(edge_props)
            final_state_ids = _determine_attached_final_state(
                graph, out_edge_id
            )
            decay_products.append(
                DecayProduct(
                    helicity_particle.particle,
                    helicity_particle.helicity,
                    final_state_ids,
                )
            )

        in_edge_ids = graph.get_edge_ids_ingoing_to_node(node_id)
        if len(in_edge_ids) != 1:
            raise ValueError("This node does not represent a two body decay!")
        ingoing_edge_id = in_edge_ids[0]
        edge_props = graph.get_edge_props(ingoing_edge_id)
        helicity_particle = create_helicity_particle(edge_props)
        helicity_decay = HelicityDecay(helicity_particle, decay_products)

        recoil_edge_id = _get_recoil_edge(graph, ingoing_edge_id)
        if recoil_edge_id is not None:
            helicity_decay.recoil_system = RecoilSystem(
                _determine_attached_final_state(graph, recoil_edge_id)
            )
            parent_recoil_edge_id = _get_parent_recoil_edge(
                graph, ingoing_edge_id
            )
            if parent_recoil_edge_id is not None:
                helicity_decay.recoil_system.parent_recoil_final_state = (
                    _determine_attached_final_state(
                        graph, parent_recoil_edge_id
                    )
                )

        return helicity_decay

    def __generate_amplitude_coefficient(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> Tuple[FitParameter, FitParameter]:
        """Generate coefficient info for a sequential amplitude graph.

        Generally, each partial amplitude of a sequential amplitude graph
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        """
        seq_par_suffix = (
            self.name_generator.generate_sequential_amplitude_suffix(graph)
        )
        magnitude = self.__register_parameter(
            name=f"Magnitude_{seq_par_suffix}", value=1.0, fix=False
        )
        phase = self.__register_parameter(
            name=f"Phase_{seq_par_suffix}", value=0.0, fix=False
        )
        return magnitude, phase

    def __generate_amplitude_prefactor(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> Optional[float]:
        prefactor = _get_prefactor(graph)
        if prefactor != 1.0:
            for node_id in graph.nodes:
                raw_suffix = (
                    self.name_generator.generate_amplitude_coefficient_name(
                        graph, node_id
                    )
                )
                if (
                    raw_suffix
                    in self.name_generator.parity_partner_coefficient_mapping
                ):
                    coefficient_suffix = (
                        self.name_generator.parity_partner_coefficient_mapping[
                            raw_suffix
                        ]
                    )
                    if coefficient_suffix != raw_suffix:
                        return prefactor
        return None

    def __register_parameter(
        self, name: str, value: float, fix: bool = False
    ) -> FitParameter:
        if name in self.fit_parameters:
            return self.fit_parameters[name]
        parameter = FitParameter(name=name, value=value, is_fixed=fix)
        self.fit_parameters.add(parameter)
        return parameter


def __validate_float_type(
    interaction_property: Optional[Union[Spin, float]]
) -> Optional[float]:
    if interaction_property is not None and not isinstance(
        interaction_property, (float, int)
    ):
        raise TypeError(
            f"{interaction_property.__class__.__name__} is not of type {float.__name__}"
        )
    return interaction_property
