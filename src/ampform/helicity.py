"""Generate an amplitude model with the helicity formalism."""

import logging
import operator
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import attr
import sympy as sp
from attr.validators import instance_of
from qrules import ParticleCollection, Result
from qrules.combinatorics import (
    perform_external_edge_identical_particle_combinatorics,
)
from qrules.particle import Particle, ParticleWithSpin, Spin
from qrules.topology import StateTransitionGraph
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner
from sympy.printing.latex import LatexPrinter

from ._graph_info import (
    generate_particle_collection,
    get_angular_momentum,
    get_coupled_spin,
    get_prefactor,
    group_graphs_same_initial_and_final,
)
from .dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
    verify_signature,
)
from .kinematics import (
    HelicityAdapter,
    ReactionInfo,
    get_helicity_angle_label,
    get_invariant_mass_label,
)

ParameterValue = Union[float, complex, int]


@attr.s(frozen=True)
class State:
    particle: Particle = attr.ib(
        validator=attr.validators.instance_of(Particle)
    )
    spin_projection: float = attr.ib(converter=float)


@attr.s(frozen=True, auto_attribs=True)
class _EdgeWithState:
    edge_id: int
    state: State

    @classmethod
    def from_graph(
        cls, graph: StateTransitionGraph, edge_id: int
    ) -> "_EdgeWithState":
        particle, spin_projection = graph.get_edge_props(edge_id)
        return cls(
            edge_id=edge_id,
            state=State(
                particle=particle,
                spin_projection=spin_projection,
            ),
        )


@attr.s(frozen=True, auto_attribs=True)
class _TwoBodyDecay:
    parent: _EdgeWithState
    children: Tuple[_EdgeWithState, _EdgeWithState]

    @classmethod
    def from_graph(
        cls, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> "_TwoBodyDecay":
        topology = graph.topology
        in_edge_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        out_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(in_edge_ids) != 1 or len(out_edge_ids) != 2:
            raise ValueError(
                f"Node {node_id} does not represent a 1-to-2 body decay!"
            )
        ingoing_edge_id = next(iter(in_edge_ids))

        sorted_by_id = sorted(out_edge_ids)
        final__edge_ids = [
            i for i in sorted_by_id if i in topology.outgoing_edge_ids
        ]
        intermediate_edge_ids = [
            i for i in sorted_by_id if i in topology.intermediate_edge_ids
        ]
        sorted_by_ending = tuple(intermediate_edge_ids + final__edge_ids)
        out_edge_id1, out_edge_id2, *_ = tuple(sorted_by_ending)

        return cls(
            parent=_EdgeWithState.from_graph(graph, ingoing_edge_id),
            children=(
                _EdgeWithState.from_graph(graph, out_edge_id1),
                _EdgeWithState.from_graph(graph, out_edge_id2),
            ),
        )


@attr.s(frozen=True)
class HelicityModel:
    _expression: sp.Expr = attr.ib(
        validator=attr.validators.instance_of(sp.Expr)
    )
    _parameter_defaults: Dict[sp.Symbol, ParameterValue] = attr.ib(
        validator=attr.validators.instance_of(dict)
    )
    _components: Dict[str, sp.Expr] = attr.ib(
        validator=attr.validators.instance_of(dict)
    )
    _adapter: HelicityAdapter = attr.ib(
        validator=attr.validators.instance_of(HelicityAdapter)
    )
    particles: ParticleCollection = attr.ib(
        validator=instance_of(ParticleCollection)
    )

    @property
    def expression(self) -> sp.Expr:
        return self._expression

    @property
    def components(self) -> Dict[str, sp.Expr]:
        return self._components

    @property
    def parameter_defaults(self) -> Dict[sp.Symbol, ParameterValue]:
        return self._parameter_defaults

    @property
    def adapter(self) -> HelicityAdapter:
        return self._adapter


class _HelicityAmplitudeNameGenerator:
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
            + R" \to "
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
        for node_id in graph.topology.nodes:
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
            nodelist = graph.topology.nodes
        names: List[str] = []
        for node in nodelist:
            (in_hel_info, out_hel_info) = self._retrieve_helicity_info(
                graph, node
            )

            name = (
                _generate_particles_string(in_hel_info)
                + R" \to "
                + _generate_particles_string(out_hel_info)
            )
            names.append(name)
        return "; ".join(names)

    @staticmethod
    def _retrieve_helicity_info(
        graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> Tuple[List[ParticleWithSpin], List[ParticleWithSpin]]:
        in_edges = graph.topology.get_edge_ids_ingoing_to_node(node_id)
        out_edges = graph.topology.get_edge_ids_outgoing_from_node(node_id)

        in_names_hel_list = _get_helicity_particles(graph, in_edges)
        out_names_hel_list = _get_helicity_particles(graph, out_edges)

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
            + R" \to "
            + _generate_particles_string(out_hel_info)
        )

    def generate_sequential_amplitude_suffix(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> str:
        """Generate unique suffix for a sequential amplitude graph."""
        coefficient_names: List[str] = []
        for node_id in graph.topology.nodes:
            suffix = self.generate_amplitude_coefficient_name(graph, node_id)
            if suffix in self.parity_partner_coefficient_mapping:
                suffix = self.parity_partner_coefficient_mapping[suffix]
            coefficient_names.append(suffix)
        return "; ".join(coefficient_names)


class _CanonicalAmplitudeNameGenerator(_HelicityAmplitudeNameGenerator):
    def generate_unique_amplitude_name(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        node_id: Optional[int] = None,
    ) -> str:
        if isinstance(node_id, int):
            node_ids = frozenset({node_id})
        else:
            node_ids = graph.topology.nodes
        names: List[str] = []
        for node in node_ids:
            helicity_name = super().generate_unique_amplitude_name(graph, node)
            name = (
                helicity_name[:-1]
                + self._generate_clebsch_gordan_string(graph, node)
                + helicity_name[-1]
            )
            names.append(name)
        return "; ".join(names)

    @staticmethod
    def _generate_clebsch_gordan_string(
        graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> str:
        node_props = graph.get_node_props(node_id)
        ang_orb_mom = sp.Rational(get_angular_momentum(node_props).magnitude)
        spin = sp.Rational(get_coupled_spin(node_props).magnitude)
        return f",L={ang_orb_mom},S={spin}"


def _get_graph_group_unique_label(
    graph_group: List[StateTransitionGraph[ParticleWithSpin]],
) -> str:
    label = ""
    if graph_group:
        first_graph = next(iter(graph_group))
        ise = first_graph.topology.incoming_edge_ids
        fse = first_graph.topology.outgoing_edge_ids
        is_names = _get_helicity_particles(first_graph, ise)
        fs_names = _get_helicity_particles(first_graph, fse)
        label += (
            _generate_particles_string(is_names)
            + R" \to "
            + _generate_particles_string(fs_names)
        )
    return label


def _get_helicity_particles(
    graph: StateTransitionGraph[ParticleWithSpin], edge_ids: Iterable[int]
) -> List[ParticleWithSpin]:
    helicity_list: List[ParticleWithSpin] = []
    for i in edge_ids:
        particle, spin_projection = graph.get_edge_props(i)
        if isinstance(spin_projection, float) and spin_projection.is_integer():
            spin_projection = int(spin_projection)
        helicity_list.append((particle, spin_projection))

    # in order to ensure correct naming of amplitude coefficients the list has
    # to be sorted by name. The same coefficient names have to be created for
    # two graphs that only differ from a kinematic standpoint
    # (swapped external edges)
    return sorted(helicity_list, key=lambda entry: entry[0].name)


def _generate_particles_string(
    helicity_list: List[ParticleWithSpin],
    use_helicity: bool = True,
    make_parity_partner: bool = False,
) -> str:
    output_string = ""
    for particle, spin_projection in helicity_list:
        if particle.latex is not None:
            output_string += particle.latex
        else:
            output_string += particle.name
        if use_helicity:
            if make_parity_partner:
                helicity = -1 * spin_projection
            else:
                helicity = spin_projection
            if helicity > 0:
                helicity_str = f"+{helicity}"
            else:
                helicity_str = str(helicity)
            output_string += f"_{{{helicity_str}}}"
        output_string += " "
    return output_string[:-1]


def _generate_kinematic_variable_set(
    transition: StateTransitionGraph[ParticleWithSpin], node_id: int
) -> TwoBodyKinematicVariableSet:
    decay = _TwoBodyDecay.from_graph(transition, node_id)
    inv_mass, phi, theta = _generate_kinematic_variables(transition, node_id)
    child1_mass = sp.Symbol(
        get_invariant_mass_label(
            transition.topology, decay.children[0].edge_id
        ),
        real=True,
    )
    child2_mass = sp.Symbol(
        get_invariant_mass_label(
            transition.topology, decay.children[1].edge_id
        ),
        real=True,
    )
    return TwoBodyKinematicVariableSet(
        in_edge_inv_mass=inv_mass,
        out_edge_inv_mass1=child1_mass,
        out_edge_inv_mass2=child2_mass,
        helicity_theta=theta,
        helicity_phi=phi,
        angular_momentum=_extract_angular_momentum(
            transition,
            node_id,
        ),
    )


def _extract_angular_momentum(
    transition: StateTransitionGraph[ParticleWithSpin], node_id: int
) -> int:
    node_props = transition.get_node_props(node_id)
    if node_props.l_magnitude is not None:
        return node_props.l_magnitude

    edge_id = None
    if len(transition.topology.get_edge_ids_ingoing_to_node(node_id)) == 1:
        edge_id = tuple(
            transition.topology.get_edge_ids_ingoing_to_node(node_id)
        )[0]
    elif (
        len(transition.topology.get_edge_ids_outgoing_from_node(node_id)) == 1
    ):
        edge_id = tuple(
            transition.topology.get_edge_ids_outgoing_from_node(node_id)
        )[0]

    if edge_id is None:
        raise ValueError(
            f"StateTransitionGraph does not have one to two body structure"
            f" at node with id={node_id}"
        )
    spin_magnitude = transition.get_edge_props(edge_id)[0].spin

    if spin_magnitude.is_integer():
        return int(spin_magnitude)

    raise ValueError(
        f"Spin magnitude ({spin_magnitude}) of single particle state cannot be"
        f" used as the angular momentum as it is not integral!"
    )


def _generate_kinematic_variables(
    transition: StateTransitionGraph[ParticleWithSpin], node_id: int
) -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Generate symbol for invariant mass, phi angle, and theta angle."""
    decay = _TwoBodyDecay.from_graph(transition, node_id)
    phi_label, theta_label = get_helicity_angle_label(
        transition.topology, decay.children[0].edge_id
    )
    inv_mass_label = get_invariant_mass_label(
        transition.topology, decay.parent.edge_id
    )
    return (
        sp.Symbol(inv_mass_label, real=True),
        sp.Symbol(phi_label, real=True),
        sp.Symbol(theta_label, real=True),
    )


class HelicityAmplitudeBuilder:  # pylint: disable=too-many-instance-attributes
    """Amplitude model generator for the helicity formalism."""

    def __init__(self, reaction_result: Result) -> None:
        self.name_generator = _HelicityAmplitudeNameGenerator()
        self.__graphs = reaction_result.transitions
        self.__parameter_defaults: Dict[sp.Symbol, ParameterValue] = {}
        self.__components: Dict[str, sp.Expr] = {}
        self.__dynamics_choices: Dict[
            _TwoBodyDecay, ResonanceDynamicsBuilder
        ] = {}

        if len(self.__graphs) < 1:
            raise ValueError(
                f"At least one {StateTransitionGraph.__name__} required to"
                " genenerate an amplitude model!"
            )
        first_graph = next(iter(self.__graphs))
        reaction_info = ReactionInfo.from_graph(first_graph)
        self.__adapter = HelicityAdapter(reaction_info)
        for graph in self.__graphs:
            self.__adapter.register_transition(graph)
        self.__particles = generate_particle_collection(self.__graphs)

    def set_dynamics(
        self, particle_name: str, dynamics_builder: ResonanceDynamicsBuilder
    ) -> None:
        verify_signature(dynamics_builder, ResonanceDynamicsBuilder)
        for transition in self.__graphs:
            for node_id in transition.topology.nodes:
                decay = _TwoBodyDecay.from_graph(transition, node_id)
                decay_particle = decay.parent.state.particle
                if decay_particle.name == particle_name:
                    self.__dynamics_choices[decay] = dynamics_builder

    def generate(self) -> HelicityModel:
        self.__components = {}
        self.__parameter_defaults = {}
        return HelicityModel(
            expression=self.__generate_intensity(),
            components=self.__components,
            parameter_defaults=self.__parameter_defaults,
            adapter=self.__adapter,
            particles=self.__particles,
        )

    def __generate_intensity(self) -> sp.Expr:
        graph_groups = group_graphs_same_initial_and_final(self.__graphs)
        logging.debug("There are %d graph groups", len(graph_groups))

        self.__create_parameter_couplings(graph_groups)
        coherent_intensities = []
        for graph_group in graph_groups:
            coherent_intensities.append(
                self.__generate_coherent_intensity(graph_group)
            )
        if len(coherent_intensities) == 0:
            raise ValueError("List of coherent intensities cannot be empty")
        return sum(coherent_intensities)

    def __create_dynamics(
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> sp.Expr:
        decay = _TwoBodyDecay.from_graph(graph, node_id)
        if decay in self.__dynamics_choices:
            builder = self.__dynamics_choices[decay]
            variable_set = _generate_kinematic_variable_set(graph, node_id)
            expression, parameters = builder(
                decay.parent.state.particle, variable_set
            )
            for par, value in parameters.items():
                if par in self.__parameter_defaults:
                    previous_value = self.__parameter_defaults[par]
                    if value != previous_value:
                        logging.warning(
                            f"Default value for parameter {par.name}"
                            f" inconsistent {value} and {previous_value}"
                        )
                self.__parameter_defaults[par] = value

            return expression

        return 1

    def __create_parameter_couplings(
        self, graph_groups: List[List[StateTransitionGraph[ParticleWithSpin]]]
    ) -> None:
        for graph_group in graph_groups:
            for graph in graph_group:
                self.name_generator.register_amplitude_coefficient_name(graph)

    def __generate_coherent_intensity(
        self,
        graph_group: List[StateTransitionGraph[ParticleWithSpin]],
    ) -> sp.Expr:
        graph_group_label = _get_graph_group_unique_label(graph_group)
        expression: List[sp.Expr] = []
        for graph in graph_group:
            sequential_graphs = (
                perform_external_edge_identical_particle_combinatorics(graph)
            )
            for seq_graph in sequential_graphs:
                expression.append(self.__generate_sequential_decay(seq_graph))
        amplitude_sum = sum(expression)
        coh_intensity = abs(amplitude_sum) ** 2
        self.__components[fR"I[{graph_group_label}]"] = coh_intensity
        return coh_intensity

    def __generate_sequential_decay(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> sp.Expr:
        partial_decays: List[sp.Symbol] = [
            self._generate_partial_decay(graph, node_id)
            for node_id in graph.topology.nodes
        ]
        sequential_amplitudes = reduce(operator.mul, partial_decays)

        coefficient = self.__generate_amplitude_coefficient(graph)
        prefactor = self.__generate_amplitude_prefactor(graph)
        expression = coefficient * sequential_amplitudes
        if prefactor is not None:
            expression = prefactor * expression
        self.__components[
            f"A[{self.name_generator.generate_unique_amplitude_name(graph)}]"
        ] = expression
        return expression

    def _generate_partial_decay(  # pylint: disable=too-many-locals
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> sp.Expr:
        wigner_d = self._generate_wigner_d(graph, node_id)
        dynamics_symbol = self.__create_dynamics(graph, node_id)
        return wigner_d * dynamics_symbol

    @staticmethod
    def _generate_wigner_d(
        graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> sp.Symbol:
        decay = _TwoBodyDecay.from_graph(graph, node_id)
        _, phi, theta = _generate_kinematic_variables(graph, node_id)

        return Wigner.D(
            j=sp.nsimplify(decay.parent.state.particle.spin),
            m=sp.nsimplify(decay.parent.state.spin_projection),
            mp=sp.nsimplify(
                decay.children[0].state.spin_projection
                - decay.children[1].state.spin_projection
            ),
            alpha=-phi,
            beta=theta,
            gamma=0,
        )

    def __generate_amplitude_coefficient(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> sp.Symbol:
        """Generate coefficient parameter for a sequential amplitude graph.

        Generally, each partial amplitude of a sequential amplitude graph
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        """
        suffix = self.name_generator.generate_sequential_amplitude_suffix(
            graph
        )
        coefficient_symbol = sp.Symbol(f"C[{suffix}]")
        self.__parameter_defaults[coefficient_symbol] = complex(1, 0)
        return coefficient_symbol

    def __generate_amplitude_prefactor(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> Optional[float]:
        prefactor = get_prefactor(graph)
        if prefactor != 1.0:
            for node_id in graph.topology.nodes:
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


class CanonicalAmplitudeBuilder(HelicityAmplitudeBuilder):
    r"""Amplitude model generator for the canonical helicity formalism.

    This class defines a full amplitude in the canonical formalism, using the
    helicity formalism as a foundation. The key here is that we take the full
    helicity intensity as a template, and just exchange the helicity amplitudes
    :math:`F` as a sum of canonical amplitudes :math:`A`:

    .. math::

        F^J_{\lambda_1,\lambda_2} = \sum_{LS} \mathrm{norm}(A^J_{LS})C^2.

    Here, :math:`C` stands for `Clebsch-Gordan factor
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_.
    """

    def __init__(self, reaction_result: Result) -> None:
        super().__init__(reaction_result)
        self.name_generator = _CanonicalAmplitudeNameGenerator()

    def _generate_partial_decay(  # pylint: disable=too-many-locals
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> sp.Symbol:
        amplitude = super()._generate_partial_decay(graph, node_id)

        node_props = graph.get_node_props(node_id)
        ang_mom = get_angular_momentum(node_props)
        spin = get_coupled_spin(node_props)
        if ang_mom.projection != 0.0:
            raise ValueError(f"Projection of L is non-zero!: {ang_mom}")

        topology = graph.topology
        in_edge_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        out_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)

        in_edge_id = next(iter(in_edge_ids))
        particle, spin_projection = graph.get_edge_props(in_edge_id)
        parent_spin = Spin(
            particle.spin,
            spin_projection,
        )

        daughter_spins: List[Spin] = []
        for out_edge_id in out_edge_ids:
            particle, spin_projection = graph.get_edge_props(out_edge_id)
            daughter_spin = Spin(
                particle.spin,
                spin_projection,
            )
            if daughter_spin is not None and isinstance(daughter_spin, Spin):
                daughter_spins.append(daughter_spin)

        decay_particle_lambda = (
            daughter_spins[0].projection - daughter_spins[1].projection
        )

        cg_ls = CG(
            j1=sp.nsimplify(ang_mom.magnitude),
            m1=sp.nsimplify(ang_mom.projection),
            j2=sp.nsimplify(spin.magnitude),
            m2=sp.nsimplify(decay_particle_lambda),
            j3=sp.nsimplify(parent_spin.magnitude),
            m3=sp.nsimplify(decay_particle_lambda),
        )
        cg_ss = CG(
            j1=sp.nsimplify(daughter_spins[0].magnitude),
            m1=sp.nsimplify(daughter_spins[0].projection),
            j2=sp.nsimplify(daughter_spins[1].magnitude),
            m2=sp.nsimplify(-daughter_spins[1].projection),
            j3=sp.nsimplify(spin.magnitude),
            m3=sp.nsimplify(decay_particle_lambda),
        )
        return cg_ls * cg_ss * amplitude


# https://github.com/sympy/sympy/issues/21001
# pylint: disable=protected-access, unused-argument
def _latex_fix(self: Type[CG], printer: LatexPrinter, *args: Any) -> str:
    j3, m3, j1, m1, j2, m2 = map(
        printer._print,
        (self.j3, self.m3, self.j1, self.m1, self.j2, self.m2),
    )
    return f"{{C^{{{j3},{m3}}}_{{{j1},{m1},{j2},{m2}}}}}"


CG._latex = _latex_fix
