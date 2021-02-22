"""Data objects for an `~expertsystem.amplitude` model."""


from abc import ABC
from collections import abc
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set

import attr

from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction.quantum_numbers import ParticleWithSpin
from expertsystem.reaction.topology import StateTransitionGraph


@attr.s
class FitParameter:
    name: str = attr.ib()
    value: float = attr.ib()
    fix: bool = attr.ib(default=False)


class FitParameters(abc.Mapping):
    def __init__(
        self, parameters: Optional[Iterable[FitParameter]] = None
    ) -> None:
        self.__parameters: Dict[str, FitParameter] = dict()
        if parameters is not None:
            if not isinstance(parameters, abc.Iterable):
                raise ValueError(
                    f"Cannot construct a {self.__class__.__name__} "
                    f"from a {parameters.__class__.__name__}"
                )
            self.__parameters.update(
                {
                    par.name: par
                    for par in parameters
                    if isinstance(par, FitParameter)
                }
            )

    @property
    def parameter_names(self) -> Set[str]:
        return set(self.__parameters)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FitParameters):
            if self.parameter_names != other.parameter_names:
                return False
            for par_name in self.keys():
                if self[par_name] != other[par_name]:
                    return False
            return True
        raise NotImplementedError

    def __getitem__(self, name: str) -> FitParameter:
        return self.__parameters[name]

    def __iter__(self) -> Iterator[str]:
        return self.__parameters.__iter__()

    def __len__(self) -> int:
        return len(self.__parameters)

    def __repr__(self) -> str:
        output = f"{self.__class__.__name__}(["
        for parameter in sorted(
            self.__parameters.values(), key=lambda p: p.name
        ):
            output += f"\n    {parameter},"
        output += "\n])"
        return output

    def add(self, parameter: FitParameter) -> None:
        _assert_arg_type(parameter, FitParameter)
        if parameter.name in self.__parameters:
            raise KeyError(f'Parameter "{parameter.name}" already exists')
        self.__parameters[parameter.name] = parameter

    def filter(  # noqa: A003
        self, function: Callable[[FitParameter], bool]
    ) -> "FitParameters":
        """Search by `FitParameter` properties with a :code:`lambda` function."""
        return FitParameters(
            [parameter for parameter in self.values() if function(parameter)]
        )


class FormFactor(ABC):
    pass


@attr.s
class BlattWeisskopf(FormFactor):
    meson_radius: FitParameter = attr.ib()


@attr.s
class Dynamics:
    form_factor: Optional[FormFactor] = attr.ib(default=None)


class NonDynamic(Dynamics):
    pass


@attr.s
class RelativisticBreitWigner(Dynamics):
    pole_real: FitParameter = attr.ib(kw_only=True)
    pole_imag: FitParameter = attr.ib(kw_only=True)


class ParticleDynamics(abc.MutableMapping):
    """Assign dynamics to certain particles in a `.ParticleCollection`."""

    def __init__(
        self,
        particles: ParticleCollection,
        parameters: FitParameters,
    ) -> None:
        _assert_arg_type(particles, ParticleCollection)
        _assert_arg_type(parameters, FitParameters)
        self.__particles = particles
        self.__parameters = parameters
        self.__dynamics: Dict[str, Dynamics] = dict()

    def __delitem__(self, particle_name: str) -> None:
        del self.__dynamics[particle_name]

    def __getitem__(self, particle_name: str) -> Dynamics:
        return self.__dynamics[particle_name]

    def __setitem__(self, particle_name: str, dynamics: Dynamics) -> None:
        _assert_arg_type(dynamics, Dynamics)
        if particle_name not in self.__particles:
            raise KeyError(
                f'Particle "{particle_name}" not in {ParticleCollection.__name__}'
            )
        for field in attr.fields(dynamics.__class__):
            if field.type is FitParameter:
                parameter = getattr(dynamics, field.name)
                self.__register_parameter(parameter)
        self.__dynamics[particle_name] = dynamics

    def __iter__(self) -> Iterator[str]:
        return self.__dynamics.__iter__()

    def __len__(self) -> int:
        return len(self.__dynamics)

    @property
    def parameters(self) -> FitParameters:
        return self.__parameters

    def set_non_dynamic(self, particle_name: str) -> NonDynamic:
        dynamics = NonDynamic(
            form_factor=self.__create_form_factor(particle_name)
        )
        self[particle_name] = dynamics
        return dynamics

    def set_breit_wigner(
        self, particle_name: str, relativistic: bool = True
    ) -> RelativisticBreitWigner:
        if not relativistic:
            raise NotImplementedError
        particle = self.__particles[particle_name]
        pole_real = FitParameter(
            name=f"Position_{particle.name}", value=particle.mass
        )
        pole_imag = FitParameter(
            name=f"Width_{particle.name}", value=particle.width
        )
        self.__register_parameter(pole_real)
        self.__register_parameter(pole_imag)
        dynamics = RelativisticBreitWigner(
            pole_real=pole_real,
            pole_imag=pole_imag,
            form_factor=self.__create_form_factor(particle.name),
        )
        self[particle_name] = dynamics
        return dynamics

    def __create_form_factor(self, particle_name: str) -> BlattWeisskopf:
        meson_radius = FitParameter(
            name=f"MesonRadius_{particle_name}",
            value=1.0,
            fix=True,
        )
        self.__register_parameter(meson_radius)
        return BlattWeisskopf(meson_radius)

    def __register_parameter(self, parameter: FitParameter) -> None:
        if parameter.name in self.__parameters:
            if parameter is not self.__parameters[parameter.name]:
                raise ValueError(
                    f'Fit parameter "{parameter.name}" already exists'
                )
        else:
            self.__parameters.add(parameter)


class KinematicsType(Enum):
    HELICITY = auto()


def _determine_default_kinematics(
    kinematics_type: Optional[KinematicsType],
) -> KinematicsType:
    if kinematics_type is None:
        return KinematicsType.HELICITY
    return kinematics_type


@attr.s(frozen=True)
class Kinematics:
    initial_state: Dict[int, Particle] = attr.ib()
    final_state: Dict[int, Particle] = attr.ib()
    type: KinematicsType = attr.ib(  # noqa: A003
        default=KinematicsType.HELICITY,
        converter=_determine_default_kinematics,
    )

    def __attrs_post_init__(self) -> None:
        overlapping_ids = set(self.initial_state) & set(self.final_state)
        if len(overlapping_ids) > 0:
            raise ValueError(
                "Initial and final state have overlapping IDs",
                overlapping_ids,
            )

    @property
    def id_to_particle(self) -> Dict[int, Particle]:
        return {**self.initial_state, **self.final_state}

    @staticmethod
    def from_graph(
        graph: StateTransitionGraph[ParticleWithSpin],
        kinematics_type: Optional[KinematicsType] = None,
    ) -> "Kinematics":
        initial_state = dict()
        for state_id in graph.topology.incoming_edge_ids:
            initial_state[state_id] = graph.get_edge_props(state_id)[0]
        final_state = dict()
        for state_id in graph.topology.outgoing_edge_ids:
            final_state[state_id] = graph.get_edge_props(state_id)[0]
        return Kinematics(
            type=kinematics_type,
            initial_state=initial_state,
            final_state=final_state,
        )


class Node(ABC):
    pass


class AmplitudeNode(Node):
    pass


class DecayNode(AmplitudeNode):
    pass


class IntensityNode(Node):
    pass


@attr.s
class SequentialAmplitude(AmplitudeNode):
    amplitudes: List[AmplitudeNode] = attr.ib(factory=list)


@attr.s
class CoefficientAmplitude(AmplitudeNode):
    component: str = attr.ib()
    magnitude: FitParameter = attr.ib()
    phase: FitParameter = attr.ib()
    amplitude: AmplitudeNode = attr.ib()
    prefactor: Optional[float] = attr.ib(default=None)


@attr.s
class StrengthIntensity(IntensityNode):
    component: str = attr.ib()
    strength: FitParameter = attr.ib()
    intensity: IntensityNode = attr.ib()


@attr.s
class NormalizedIntensity(IntensityNode):
    intensity: IntensityNode = attr.ib()


@attr.s
class IncoherentIntensity(IntensityNode):
    intensities: List[IntensityNode] = attr.ib(factory=list)


@attr.s
class CoherentIntensity(IntensityNode):
    component: str = attr.ib()
    amplitudes: List[AmplitudeNode] = attr.ib(factory=list)


@attr.s
class HelicityParticle:
    particle: Particle = attr.ib()
    helicity: float = attr.ib()


@attr.s
class DecayProduct(HelicityParticle):
    final_state_ids: List[int] = attr.ib()


@attr.s
class RecoilSystem:
    recoil_final_state: List[int] = attr.ib()
    parent_recoil_final_state: Optional[List[int]] = attr.ib(default=None)


@attr.s
class ClebschGordan:
    J: float = attr.ib()  # pylint: disable=invalid-name
    M: float = attr.ib()  # pylint: disable=invalid-name
    j_1: float = attr.ib()
    m_1: float = attr.ib()
    j_2: float = attr.ib()
    m_2: float = attr.ib()


@attr.s
class HelicityDecay(DecayNode):
    decaying_particle: HelicityParticle = attr.ib()
    decay_products: List[DecayProduct] = attr.ib()
    recoil_system: Optional[RecoilSystem] = attr.ib(default=None)


@attr.s
class CanonicalDecay(DecayNode):
    decaying_particle: HelicityParticle = attr.ib()
    decay_products: List[DecayProduct] = attr.ib()
    l_s: ClebschGordan = attr.ib()
    s2s3: ClebschGordan = attr.ib()
    recoil_system: Optional[RecoilSystem] = attr.ib(default=None)


@attr.s
class AmplitudeModel:
    kinematics: Kinematics = attr.ib()
    particles: ParticleCollection = attr.ib()
    parameters: FitParameters = attr.ib()
    intensity: IntensityNode = attr.ib()
    dynamics: ParticleDynamics = attr.ib()


def _assert_arg_type(value: Any, value_type: type) -> None:
    if not isinstance(value, value_type):
        raise TypeError(
            f"Argument {type(value)} has to be of type {value_type}"
        )
