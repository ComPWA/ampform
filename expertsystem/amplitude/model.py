"""Data objects for an `~expertsystem.amplitude` model."""


from abc import ABC
from collections import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
)

from expertsystem.data import Particle, ParticleCollection


@dataclass
class FitParameter:
    name: str
    value: float = 0.0
    is_fixed: bool = False


class FitParameters(abc.Mapping):
    def __init__(
        self, parameters: Optional[Iterable[FitParameter]] = None
    ) -> None:
        self.__parameters: Dict[str, FitParameter] = dict()
        if parameters is not None:
            if not isinstance(parameters, (list, set, tuple)):
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

    def __getitem__(self, name: str) -> FitParameter:
        return self.__parameters[name]

    def __iter__(self) -> Iterator[str]:
        return self.__parameters.__iter__()

    def __len__(self) -> int:
        return len(self.__parameters)

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


class Dynamics(ABC):
    pass


class FormFactor(ABC):
    pass


@dataclass
class BlattWeisskopf(FormFactor):
    meson_radius: FitParameter


@dataclass
class NonDynamic(Dynamics):
    form_factor: BlattWeisskopf


@dataclass
class RelativisticBreitWigner(Dynamics):
    pole_position: FitParameter
    pole_width: FitParameter
    form_factor: BlattWeisskopf


class ParticleDynamics(abc.Mapping):
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

    def __getitem__(self, particle_name: str) -> Dynamics:
        return self.__dynamics[particle_name]

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
        self.__set(particle_name, dynamics)
        return dynamics

    def set_breit_wigner(
        self, particle_name: str, relativistic: bool = True
    ) -> RelativisticBreitWigner:
        if not relativistic:
            raise NotImplementedError
        particle = self.__particles[particle_name]
        pole_position = self.__register_parameter(
            name=f"Position_{particle.name}",
            value=particle.mass,
            fix=False,
        )
        pole_width = self.__register_parameter(
            name=f"Width_{particle.name}",
            value=particle.width,
            fix=False,
        )
        dynamics = RelativisticBreitWigner(
            pole_position=pole_position,
            pole_width=pole_width,
            form_factor=self.__create_form_factor(particle.name),
        )
        self.__set(particle_name, dynamics)
        return dynamics

    def __create_form_factor(self, particle_name: str) -> BlattWeisskopf:
        meson_radius = self.__register_parameter(
            name=f"MesonRadius_{particle_name}",
            value=1.0,
            fix=True,
        )
        return BlattWeisskopf(meson_radius)

    def __set(self, particle_name: str, value: Dynamics) -> None:
        _assert_arg_type(value, Dynamics)
        if particle_name not in self.__particles:
            raise KeyError(
                f'Particle "{particle_name}" not in {ParticleCollection.__name__}'
            )
        self.__dynamics[particle_name] = value

    def __register_parameter(
        self, name: str, value: float, fix: bool = False
    ) -> FitParameter:
        if name in self.__parameters:
            return self.__parameters[name]
        parameter = FitParameter(name=name, value=value, is_fixed=fix)
        self.__parameters.add(parameter)
        return parameter


class KinematicsType(Enum):
    Helicity = auto()


class Kinematics:
    def __init__(
        self,
        particles: ParticleCollection,
        kinematics_type: KinematicsType = KinematicsType.Helicity,
    ) -> None:
        _assert_arg_type(particles, ParticleCollection)
        _assert_arg_type(kinematics_type, KinematicsType)
        self.__particles = particles
        self.__initial_state: Dict[int, Particle] = dict()
        self.__final_state: Dict[int, Particle] = dict()
        self.__kinematics_type: KinematicsType = kinematics_type

    @property
    def initial_state(self) -> Dict[int, Particle]:
        return self.__initial_state

    @property
    def final_state(self) -> Dict[int, Particle]:
        return self.__final_state

    @property
    def kinematics_type(self) -> KinematicsType:
        return self.__kinematics_type

    def set_reaction(
        self,
        initial_state: Sequence[str],
        final_state: Sequence[str],
        intermediate_states: int,
    ) -> None:
        ini_particles = [self.__particles[name] for name in initial_state]
        final_particles = [self.__particles[name] for name in final_state]
        self.__initial_state = dict(enumerate(ini_particles))
        self.__final_state = dict(
            enumerate(
                final_particles, start=len(initial_state) + intermediate_states
            )
        )

    def id_to_particle(self, state_id: int) -> Particle:
        particle = self.__initial_state.get(
            state_id, self.__final_state.get(state_id, None)
        )
        if particle is None:
            raise KeyError(f"Kinematics does not contain state ID {state_id}")
        return particle


class Node(ABC):
    pass


class AmplitudeNode(Node):
    pass


class DecayNode(AmplitudeNode):
    pass


class IntensityNode(Node):
    pass


@dataclass
class SequentialAmplitude(AmplitudeNode):
    amplitudes: List[AmplitudeNode] = field(default_factory=list)


@dataclass
class CoefficientAmplitude(AmplitudeNode):
    component: str
    magnitude: FitParameter
    phase: FitParameter
    amplitude: AmplitudeNode
    prefactor: Optional[float] = None


@dataclass
class StrengthIntensity(IntensityNode):
    component: str
    strength: FitParameter
    intensity: IntensityNode


@dataclass
class NormalizedIntensity(IntensityNode):
    intensity: IntensityNode


@dataclass
class IncoherentIntensity(IntensityNode):
    intensities: List[IntensityNode] = field(default_factory=list)


@dataclass
class CoherentIntensity(IntensityNode):
    component: str
    amplitudes: List[AmplitudeNode] = field(default_factory=list)


@dataclass
class HelicityParticle:
    particle: Particle
    helicity: float


@dataclass
class DecayProduct(HelicityParticle):
    final_state_ids: List[int]


@dataclass
class RecoilSystem:
    recoil_final_state: List[int]
    parent_recoil_final_state: Optional[List[int]] = None


@dataclass
class ClebschGordan:
    J: float  # pylint: disable=invalid-name
    M: float  # pylint: disable=invalid-name
    j_1: float
    m_1: float
    j_2: float
    m_2: float


@dataclass
class HelicityDecay(DecayNode):
    decaying_particle: HelicityParticle
    decay_products: List[DecayProduct]
    recoil_system: Optional[RecoilSystem] = None


@dataclass
class CanonicalDecay(DecayNode):
    decaying_particle: HelicityParticle
    decay_products: List[DecayProduct]
    l_s: ClebschGordan
    s2s3: ClebschGordan
    recoil_system: Optional[RecoilSystem] = None


@dataclass
class AmplitudeModel:
    kinematics: Kinematics
    particles: ParticleCollection
    parameters: FitParameters
    intensity: IntensityNode
    dynamics: ParticleDynamics


def _assert_arg_type(value: Any, value_type: type) -> None:
    if not isinstance(value, value_type):
        raise TypeError(
            f"Argument {type(value)} has to be of type {value_type}"
        )
