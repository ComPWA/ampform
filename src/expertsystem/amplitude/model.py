"""Data objects for an `~expertsystem.amplitude` model."""


from abc import ABC
from collections import abc
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

import attr

from expertsystem.particle import Particle, ParticleCollection


@attr.s
class FitParameter:
    name: str = attr.ib()
    value: float = attr.ib(default=0.0)
    is_fixed: bool = attr.ib(default=False)


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


@attr.s
class BlattWeisskopf(FormFactor):
    meson_radius: FitParameter = attr.ib()


@attr.s
class NonDynamic(Dynamics):
    form_factor: BlattWeisskopf = attr.ib()


@attr.s
class RelativisticBreitWigner(Dynamics):
    pole_position: FitParameter = attr.ib()
    pole_width: FitParameter = attr.ib()
    form_factor: BlattWeisskopf = attr.ib()


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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Kinematics):
            return (
                self.initial_state == other.initial_state
                and self.final_state == other.final_state
                and self.kinematics_type == other.kinematics_type
            )
        raise NotImplementedError

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

    def add_initial_state(self, state_id: int, particle_name: str) -> None:
        _assert_arg_type(particle_name, str)
        _assert_arg_type(state_id, int)
        if state_id in self.__initial_state:
            raise ValueError(f"Initial state ID {state_id} already exists")
        particle = self.__particles[particle_name]
        self.__initial_state[state_id] = particle

    def add_final_state(self, state_id: int, particle_name: str) -> None:
        _assert_arg_type(particle_name, str)
        _assert_arg_type(state_id, int)
        if state_id in self.__final_state:
            raise ValueError(f"Initial state ID {state_id} already exists")
        particle = self.__particles[particle_name]
        self.__final_state[state_id] = particle


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
