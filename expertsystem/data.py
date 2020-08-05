"""A collection of data containers."""

__all__ = [  # fix order in API
    "ParticleCollection",
    "Particle",
    "ComplexEnergyState",
    "QuantumState",
    "Parity",
    "Spin",
    "ComplexEnergy",
]


from collections import abc
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    TypeVar,
    Union,
    ValuesView,
)


class Parity:
    """Safe, immutable data container for parity."""

    def __init__(self, value: Union[float, int, str]) -> None:
        value = float(value)
        if value not in [-1.0, +1.0]:
            raise ValueError("Parity can only be +1 or -1")
        self.__value: int = int(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parity):
            return self.__value == other.value
        return self.__value == other

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({"+1" if self.__value > 0 else "-1"})'
        )

    @property
    def value(self) -> int:
        return self.__value


class Spin(abc.Hashable):
    """Safe, immutable data container for spin **with projection**."""

    def __init__(self, magnitude: float, projection: float) -> None:
        magnitude = float(magnitude)
        projection = float(projection)
        if abs(projection) > magnitude:
            raise ValueError(
                "Spin projection cannot be larger than its magnitude:\n"
                f"  {projection} > {magnitude}"
            )
        if projection == -0.0:
            projection = 0.0
        self.__magnitude = magnitude
        self.__projection = projection

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Spin):
            return (
                self.__magnitude == other.magnitude
                and self.__projection == other.projection
            )
        return self.__magnitude == other

    def __float__(self) -> float:
        return self.__magnitude

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}{self.__magnitude, self.__projection}"
        )

    @property
    def magnitude(self) -> float:
        return self.__magnitude

    @property
    def projection(self) -> float:
        return self.__projection

    def __hash__(self) -> int:
        return hash(repr(self))


_T = TypeVar("_T", float, Spin)


@dataclass(frozen=True)
class QuantumState(
    Generic[_T]
):  # pylint: disable=too-many-instance-attributes
    """Set of quantum numbers with a **generic type spin**.

    This is to make spin projection required in `.QuantumState` and unavailable
    in `.Particle`.
    """

    spin: _T
    charge: int = 0
    isospin: Optional[Spin] = None
    strangeness: int = 0
    charmness: int = 0
    bottomness: int = 0
    topness: int = 0
    baryon_number: int = 0
    electron_lepton_number: int = 0
    muon_lepton_number: int = 0
    tau_lepton_number: int = 0
    parity: Optional[Parity] = None
    c_parity: Optional[Parity] = None
    g_parity: Optional[Parity] = None


class ComplexEnergy:
    """Defines a complex valued energy.

    Resembles a position (pole) in the complex energy plane.
    """

    def __init__(self, energy: complex):
        self.__energy = complex(energy)

    @property
    def complex_energy(self) -> complex:
        return self.__energy

    @property
    def mass(self) -> float:
        return self.__energy.real

    @property
    def width(self) -> float:
        return self.__energy.imag

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ComplexEnergy):
            return self.complex_energy == other.complex_energy
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.complex_energy}"


class ComplexEnergyState(ComplexEnergy):
    """Pole in the complex energy plane, with quantum numbers."""

    def __init__(self, energy: complex, state: QuantumState[Spin]):
        super().__init__(energy)
        self.state: QuantumState[Spin] = state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.complex_energy, self.state}"


class Particle(ComplexEnergy):
    """Immutable container of data defining a physical particle.

    Can **only** contain info that the `PDG <http://pdg.lbl.gov/>`_ would list.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        pid: int,
        state: QuantumState[float],
        mass: float,
        width: float = 0.0,
    ):
        super().__init__(complex(mass, width))
        self.__name: str = name
        self.__pid: int = pid
        self.state: QuantumState[float] = state

    @property
    def name(self) -> str:
        return self.__name

    @property
    def pid(self) -> int:
        return self.__pid

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Particle):
            return (
                self.name == other.name
                and self.pid == other.pid
                and super().__eq__(other)
                and self.state == other.state
            )
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name, self.pid, self.state, self.mass, self.width}"


class ParticleCollection(abc.Mapping):
    """Safe, `dict`-like collection of `.Particle` instances."""

    def __init__(
        self, particles: Optional[Dict[str, Particle]] = None
    ) -> None:
        self.__particles: Dict[str, Particle] = dict()
        if particles is not None:
            if isinstance(particles, dict):
                self.__particles.update(particles)

    def __getitem__(self, particle_name: str) -> Particle:
        return self.__particles[particle_name]

    def __contains__(self, particle_name: object) -> bool:
        return particle_name in self.__particles

    def __iter__(self) -> Iterator[str]:
        return self.__particles.__iter__()

    def __len__(self) -> int:
        return len(self.__particles)

    def __iadd__(
        self, other: Union[Particle, "ParticleCollection"]
    ) -> "ParticleCollection":
        if isinstance(other, Particle):
            self.add(other)
        elif isinstance(other, ParticleCollection):
            self.merge(other)
        else:
            raise NotImplementedError
        return self

    def __repr__(self) -> str:
        return str(self.__particles)

    def add(self, particle: Particle) -> None:
        self.__particles[particle.name] = particle

    def find(self, search_term: Union[int, str]) -> Particle:
        """Search for a particle by either name (`str`) or PID (`int`)."""
        if isinstance(search_term, str):
            particle_name = search_term
            return self.__particles[particle_name]
        if isinstance(search_term, int):
            pid = search_term
            search_results = [
                particle for particle in self.values() if particle.pid == pid
            ]
            if len(search_results) == 0:
                raise LookupError(f"Could not find particle with PID {pid}")
            if len(search_results) > 1:
                error_message = f"Found multiple results for PID {pid}!:"
                for particle in search_results:
                    error_message += f"\n  - {particle.name}"
                raise LookupError(error_message)
            return search_results[0]
        raise NotImplementedError(
            f"Cannot search for a search term of type {type(search_term)}"
        )

    def find_subset(
        self, search_term: Union[int, str]
    ) -> "ParticleCollection":
        """Perform a 'fuzzy' search for a particle by name or PID.

        Like `~.ParticleCollection.find`, but returns several results in the
        form of a new `.ParticleCollection`.
        """
        if isinstance(search_term, str):
            search_results = {
                particle.name: particle
                for particle in self.values()
                if search_term in particle.name
            }
            return ParticleCollection(search_results)
        if isinstance(search_term, int):
            pid = search_term
            output = ParticleCollection()
            particle = self.find(pid)
            output.add(particle)
            return output
        raise NotImplementedError(
            f"Cannot search for a search term of type {type(search_term)}"
        )

    def items(self) -> ItemsView[str, Particle]:
        return self.__particles.items()

    def keys(self) -> KeysView[str]:
        return self.__particles.keys()

    def values(self) -> ValuesView[Particle]:
        return self.__particles.values()

    def merge(self, other: "ParticleCollection") -> None:
        for particle in other.values():
            self.add(particle)
