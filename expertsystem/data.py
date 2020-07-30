"""A collection of data containers."""

from collections import abc
from typing import (
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    Union,
    ValuesView,
)
from typing import NamedTuple


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
            f'<{self.__class__.__name__} {"+1" if self.__value > 0 else "-1"}>'
        )

    @property
    def value(self) -> int:
        return self.__value


class Spin(abc.Hashable):
    """Safe, immutable data container for (iso)spin."""

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
        return f"<{self.__class__.__name__} {self.__magnitude, self.__projection}>"

    @property
    def magnitude(self) -> float:
        return self.__magnitude

    @property
    def projection(self) -> float:
        return self.__projection

    def __hash__(self) -> int:
        return hash(repr(self))


class Particle(NamedTuple):
    """Immutable data container for particle info."""

    name: str
    pid: int
    charge: int
    spin: float
    mass: float
    strangeness: int = 0
    charmness: int = 0
    bottomness: int = 0
    topness: int = 0
    baryon_number: int = 0
    electron_number: int = 0
    muon_number: int = 0
    tau_number: int = 0
    width: Optional[float] = None
    isospin: Optional[Spin] = None
    parity: Optional[Parity] = None
    c_parity: Optional[Parity] = None
    g_parity: Optional[Parity] = None


class ParticleCollection(abc.Mapping):
    """Safe, `dict`-like collection of `.Particle` instances."""

    def __init__(self) -> None:
        self.__particles: Dict[str, Particle] = dict()

    def __getitem__(self, particle_name: str) -> Particle:
        return self.__particles[particle_name]

    def __contains__(self, particle_name: object) -> bool:
        return particle_name in self.__particles

    def __iter__(self) -> Iterator[str]:
        return self.__particles.__iter__()

    def __len__(self) -> int:
        return len(self.__particles)

    def __repr__(self) -> str:
        return str(self.__particles)

    def add(self, particle: Particle) -> None:
        self.__particles[particle.name] = particle

    def items(self) -> ItemsView[str, Particle]:
        return self.__particles.items()

    def keys(self) -> KeysView[str]:
        return self.__particles.keys()

    def values(self) -> ValuesView[Particle]:
        return self.__particles.values()
