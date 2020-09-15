"""A collection of data containers."""

__all__ = [  # fix order in API
    "ParticleCollection",
    "Particle",
    "Parity",
    "Spin",
    "create_antiparticle",
    "create_particle",
    "GellmannNishijima",
]


import logging
from collections import abc
from dataclasses import dataclass
from typing import (
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    Union,
    ValuesView,
)


class Parity:
    """Safe, immutable data container for parity."""

    def __init__(self, value: Union[float, int, str]) -> None:
        value = float(value)
        if value not in [-1.0, +1.0]:
            raise ValueError(f"Parity can only be +1 or -1, not {value}")
        self.__value: int = int(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parity):
            return self.__value == other.value
        return self.__value == other

    def __int__(self) -> int:
        return self.value

    def __neg__(self) -> "Parity":
        return Parity(-self.value)

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
        if magnitude % 0.5 != 0.0:
            raise ValueError(
                f"Spin magnitude {magnitude} has to be a multitude of 0.5"
            )
        if abs(projection) > magnitude:
            raise ValueError(
                "Spin projection cannot be larger than its magnitude:\n"
                f"  {projection} > {magnitude}"
            )
        if not (projection - magnitude).is_integer():
            raise ValueError(
                f"{self.__class__.__name__}{magnitude, projection}: "
                "(projection - magnitude) should be integer! "
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

    def __neg__(self) -> "Spin":
        return Spin(self.magnitude, -self.projection)

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


@dataclass(frozen=True)
class Particle:  # pylint: disable=too-many-instance-attributes
    """Immutable container of data defining a physical particle.

    A Particle is defined by the minimum set of the quantum numbers that every
    possible instances of that particle have in common (the "static" quantum
    numbers of the particle). A "non-static" quantum number is the spin
    projection. Hence Particles do NOT contain spin projection information.
    """

    name: str
    pid: int
    spin: float
    mass: float
    width: float = 0.0
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

    @property
    def energy(self) -> complex:
        return complex(self.mass, self.width)

    def __post_init__(self) -> None:
        if (
            self.isospin is not None
            and GellmannNishijima.compute_charge(self) != self.charge
        ):
            raise ValueError(
                f"Cannot construct particle {self.name} because its quantum numbers"
                " don't agree with the Gell-Mann–Nishijima formula:\n"
                f"  Q[{self.charge}] != "
                f"Iz[{self.isospin.projection}] + 1/2 "
                f"(B[{self.baryon_number}] + "
                f" S[{self.strangeness}] + "
                f" C[{self.charmness}] +"
                f" B'[{self.bottomness}] +"
                f" T[{self.strangeness}]"
                ")"
            )


class GellmannNishijima:
    r"""Collection of conversion methods using Gell-Mann–Nishijima.

    The methods in this class use the `Gell-Mann–Nishijima formula
    <https://en.wikipedia.org/wiki/Gell-Mann%E2%80%93Nishijima_formula>`_:

    .. math::
        Q = I_3 + \frac{1}{2}(B+S+C+B'+T)

    where
    :math:`Q` is charge (computed),
    :math:`I_3` is `.Spin.projection` of `~.Particle.isospin`,
    :math:`B` is `~.Particle.baryon_number`,
    :math:`S` is `~.Particle.strangeness`,
    :math:`C` is `~.Particle.charmness`,
    :math:`B'` is `~.Particle.bottomness`, and
    :math:`T` is `~.Particle.topness`.
    """

    @staticmethod
    def compute_charge(state: Particle) -> Optional[float]:
        """Compute charge using the Gell-Mann–Nishijima formula.

        If isospin is not `None`, returns the value :math:`Q`: computed with
        the `Gell-Mann–Nishijima formula <.GellmannNishijima>`.
        """
        if state.isospin is None:
            return None
        if state.isospin.projection is None:
            raise ValueError(
                "Isospin projection must be defined if a magnitude is defined!"
            )
        computed_charge = state.isospin.projection + 0.5 * (
            state.baryon_number
            + state.strangeness
            + state.charmness
            + state.bottomness
            + state.topness
        )
        return computed_charge

    @staticmethod
    def compute_isospin_projection(  # pylint: disable=too-many-arguments
        charge: float,
        baryon_number: float,
        strangeness: float,
        charmness: float,
        bottomness: float,
        topness: float,
    ) -> float:
        """Compute isospin projection using the Gell-Mann–Nishijima formula.

        See `~.GellmannNishijima.compute_charge`, but then computed for
        :math:`I_3`.
        """
        return charge - 0.5 * (
            baryon_number + strangeness + charmness + bottomness + topness
        )


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
        if particle.name in self.__particles:
            logging.warning(
                f"{self.__class__.__name__}: Overwriting particle {particle.name}"
            )
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


def create_particle(  # pylint: disable=too-many-arguments,too-many-locals
    template_particle: Particle,
    name: Optional[str] = None,
    pid: Optional[int] = None,
    mass: Optional[float] = None,
    width: Optional[float] = None,
    charge: Optional[int] = None,
    spin: Optional[float] = None,
    isospin: Optional[Spin] = None,
    strangeness: Optional[int] = None,
    charmness: Optional[int] = None,
    bottomness: Optional[int] = None,
    topness: Optional[int] = None,
    baryon_number: Optional[int] = None,
    electron_lepton_number: Optional[int] = None,
    muon_lepton_number: Optional[int] = None,
    tau_lepton_number: Optional[int] = None,
    parity: Optional[int] = None,
    c_parity: Optional[int] = None,
    g_parity: Optional[int] = None,
) -> Particle:
    return Particle(
        name=name if name else template_particle.name,
        pid=pid if pid else template_particle.pid,
        mass=mass if mass else template_particle.mass,
        width=width if width else template_particle.width,
        spin=spin if spin else template_particle.spin,
        charge=charge if charge else template_particle.charge,
        strangeness=strangeness
        if strangeness
        else template_particle.strangeness,
        charmness=charmness if charmness else template_particle.charmness,
        bottomness=bottomness if bottomness else template_particle.bottomness,
        topness=topness if topness else template_particle.topness,
        baryon_number=baryon_number
        if baryon_number
        else template_particle.baryon_number,
        electron_lepton_number=electron_lepton_number
        if electron_lepton_number
        else template_particle.electron_lepton_number,
        muon_lepton_number=muon_lepton_number
        if muon_lepton_number
        else template_particle.muon_lepton_number,
        tau_lepton_number=tau_lepton_number
        if tau_lepton_number
        else template_particle.tau_lepton_number,
        isospin=template_particle.isospin
        if isospin is None
        else template_particle.isospin,
        parity=template_particle.parity if parity is None else Parity(parity),
        c_parity=template_particle.c_parity
        if c_parity is None
        else Parity(c_parity),
        g_parity=template_particle.g_parity
        if g_parity is None
        else Parity(g_parity),
    )


def create_antiparticle(
    template_particle: Particle, new_name: str = None
) -> Particle:
    isospin: Optional[Spin] = None
    if template_particle.isospin:
        isospin = -template_particle.isospin
    parity: Optional[Parity] = None
    if template_particle.parity is not None:
        if template_particle.spin.is_integer():
            parity = template_particle.parity
        else:
            parity = -template_particle.parity
    return Particle(
        name=new_name if new_name else "anti-" + template_particle.name,
        pid=-template_particle.pid,
        mass=template_particle.mass,
        width=template_particle.width,
        charge=-template_particle.charge,
        spin=template_particle.spin,
        isospin=isospin,
        strangeness=-template_particle.strangeness,
        charmness=-template_particle.charmness,
        bottomness=-template_particle.bottomness,
        topness=-template_particle.topness,
        baryon_number=-template_particle.baryon_number,
        electron_lepton_number=-template_particle.electron_lepton_number,
        muon_lepton_number=-template_particle.muon_lepton_number,
        tau_lepton_number=-template_particle.tau_lepton_number,
        parity=parity,
        c_parity=template_particle.c_parity,
        g_parity=template_particle.g_parity,
    )
