"""A collection of data containers."""

__all__ = [  # fix order in API
    "ParticleCollection",
    "Particle",
    "ComplexEnergyState",
    "QuantumState",
    "Parity",
    "Spin",
    "ComplexEnergy",
    "create_antiparticle",
    "create_particle",
    "GellmannNishijima",
]


import logging
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
        if (
            state.isospin is not None
            and GellmannNishijima.compute_charge(state) != state.charge
        ):
            raise ValueError(
                f"Cannot construct particle {name} because its quantum numbers"
                " don't agree with the Gell-Mann–Nishijima formula:\n"
                f"  Q[{state.charge}] != "
                f"Iz[{state.isospin.projection}] + 1/2 "
                f"(B[{state.baryon_number}] + "
                f" S[{state.strangeness}] + "
                f" C[{state.charmness}] +"
                f" B'[{state.bottomness}] +"
                f" T[{state.strangeness}]"
                ")"
            )
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


class GellmannNishijima:
    r"""Collection of conversion methods using Gell-Mann–Nishijima.

    The methods in this class use the `Gell-Mann–Nishijima formula
    <https://en.wikipedia.org/wiki/Gell-Mann%E2%80%93Nishijima_formula>`_:

    .. math::
        Q = I_3 + \frac{1}{2}(B+S+C+B'+T)

    where
    :math:`Q` is charge (computed),
    :math:`I_3` is `.Spin.projection`,
    :math:`B` is `~.QuantumState.baryon_number`,
    :math:`S` is `~.QuantumState.strangeness`,
    :math:`C` is `~.QuantumState.charmness`,
    :math:`B'` is `~.QuantumState.bottomness`, and
    :math:`T` is `~.QuantumState.topness`.
    """

    @staticmethod
    def compute_charge(state: QuantumState) -> Optional[float]:
        """Compute charge using the Gell-Mann–Nishijima formula.

        If isospin is not `None`, returns the value :math:`Q`: computed with
        the `Gell-Mann–Nishijima formula <.GellmannNishijima>`.
        """
        if state.isospin is None:
            return None
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
    state: Optional[QuantumState[float]] = None,
) -> Particle:
    if state is not None:
        new_state = state
    else:
        new_state = QuantumState[float](
            spin=spin if spin else template_particle.state.spin,
            charge=charge if charge else template_particle.state.charge,
            strangeness=strangeness
            if strangeness
            else template_particle.state.strangeness,
            charmness=charmness
            if charmness
            else template_particle.state.charmness,
            bottomness=bottomness
            if bottomness
            else template_particle.state.bottomness,
            topness=topness if topness else template_particle.state.topness,
            baryon_number=baryon_number
            if baryon_number
            else template_particle.state.baryon_number,
            electron_lepton_number=electron_lepton_number
            if electron_lepton_number
            else template_particle.state.electron_lepton_number,
            muon_lepton_number=muon_lepton_number
            if muon_lepton_number
            else template_particle.state.muon_lepton_number,
            tau_lepton_number=tau_lepton_number
            if tau_lepton_number
            else template_particle.state.tau_lepton_number,
            isospin=template_particle.state.isospin
            if isospin is None
            else template_particle.state.isospin,
            parity=template_particle.state.parity
            if parity is None
            else Parity(parity),
            c_parity=template_particle.state.c_parity
            if c_parity is None
            else Parity(c_parity),
            g_parity=template_particle.state.g_parity
            if g_parity is None
            else Parity(g_parity),
        )
    new_particle = Particle(
        name=name if name else template_particle.name,
        pid=pid if pid else template_particle.pid,
        mass=mass if mass else template_particle.mass,
        width=width if width else template_particle.width,
        state=new_state,
    )
    return new_particle


def create_antiparticle(
    template_particle: Particle, new_name: str = None
) -> Particle:
    isospin: Optional[Spin] = None
    if template_particle.state.isospin:
        isospin = -template_particle.state.isospin
    parity: Optional[Parity] = None
    if template_particle.state.parity is not None:
        if template_particle.state.spin.is_integer():
            parity = template_particle.state.parity
        else:
            parity = -template_particle.state.parity
    return Particle(
        name=new_name if new_name else "anti-" + template_particle.name,
        pid=-template_particle.pid,
        mass=template_particle.mass,
        width=template_particle.width,
        state=QuantumState[float](
            charge=-template_particle.state.charge,
            spin=template_particle.state.spin,
            isospin=isospin,
            strangeness=-template_particle.state.strangeness,
            charmness=-template_particle.state.charmness,
            bottomness=-template_particle.state.bottomness,
            topness=-template_particle.state.topness,
            baryon_number=-template_particle.state.baryon_number,
            electron_lepton_number=-template_particle.state.electron_lepton_number,
            muon_lepton_number=-template_particle.state.muon_lepton_number,
            tau_lepton_number=-template_particle.state.tau_lepton_number,
            parity=parity,
            c_parity=template_particle.state.c_parity,
            g_parity=template_particle.state.g_parity,
        ),
    )
