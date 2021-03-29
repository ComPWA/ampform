"""Definitions used internally for type hints and signatures.

The `expertsystem` is strictly typed (enforced through :doc:`mypy
<mypy:index>`). This module bundles structures and definitions that don't serve
as data containers but only as type hints. `.EdgeQuantumNumbers` and
`.NodeQuantumNumbers` are the main structures and serve as a bridge between the
:mod:`.reaction.particle` and the :mod:`.reaction` module.
"""

from decimal import Decimal
from fractions import Fraction
from functools import total_ordering
from typing import Any, Generator, NewType, Optional, Union

import attr
from attr.validators import instance_of


@total_ordering
@attr.s(frozen=True, repr=False, eq=False, hash=True)
class Parity:
    value: int = attr.ib(validator=instance_of(int))

    @value.validator
    def __check_plusminus(  # type: ignore  # pylint: disable=no-self-use,unused-argument
        self, _: attr.Attribute, value: int
    ) -> None:
        if value not in [-1, +1]:
            raise ValueError(f"Parity can only be +1 or -1, not {value}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parity):
            return self.value == other.value
        return self.value == other

    def __gt__(self, other: Any) -> bool:
        return self.value > int(other)

    def __int__(self) -> int:
        return self.value

    def __neg__(self) -> "Parity":
        return Parity(-self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({_to_fraction(self.value)})"


def _to_fraction(value: Union[float, int], render_plus: bool = False) -> str:
    label = str(Fraction(value))
    if render_plus and value > 0:
        return f"+{label}"
    return label


@attr.s(frozen=True, init=False)
class EdgeQuantumNumbers:  # pylint: disable=too-many-instance-attributes
    """Definition of quantum numbers for edges.

    This class defines the types that are used in the
    `~.reaction.conservation_rules`, for instance in
    `.additive_quantum_number_rule`. You can also create data classes (see
    `attr.s`) with data members that are typed as the data members of
    `.EdgeQuantumNumbers` (see for example `.HelicityParityEdgeInput`) and use
    them in conservation rules that satisfy the appropriate rule protocol (see
    `.ConservationRule`, `.EdgeQNConservationRule`).
    """

    pid = NewType("pid", int)
    mass = NewType("mass", float)
    width = NewType("width", float)
    spin_magnitude = NewType("spin_magnitude", float)
    spin_projection = NewType("spin_projection", float)
    charge = NewType("charge", int)
    isospin_magnitude = NewType("isospin_magnitude", float)
    isospin_projection = NewType("isospin_projection", float)
    strangeness = NewType("strangeness", int)
    charmness = NewType("charmness", int)
    bottomness = NewType("bottomness", int)
    topness = NewType("topness", int)
    baryon_number = NewType("baryon_number", int)
    electron_lepton_number = NewType("electron_lepton_number", int)
    muon_lepton_number = NewType("muon_lepton_number", int)
    tau_lepton_number = NewType("tau_lepton_number", int)
    parity = NewType("parity", Parity)
    c_parity = NewType("c_parity", Parity)
    g_parity = NewType("g_parity", Parity)


for edge_qn_name, edge_qn_type in EdgeQuantumNumbers.__dict__.items():
    if not edge_qn_name.startswith("__"):
        edge_qn_type.__qualname__ = f"EdgeQuantumNumbers.{edge_qn_name}"
        edge_qn_type.__module__ = __name__


# for static typing
EdgeQuantumNumber = Union[
    EdgeQuantumNumbers.pid,
    EdgeQuantumNumbers.mass,
    EdgeQuantumNumbers.width,
    EdgeQuantumNumbers.spin_magnitude,
    EdgeQuantumNumbers.spin_projection,
    EdgeQuantumNumbers.charge,
    EdgeQuantumNumbers.isospin_magnitude,
    EdgeQuantumNumbers.isospin_projection,
    EdgeQuantumNumbers.strangeness,
    EdgeQuantumNumbers.charmness,
    EdgeQuantumNumbers.bottomness,
    EdgeQuantumNumbers.topness,
    EdgeQuantumNumbers.baryon_number,
    EdgeQuantumNumbers.electron_lepton_number,
    EdgeQuantumNumbers.muon_lepton_number,
    EdgeQuantumNumbers.tau_lepton_number,
    EdgeQuantumNumbers.parity,
    EdgeQuantumNumbers.c_parity,
    EdgeQuantumNumbers.g_parity,
]


@attr.s(frozen=True, init=False)
class NodeQuantumNumbers:
    """Definition of quantum numbers for interaction nodes."""

    l_magnitude = NewType("l_magnitude", float)
    l_projection = NewType("l_projection", float)
    s_magnitude = NewType("s_magnitude", float)
    s_projection = NewType("s_projection", float)
    parity_prefactor = NewType("parity_prefactor", float)


for node_qn_name, node_qn_type in NodeQuantumNumbers.__dict__.items():
    if not node_qn_name.startswith("__"):
        node_qn_type.__qualname__ = f"NodeQuantumNumbers.{node_qn_name}"
        node_qn_type.__module__ = __name__


# for static typing
NodeQuantumNumber = Union[
    NodeQuantumNumbers.l_magnitude,
    NodeQuantumNumbers.l_projection,
    NodeQuantumNumbers.s_magnitude,
    NodeQuantumNumbers.s_projection,
    NodeQuantumNumbers.parity_prefactor,
]


def _to_optional_float(optional_float: Optional[float]) -> Optional[float]:
    if optional_float is None:
        return None
    return float(optional_float)


def _to_optional_int(optional_int: Optional[int]) -> Optional[int]:
    if optional_int is None:
        return None
    return int(optional_int)


@attr.s(frozen=True)
class InteractionProperties:
    """Immutable data structure containing interaction properties.

    .. note:: As opposed to `NodeQuantumNumbers`, the `InteractionProperties`
        class serves as an interface to the user.
    """

    l_magnitude: Optional[int] = attr.ib(  # L cannot be half integer
        default=None, converter=_to_optional_int
    )
    l_projection: Optional[int] = attr.ib(
        default=None, converter=_to_optional_int
    )
    s_magnitude: Optional[float] = attr.ib(
        default=None, converter=_to_optional_float
    )
    s_projection: Optional[float] = attr.ib(
        default=None, converter=_to_optional_float
    )
    parity_prefactor: Optional[float] = attr.ib(
        default=None, converter=_to_optional_float
    )


def arange(
    x_1: float, x_2: float, delta: float
) -> Generator[float, None, None]:
    current = Decimal(x_1)
    while current < x_2:
        yield float(current)
        current += Decimal(delta)
