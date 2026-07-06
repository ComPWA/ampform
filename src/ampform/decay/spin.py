"""Functions for generating spin projections and LS couplings."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Generator

    from ampform.decay import ParticleLike


def generate_ls_couplings(
    parent_spin: SupportsFloat,
    child1_spin: SupportsFloat,
    child2_spin: SupportsFloat,
    max_L: int = 3,  # ruff: ignore[invalid-argument-name]
) -> list[tuple[int, sp.Rational]]:
    """Generate a list of allowed LS couplings.

    >>> generate_ls_couplings(1.5, 0.5, 0)
    [(1, 1/2), (2, 1/2)]
    """
    s1 = float(child1_spin)
    s2 = float(child2_spin)
    angular_momenta = create_rational_range(0, max_L)
    coupled_spins = create_rational_range(abs(s1 - s2), s1 + s2)
    ls_couplings = {
        (int(L), S)
        for L in angular_momenta
        for S in coupled_spins
        if abs(L - S) <= parent_spin <= L + S
    }
    return sorted(ls_couplings)


def filter_parity_violating_ls(
    ls_couplings: list[tuple[int, sp.Rational]],
    parent_parity: SupportsInt,
    child1_parity: SupportsInt,
    child2_parity: SupportsInt,
) -> list[tuple[int, sp.Rational]]:
    """Filter parity-violating LS combinations from a list of LS couplings.

    >>> LS = generate_ls_couplings(0.5, 1.5, 0)  # Λc → Λ(1520)π
    >>> LS
    [(1, 3/2), (2, 3/2)]
    >>> filter_parity_violating_ls(LS, +1, -1, -1)
    [(2, 3/2)]
    """
    η0, η1, η2 = (
        int(parent_parity),
        int(child1_parity),
        int(child2_parity),
    )
    return [(L, S) for L, S in ls_couplings if η0 == η1 * η2 * (-1) ** L]


def get_spin_projections(particle: ParticleLike) -> list[sp.Rational]:
    r"""Get the allowed spin projections (helicities) of a particle.

    The projections are determined from the spin magnitude, where massless particles,
    like the photon, are the edge case: they have no longitudinal (zero) projection.

    >>> from ampform.decay import Particle
    >>> photon = Particle(
    ...     "gamma", latex=R"\gamma", spin=1, parity=-1, mass=0.0, width=0.0
    ... )
    >>> get_spin_projections(photon)
    [-1, 1]
    >>> omega = Particle(
    ...     "omega(782)", latex=R"\omega", spin=1, parity=-1, mass=0.78, width=0.01
    ... )
    >>> get_spin_projections(omega)
    [-1, 0, 1]
    """
    return create_spin_range(particle.spin, no_zero_spin=particle.mass == 0)


def create_spin_range(
    spin: SupportsFloat, no_zero_spin: bool = False
) -> list[sp.Rational]:
    """Create a range of allowed spin projections.

    >>> create_spin_range(1.5)
    [-3/2, -1/2, 1/2, 3/2]
    >>> create_spin_range(1, no_zero_spin=True)
    [-1, 1]
    """
    spin_projections = create_rational_range(-float(spin), +float(spin))
    if no_zero_spin and 0 in spin_projections and len(spin_projections) > 1:
        spin_projections.remove(0)
    return spin_projections


def create_rational_range(
    __from: SupportsFloat, __to: SupportsFloat, /
) -> list[sp.Rational]:
    """Create a range of rational numbers, especially useful for spin projections.

    >>> create_rational_range(-0.5, +1.5)
    [-1/2, 1/2, 3/2]
    """
    spin_range = arange(float(__from), +float(__to) + 0.5)
    return [sp.Rational(x) for x in spin_range]


def arange(x_1: float, x_2: float, delta: float = 1.0) -> Generator[float, None, None]:
    current = Decimal(x_1)
    while current < x_2:
        yield float(current)
        current += Decimal(delta)
