from __future__ import annotations

from decimal import Decimal
from typing import SupportsFloat


def create_spin_range(
    spin_magnitude: SupportsFloat, no_zero_spin: bool = False
) -> list[float]:
    """Create a list of allowed spin projections.

    >>> create_spin_range(0)
    [0.0]
    >>> create_spin_range(0.5)
    [-0.5, 0.5]
    >>> create_spin_range(1)
    [-1.0, 0.0, 1.0]
    >>> create_spin_range(1, no_zero_spin=True)
    [-1.0, 1.0]
    >>> projections = create_spin_range(5)
    >>> list(map(int, projections))
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    """
    spin_magnitude_float = float(spin_magnitude)
    spin_projections = []
    projection = Decimal(-spin_magnitude_float)
    while projection <= spin_magnitude_float:
        if projection == -0.0:
            projection = Decimal("0.0")
        spin_projections.append(float(projection))
        projection += 1
    if no_zero_spin and len(spin_projections) > 1:
        spin_projections.remove(0.0)
    return spin_projections
