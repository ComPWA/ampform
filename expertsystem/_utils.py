"""A collection of general, stand-alone utilities for internal use."""


from decimal import Decimal
from math import copysign
from typing import Generator, Union


def arange(
    x_1: float, x_2: float, delta: float
) -> Generator[float, None, None]:
    current = Decimal(x_1)
    while current < x_2:
        yield float(current)
        current += Decimal(delta)


def sign(value: Union[float, int]) -> int:
    return int(copysign(1, value))
