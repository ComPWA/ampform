from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform.dynamics import EnergyDependentWidth
from ampform.sympy._cache import get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from ampform.helicity import HelicityModel


@pytest.mark.parametrize(
    ("expected_hash", "assumptions"),
    [
        ("87ecd62a4c584b2a", dict()),
        ("1c4981af01b2fc14", dict(real=True)),
        ("115ed5f321a7fe96", dict(rational=True)),
    ],
)
def test_get_readable_hash(
    assumptions: dict, expected_hash: str, caplog: LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)
    assert h == expected_hash
    assert not caplog.text


def test_get_readable_hash_energy_dependent_width():
    angular_momentum = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d", nonnegative=True)
    expr = EnergyDependentWidth(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=angular_momentum,
        meson_radius=d,
    )
    h = get_readable_hash(expr)
    assert h == "e722efd64af3d92b"


def test_get_readable_hash_large(amplitude_model: tuple[str, HelicityModel]):
    formalism, model = amplitude_model
    expected_hash = {
        "canonical-helicity": "67a26cb9955c17b8",
        "helicity": "5636f9e0f8a20ad4",
    }[formalism]
    assert get_readable_hash(model.expression) == expected_hash
