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
        ("8eb22db72da08d18", dict()),
        ("40b00fbe1e8ddf6b", dict(real=True)),
        ("94569e99c6b4ceec", dict(rational=True)),
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
    assert h == "eee5788b2adb017c"


def test_get_readable_hash_large(amplitude_model: tuple[str, HelicityModel]):
    formalism, model = amplitude_model
    expected_hash = {
        "canonical-helicity": "5d0e0a326b9f7266",
        "helicity": "83149d2218ff4aea",
    }[formalism]
    assert get_readable_hash(model.expression) == expected_hash
