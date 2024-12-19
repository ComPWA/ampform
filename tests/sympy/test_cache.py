from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform import get_builder
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.sympy._cache import get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from qrules import ReactionInfo


@pytest.mark.parametrize(
    ("expected_hash", "assumptions"),
    [
        ("b0ccf9b61d730ae0ae4e1d024e765375", dict()),
        ("7db2db79aa9f7bd4caca01a2082f4638", dict(real=True)),
        ("12c6c97d066784a23076bf172eb86260", dict(rational=True)),
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
    assert h == "086a038e35f21ed6eee5788b2adb017c"


def test_get_readable_hash_large(reaction: ReactionInfo):
    model_builder = get_builder(reaction)
    for name in reaction.get_intermediate_particles().names:
        model_builder.dynamics.assign(name, create_relativistic_breit_wigner_with_ff)
    model = model_builder.formulate()
    expected_hash = {
        "canonical-helicity": "0047c8be9e94ec7d5d0e0a326b9f7266",
        "helicity": "562d5f1390b56ddb83149d2218ff4aea",
    }[reaction.formalism]
    assert get_readable_hash(model.expression) == expected_hash
