from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import pytest
import qrules
import sympy as sp

from ampform import get_builder
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.sympy._cache import get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from qrules.transition import SpinFormalism


@pytest.mark.parametrize(
    ("expected_hash", "assumptions"),
    [
        ("a7559ca6e9037446da186a62ff1ed673", dict()),
        ("f4b1fadd65890a86cbac47da2435694d", dict(real=True)),
        ("d5bdc74485ba4d3f16f387541315f64d", dict(rational=True)),
    ],
    ids=["symbol", "symbol-real", "symbol-rational"],
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
    assert h == "ccafec30a6b6974920652b9df1b13c57"


class TestLargeHash:
    initial_state: ClassVar = [("J/psi(1S)", [-1, 1])]
    final_state: ClassVar = ["gamma", "pi0", "pi0"]
    allowed_intermediate_particles: ClassVar = ["f(0)(980)", "f(0)(1500)"]
    allowed_interaction_types: ClassVar = "strong"

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            ("762cc006a8c4c0a0a88fce934a32577d", "canonical-helicity"),
            ("17fefe55a7da0810371e90bd762a176a", "helicity"),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    def test_reaction(self, expected_hash: str, formalism: SpinFormalism):
        reaction = qrules.generate_transitions(
            initial_state=self.initial_state,
            final_state=self.final_state,
            allowed_intermediate_particles=self.allowed_intermediate_particles,
            allowed_interaction_types=self.allowed_interaction_types,
            formalism=formalism,
        )
        assert get_readable_hash(reaction) == expected_hash

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            ("01bb11213896fcb173cb9b4ea4bdc3bf", "canonical-helicity"),
            ("0638a0ec2efa602e4c1f79141f7ca88a", "helicity"),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    def test_amplitude_model(self, expected_hash: str, formalism: SpinFormalism):
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [-1, 1])],
            final_state=["gamma", "pi0", "pi0"],
            allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
            allowed_interaction_types="strong",
            formalism=formalism,
        )
        builder = get_builder(reaction)
        for name in reaction.get_intermediate_particles().names:
            builder.dynamics.assign(name, create_relativistic_breit_wigner_with_ff)
        model = builder.formulate()
        assert get_readable_hash(model.expression) == expected_hash
