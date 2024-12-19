from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import pytest
import qrules
import sympy as sp
from qrules.topology import FrozenDict

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


class TestLargeHash:
    initial_state: ClassVar = [("J/psi(1S)", [-1, 1])]
    final_state: ClassVar = ["gamma", "pi0", "pi0"]
    allowed_intermediate_particles: ClassVar = ["f(0)(980)", "f(0)(1500)"]
    allowed_interaction_types: ClassVar = "strong"

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            ("UNSTABLE-HASH", "canonical-helicity"),
            ("UNSTABLE-HASH", "helicity"),
        ],
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
            ("0047c8be9e94ec7d5d0e0a326b9f7266", "canonical-helicity"),
            ("562d5f1390b56ddb83149d2218ff4aea", "helicity"),
        ],
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


class TestFrozenDict:
    def test_qrules_frozen_dict(self):
        obj: FrozenDict = FrozenDict({})
        assert get_readable_hash(obj) == "023f1d9cf3576a46b278007d6e5cb0fa"

        obj = FrozenDict({"key1": "value1"})
        assert get_readable_hash(obj) == "076915d85252aa5c87f63702437c6dbd"

        obj = FrozenDict({
            "key1": "value1",
            "key2": 2,
            "key3": (1, 2, 3),
            "key4": FrozenDict({"nested_key": "nested_value"}),
        })
        assert get_readable_hash(obj) == "c3b3a661e63f57003fc3deea5d23ede9"
