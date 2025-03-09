from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, ClassVar

import pytest
import qrules
import sympy as sp
from frozendict import frozendict

import ampform
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import RelativisticBreitWignerBuilder
from ampform.sympy._cache import get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from qrules.transition import SpinFormalism


@pytest.mark.parametrize(
    ("expected_hash", "assumptions"),
    [
        ("a7559ca", dict()),
        ("f4b1fad", dict(real=True)),
        ("d5bdc74", dict(rational=True)),
    ],
    ids=["symbol", "symbol-real", "symbol-rational"],
)
def test_get_readable_hash(
    assumptions: dict, expected_hash: str, caplog: LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)[:7]
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
    h = get_readable_hash(expr)[:7]
    assert h == "ccafec3"


class TestLargeHash:
    initial_state: ClassVar = [("J/psi(1S)", [-1, 1])]
    final_state: ClassVar = ["gamma", "pi0", "pi0"]
    allowed_intermediate_particles: ClassVar = ["f(0)(980)", "f(0)(1500)"]
    allowed_interaction_types: ClassVar = "strong"

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            (
                "762cc00" if sys.version_info >= (3, 11) else "1f5ac33",
                "canonical-helicity",
            ),
            (
                "17fefe5" if sys.version_info >= (3, 11) else "7b5fad1",
                "helicity",
            ),
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
        h = get_readable_hash(reaction)[:7]
        assert h == expected_hash

    @pytest.mark.parametrize(
        ("expected_hashes", "formalism"),
        [
            ({"4765e78", "8bf5459"}, "canonical-helicity"),
            ({"3bf2c7a", "915fff3"}, "helicity"),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    def test_amplitude_model(self, expected_hashes: set[str], formalism: SpinFormalism):
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [-1, 1])],
            final_state=["p~", "K0", "Sigma+"],
            allowed_intermediate_particles=[
                "N(1650)+",  # largest branching fraction
                "N(1675)+",  # high LS couplings
                "Sigma(1385)",  # largest branching fraction
                "Sigma(1775)",  # high LS couplings
            ],
            allowed_interaction_types="strong",
            formalism=formalism,
        )
        model_builder = ampform.get_builder(reaction)
        has_ls_couplings = formalism == "canonical-helicity"
        dynamics_builder = RelativisticBreitWignerBuilder(
            form_factor=has_ls_couplings,
            energy_dependent_width=has_ls_couplings,
        )
        for name in reaction.get_intermediate_particles().names:
            model_builder.dynamics.assign(name, dynamics_builder)
        model = model_builder.formulate()

        intensity = model.intensity.doit()
        assert any(isinstance(s, sp.Indexed) for s in intensity.free_symbols)

        intensity_hash = get_readable_hash(intensity)[:7]
        assert intensity_hash == "6a98bbf"

        amplitudes = frozendict({k: v.doit() for k, v in model.amplitudes.items()})
        unfolded_intensity = intensity.xreplace(amplitudes)
        unfolded_intensity_hash = get_readable_hash(unfolded_intensity)[:7]
        assert unfolded_intensity_hash in expected_hashes
        # Hash is not fully stable yet! See https://github.com/ComPWA/ampform-dpd/discussions/163
