from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, ClassVar

import pytest
import qrules
import sympy as sp

from ampform import get_builder
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.sympy import perform_cached_doit, perform_cached_substitution
from ampform.sympy._cache import get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from qrules.transition import SpinFormalism

    from ampform.helicity import HelicityModel


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


def test_perform_cached_doit(amplitude_model: tuple[str, HelicityModel]):
    _, model = amplitude_model
    expected_expr = model.expression.doit()
    assert expected_expr != model.expression

    unfolded_expr_1 = perform_cached_doit(model.expression)
    assert unfolded_expr_1 == expected_expr
    unfolded_expr_2 = perform_cached_doit(model.expression)
    assert unfolded_expr_2 == expected_expr


@pytest.mark.parametrize(
    "substitution_name", ["parameter_defaults", "kinematic_variables"]
)
def test_perform_cached_substitution(
    amplitude_model: tuple[str, HelicityModel], substitution_name: str
):
    _, model = amplitude_model
    full_expression = model.expression.doit()
    substitutions: dict[sp.Symbol, sp.Basic] = getattr(model, substitution_name)
    expected_expr = full_expression.xreplace(substitutions)
    assert expected_expr != full_expression

    substituted_expr_1 = perform_cached_substitution(full_expression, substitutions)
    assert substituted_expr_1 == expected_expr
    substituted_expr_2 = perform_cached_substitution(full_expression, substitutions)
    assert substituted_expr_2 == expected_expr


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
        ("expected_hash", "formalism"),
        [
            ("01bb112", "canonical-helicity"),
            ("0638a0e", "helicity"),
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
        h = get_readable_hash(model.expression)[:7]
        assert h == expected_hash
