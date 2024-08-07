from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules

import ampform
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff

if TYPE_CHECKING:
    import sympy as sp
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.benchmark(group="doit", min_rounds=1)
def test_doit_speed(benchmark: BenchmarkFixture) -> None:
    reaction = qrules.generate_transitions(
        initial_state=("psi(4160)", [-1, +1]),
        final_state=["D-", "D0", "pi+"],
        allowed_intermediate_particles=["D*(2007)0"],
        formalism="canonical-helicity",
    )
    builder = ampform.get_builder(reaction)
    for particle in reaction.get_intermediate_particles():
        builder.dynamics.assign(particle.name, create_relativistic_breit_wigner_with_ff)
    model = builder.formulate()

    intensity_expr = benchmark(_perform_doit, model.expression)
    undefined_symbols = intensity_expr.free_symbols
    undefined_symbols -= set(model.parameter_defaults)
    undefined_symbols -= set(model.kinematic_variables)
    assert not undefined_symbols


def _perform_doit(expr: sp.Expr):
    return expr.doit()
