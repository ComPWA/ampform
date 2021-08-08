# pylint: disable=no-self-use, protected-access
import re

import pytest
import sympy as sp

from ampform.dynamics.kmatrix import NonRelativisticKMatrix
from symplot import rename_symbols, substitute_indexed_symbols


class TestNonRelativisticKMatrix:
    @pytest.mark.parametrize(
        "n_channels",
        [1, 2, pytest.param(3, marks=pytest.mark.slow)],
    )
    def test_breit_wigner(self, n_channels: int):
        k_matrix = NonRelativisticKMatrix.formulate(
            n_poles=1, n_channels=n_channels
        )
        breit_wigner = k_matrix[0, 0].doit().simplify()
        breit_wigner = substitute_indexed_symbols(breit_wigner)
        breit_wigner = _remove_residue_constants(breit_wigner)
        breit_wigner = _rename_widths(breit_wigner)
        factor = ""
        if n_channels > 1:
            factor += f"{n_channels}"
            factor += "*"
        assert str(breit_wigner) == fR"-m1*w1/(-m1**2 + {factor}I*m1*w1 + s)"

    def test_interference_single_channel(self):
        k_matrix = NonRelativisticKMatrix.formulate(n_poles=2, n_channels=1)
        expr = k_matrix[0, 0].doit()
        expr = substitute_indexed_symbols(expr)
        expr = _remove_residue_constants(expr)
        expr = _rename_widths(expr)
        denominator, nominator = expr.args
        term1 = nominator.args[0] * denominator
        term2 = nominator.args[1] * denominator
        assert str(term1 / term2) == R"m1*w1*(m2**2 - s)/(m2*w2*(m1**2 - s))"


def _remove_residue_constants(expression: sp.Expr) -> sp.Expr:
    residue_constants = filter(
        lambda s: s.name.startswith(R"\gamma_"), expression.free_symbols
    )
    return expression.xreplace({gamma: 1 for gamma in residue_constants})


def _rename_widths(expression: sp.Expr) -> sp.Expr:
    """Use 'w' instead of 'Gamma' to signify widths."""
    return rename_symbols(
        expression,
        lambda s: re.sub(r"\\Gamma_{([0-9]),[0-9]}", r"w\1", s),
    )
