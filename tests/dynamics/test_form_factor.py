import pytest
import sympy as sp

from ampform.dynamics.form_factor import (
    BlattWeisskopfSquared,
    FormFactor,
    _get_polynomial_blatt_weisskopf,
)
from ampform.helicity import ParameterValues
from ampform.kinematics.phasespace import BreakupMomentumSquared

z = sp.Symbol("z", nonnegative=True, real=True)


def describe_BlattWeisskopfSquared():
    @pytest.mark.parametrize(
        ("ell", "expected"),
        [
            (0, 1),
            (1, z / (1 + z)),
            (2, z**2 / (9 + 3 * z + z**2)),
        ],
    )
    def it_matches_the_pdg_convention_without_normalization(
        ell: int, expected: sp.Expr
    ):
        expr = BlattWeisskopfSquared(z, angular_momentum=ell, normalize=False)
        assert sp.simplify(expr.doit() - expected) == 0

    def it_omits_the_normalization_for_symbolic_angular_momentum():
        ell = sp.Symbol("L", integer=True, nonnegative=True)
        expr = BlattWeisskopfSquared(z, angular_momentum=ell, normalize=False).doit()
        expected = z**2 / (9 + 3 * z + z**2)
        assert sp.simplify(expr.subs(ell, 2).doit() - expected) == 0

    def it_marks_the_normalized_function_with_a_hat():
        assert (
            sp.latex(BlattWeisskopfSquared(z, angular_momentum=1))
            == R"\hat{B}_{1}^2\left(z\right)"
        )
        expr = BlattWeisskopfSquared(z, angular_momentum=1, normalize=False)
        assert sp.latex(expr) == R"B_{1}^2\left(z\right)"


def describe_FormFactor():
    def it_forwards_the_normalization_flag():
        s, m1, m2, d = sp.symbols("s m1 m2 d", nonnegative=True)
        form_factor = FormFactor(
            s, m1, m2, angular_momentum=1, meson_radius=d, normalize=False
        )
        q2 = BreakupMomentumSquared(s, m1, m2)
        expected = BlattWeisskopfSquared(q2 * d**2, angular_momentum=1, normalize=False)
        assert form_factor.evaluate() == sp.sqrt(expected)

    def it_marks_the_normalized_form_factor_with_a_hat():
        s, m1, m2 = sp.symbols("s m1 m2", nonnegative=True)
        arguments = R"_{1}\left(s, m_{1}, m_{2}\right)"
        normalized = FormFactor(s, m1, m2, angular_momentum=1)
        non_normalized = FormFactor(s, m1, m2, angular_momentum=1, normalize=False)
        assert sp.latex(normalized) == R"\hat{\mathcal{F}}" + arguments
        assert sp.latex(non_normalized) == R"\mathcal{F}" + arguments

    def it_keeps_the_normalization_flag_when_substituting_parameter_values():
        s, m1, m2, d = sp.symbols("s m1 m2 d", nonnegative=True)
        form_factor = FormFactor(
            s, m1, m2, angular_momentum=2, meson_radius=d, normalize=False
        )
        parameters = ParameterValues({d: 5.0, m1: 0.938, m2: 0.493})
        substituted = form_factor.xreplace(parameters)
        assert substituted.normalize is False
        assert substituted.meson_radius == 5.0


@pytest.mark.parametrize(
    ("ell", "expected"),
    [
        (0, 1),
        (1, 2 * z / (z + 1)),
        (2, 13 * z**2 / (z**2 + 3 * z + 9)),
        (3, 277 * z**3 / (z**3 + 6 * z**2 + 45 * z + 225)),
        (4, 12746 * z**4 / (z**4 + 10 * z**3 + 135 * z**2 + 1575 * z + 11025)),
        (
            10,
            451873017324894386
            * z**10
            / (
                z**10
                + 55 * z**9
                + 4455 * z**8
                + 386100 * z**7
                + 33108075 * z**6
                + 2681754075 * z**5
                + 196661965500 * z**4
                + 12417798393000 * z**3
                + 628651043645625 * z**2
                + 22561587455281875 * z
                + 428670161650355625
            ),
        ),
    ],
)
def test_get_polynomial_blatt_weisskopf(ell: int, expected: sp.Expr):
    expr = _get_polynomial_blatt_weisskopf(ell)(z)
    assert expr == expected
