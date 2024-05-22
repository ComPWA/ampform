import pytest
import sympy as sp

from ampform.dynamics.form_factor import _get_polynomial_blatt_weisskopf

z = sp.Symbol("z", nonnegative=True, real=True)


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
