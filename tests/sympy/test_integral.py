import numpy as np
import pytest
import sympy as sp

from ampform.sympy import NumericalIntegral


class TestNumericalIntegral:
    @pytest.mark.parametrize(
        ("backend", "algorithm", "configuration"),
        [
            ("jax", "quadax.quadgk", None),
            ("jax", "quadax.romberg", None),
            ("numpy", "scipy.integrate.quad_vec", None),
            ("numpy", "scipy.integrate.quad_vec", {"limit": 10}),
            ("numpy", None, None),
        ],
    )
    @pytest.mark.parametrize("call_doit", [False, True])
    @pytest.mark.parametrize("dummify", [False, True])
    def test_real_value_function(
        self,
        algorithm: str | None,
        backend: str,
        call_doit: bool,
        configuration: dict[str, int | None],
        dummify: bool,
    ):
        x = sp.symbols("x")
        integral_expr = NumericalIntegral(
            x**2,
            (x, 1, 3),
            algorithm=algorithm,
            configuration=configuration,
            dummify=dummify,
        )
        if call_doit:
            integral_expr = integral_expr.doit()
        assert integral_expr.algorithm == algorithm
        assert integral_expr.configuration == configuration
        assert integral_expr.dummify is dummify
        func = sp.lambdify([], integral_expr, backend)
        assert func() == 26 / 3  # noqa: RUF069

    @pytest.mark.parametrize(
        ("p_value", "expected"),
        [
            (2, 26 / 3),
            (1, 4),
            (1j, (1 / 2 - 1j / 2) * (-1 + 3 ** (1 + 1j))),
            (
                np.array([0, 0.5, 1, 2]),
                np.array([2, 2 * 3 ** (1 / 2) - 2 / 3, 4, 8 + 2 / 3]),
            ),
        ],
    )
    def test_evaluation_over_arrays(self, p_value, expected):
        x, p = sp.symbols("x,p")
        integral_expr = NumericalIntegral(x**p, (x, 1, 3))
        func = sp.lambdify(args=[p], expr=integral_expr)
        assert pytest.approx(func(p=p_value)) == expected
