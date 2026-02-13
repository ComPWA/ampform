import warnings

import numpy as np
import sympy as sp

from ampform.dynamics.phasespace import ChewMandelstamIntegral, ChewMandelstamSWave


class TestChewMandelstam:
    def test_numerical_integral_s_wave(self):
        s = sp.Symbol("s")
        m1 = 0.938
        m2 = 0.140
        analytic_cm_expr = ChewMandelstamSWave(s, m1, m2).evaluate()
        ϵ = 1e-5  # noqa: PLC2401
        numerical_cm_expr = ChewMandelstamIntegral(s, m1, m2, L=0, epsilon=ϵ)
        analytic_cm_func = sp.lambdify(s, analytic_cm_expr.doit())
        numerical_cm_func = sp.lambdify(s, numerical_cm_expr.doit())
        s_values = np.linspace(0.1, 10, num=80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            analytic_cm_values = analytic_cm_func(s_values)
            numerical_cm_values = numerical_cm_func(s_values)
        np.testing.assert_array_almost_equal(
            analytic_cm_values,
            numerical_cm_values,
            decimal=2,
        )
