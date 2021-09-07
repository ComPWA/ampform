import pickle

import sympy as sp

from ampform.dynamics import (
    BlattWeisskopfSquared,
    EnergyDependentWidth,
    PhaseSpaceFactorAnalytic,
)
from ampform.sympy import UnevaluatedExpression


class TestUnevaluatedExpression:
    @staticmethod
    def test_pickle():
        s, m0, w0, m_a, angular_momentum, z = sp.symbols("s m0 Gamma0 m_a L z")

        # Pickle simple SymPy expression
        expr = z * angular_momentum
        pickled_obj = pickle.dumps(expr)
        imported_expr = pickle.loads(pickled_obj)
        assert expr == imported_expr

        # Pickle UnevaluatedExpression
        expr = UnevaluatedExpression()
        pickled_obj = pickle.dumps(expr)
        imported_expr = pickle.loads(pickled_obj)
        assert expr == imported_expr

        # Pickle classes derived from UnevaluatedExpression
        expr = BlattWeisskopfSquared(angular_momentum, z=z)
        pickled_obj = pickle.dumps(expr)
        imported_expr = pickle.loads(pickled_obj)
        assert expr == imported_expr

        expr = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_a,
            angular_momentum=0,
            meson_radius=1,
            phsp_factor=PhaseSpaceFactorAnalytic,
            name="Gamma_1",
        )
        pickled_obj = pickle.dumps(expr)
        imported_expr = pickle.loads(pickled_obj)
        assert expr == imported_expr
