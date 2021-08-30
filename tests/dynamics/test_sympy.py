import pickle

import sympy as sp

from ampform.dynamics import BlattWeisskopfSquared
from ampform.sympy import UnevaluatedExpression


class TestUnevaluatedExpression:
    @staticmethod
    def test_pickle():
        z = sp.Symbol("z")
        angular_momentum = sp.Symbol("L", integer=True)

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

        # Pickle class derived from UnevaluatedExpression
        expr = BlattWeisskopfSquared(angular_momentum, z=z)
        pickled_obj = pickle.dumps(expr)
        imported_expr = pickle.loads(pickled_obj)
        assert expr == imported_expr
