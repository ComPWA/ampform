import sympy as sp

from ampform.dynamics.form_factor import BlattWeisskopfSquared


class TestBlattWeisskopfSquared:
    def test_factorials(self):
        z = sp.Symbol("z")
        angular_momentum = sp.Symbol("L", integer=True)
        form_factor = BlattWeisskopfSquared(z, angular_momentum)
        form_factor_9 = form_factor.subs(angular_momentum, 8).evaluate()
        factor, z_power, _ = form_factor_9.args
        assert factor == 4392846440677
        assert z_power == z**8
