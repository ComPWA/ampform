# flake8: noqa
# pylint: disable=invalid-name

"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import textwrap
from typing import Callable, Type, Union

import sympy as sp

from ampform.dynamics import (
    BlattWeisskopf,
    breakup_momentum,
    coupled_width,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)


def update_docstring(
    class_type: Union[Callable, Type], appended_text: str
) -> None:
    appended_text = textwrap.indent(appended_text, prefix=4 * " ")
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


L = sp.Symbol("L", integer=True)
z = sp.Symbol("z", real=True)
ff2 = BlattWeisskopf(L, z) ** 2
update_docstring(
    BlattWeisskopf,
    f"""
.. math:: {sp.latex(ff2)} = {sp.latex(ff2.doit())}
    :label: BlattWeisskopf
""",
)

m, m_a, m_b = sp.symbols("m m_a m_b")
q = breakup_momentum(m, m_a, m_b)
update_docstring(
    breakup_momentum,
    f"""
.. math:: q^2(m) = {sp.latex(q ** 2)}
    :label: breakup_momentum
""",
)

m, m0, w0, ma, mb, d = sp.symbols("m m0 Gamma m_a m_b d")
running_width = coupled_width(
    mass=m,
    mass0=m0,
    gamma0=w0,
    m_a=ma,
    m_b=mb,
    angular_momentum=L,
    meson_radius=d,
)
q = breakup_momentum(m, m_a, m_b)
q0 = breakup_momentum(m0, m_a, m_b)
ff = BlattWeisskopf(L, z=(q * d) ** 2)
ff0 = BlattWeisskopf(L, z=(q0 * d) ** 2)
running_width = running_width.subs(
    {
        2 * q: sp.Symbol("q^{2}(m)"),
        2 * q0: sp.Symbol("q^{2}(m_0)"),
        ff: sp.Symbol("B_{L}(q)"),
        ff0: sp.Symbol("B_{L}(q_{0})"),
    }
)
update_docstring(
    coupled_width,
    fR"""
AmpForm uses the following shape for the "mass-dependent" width in a
`.relativistic_breit_wigner_with_ff`:

.. math:: \Gamma(m) = {sp.latex(running_width)}
    :label: coupled_width

where :math:`B_L(q)` is defined by :eq:`BlattWeisskopf` and :math:`q^2(m)` is
defined by :eq:`breakup_momentum`.
""",
)


m, m0, w0 = sp.symbols("m m0 Gamma")
rel_bw = relativistic_breit_wigner(m, m0, w0)
update_docstring(
    relativistic_breit_wigner,
    f"""
.. math:: {sp.latex(rel_bw)}
    :label: relativistic_breit_wigner
""",
)


m, m0, w0, ma, mb, d = sp.symbols("m m0 Gamma m_a m_b d")
rel_bw_with_ff = relativistic_breit_wigner_with_ff(
    mass=m,
    mass0=m0,
    gamma0=w0,
    m_a=ma,
    m_b=mb,
    angular_momentum=L,
    meson_radius=d,
)
q = breakup_momentum(m, m_a, m_b)
ff = BlattWeisskopf(L, z=(q * d) ** 2)
mass_dependent_width = coupled_width(m, m0, w0, m_a, m_b, L, d)
rel_bw_with_ff = rel_bw_with_ff.subs(
    {
        2 * q: sp.Symbol("q^{2}(m)"),
        ff: sp.Symbol(R"B_{L}\left(q(m)\right)"),
        mass_dependent_width: sp.Symbol(R"\Gamma(m)"),
    }
)
update_docstring(
    relativistic_breit_wigner_with_ff,
    fR"""
The general form of a relativistic Breit-Wigner with `.BlattWeisskopf` form
factor is:

.. math:: {sp.latex(rel_bw_with_ff)}
    :label: relativistic_breit_wigner_with_ff_general

where :math:`\Gamma(m)` is defined by :eq:`coupled_width`, :math:`B_L(q)` is
defined by :eq:`BlattWeisskopf`, and :math:`q^2(m)` is defined by
:eq:`breakup_momentum`.
""",
)
