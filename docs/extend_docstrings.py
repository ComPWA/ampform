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
""",
)


m, m0, w0 = sp.symbols("m m0 Gamma", real=True)
rel_bw = relativistic_breit_wigner(m, m0, w0)
update_docstring(
    relativistic_breit_wigner,
    f"""
.. math:: {sp.latex(rel_bw)}
""",
)


m, m0, w0, ma, mb, d = sp.symbols("m m0 Gamma m_a m_b d", real=True)
rel_bw_with_ff = relativistic_breit_wigner_with_ff(
    mass=m,
    mass0=m0,
    gamma0=w0,
    m_a=ma,
    m_b=mb,
    angular_momentum=0,
    meson_radius=d,
)
update_docstring(
    relativistic_breit_wigner_with_ff,
    f"""
For :math:`L=0`, this lineshape has the following form:

.. math:: {sp.latex(rel_bw_with_ff.doit())}
""",
)
