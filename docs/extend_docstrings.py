# flake8: noqa
# pylint: disable=invalid-name

"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import inspect
from typing import Callable, Type, Union

import sympy as sp

from ampform.dynamics import (
    BlattWeisskopf,
    breakup_momentum_squared,
    coupled_width,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)


def update_docstring(
    class_type: Union[Callable, Type], appended_text: str
) -> None:
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


def render_blatt_weisskopf() -> None:
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


def render_breakup_momentum() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    update_docstring(
        breakup_momentum_squared,
        f"""
    .. math:: q^2(s) = {sp.latex(q_squared)}
        :label: breakup_momentum_squared
    """,
    )


def render_coupled_width() -> None:
    L = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma m_a m_b d")
    running_width = coupled_width(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=L,
        meson_radius=d,
    )
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    q0_squared = breakup_momentum_squared(m0 ** 2, m_a, m_b)
    q = sp.sqrt(q_squared)
    q0 = sp.sqrt(q0_squared)
    ff = BlattWeisskopf(L, z=q_squared * d ** 2)
    ff0 = BlattWeisskopf(L, z=q0_squared * d ** 2)
    running_width = running_width.subs(
        {
            2 * q: sp.Symbol("q(s)"),
            2 * q0: sp.Symbol("q(m_0)"),
            ff: sp.Symbol("B_{L}(q)"),
            ff0: sp.Symbol("B_{L}(q_{0})"),
        }
    )
    update_docstring(
        coupled_width,
        fR"""
    AmpForm uses the following shape for the "mass-dependent" width in a
    `.relativistic_breit_wigner_with_ff`:

    .. math:: \Gamma(s) = {sp.latex(running_width)}
        :label: coupled_width

    where :math:`B_L(q)` is defined by :eq:`BlattWeisskopf` and :math:`q^2(s)`
    is defined by :eq:`breakup_momentum_squared`.
    """,
    )


def render_relativistic_breit_wigner() -> None:
    s, m0, w0 = sp.symbols("s m0 Gamma")
    rel_bw = relativistic_breit_wigner(s, m0, w0)
    update_docstring(
        relativistic_breit_wigner,
        f"""
    .. math:: {sp.latex(rel_bw)}
        :label: relativistic_breit_wigner
    """,
    )


def render_relativistic_breit_wigner_with_ff() -> None:
    L = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma m_a m_b d")
    rel_bw_with_ff = relativistic_breit_wigner_with_ff(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=L,
        meson_radius=d,
    )
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    ff = BlattWeisskopf(L, z=q_squared * d ** 2)
    mass_dependent_width = coupled_width(s, m0, w0, m_a, m_b, L, d)
    rel_bw_with_ff = rel_bw_with_ff.subs(
        {
            2 * q_squared: sp.Symbol("q^{2}(s)"),
            ff: sp.Symbol(R"B_{L}\left(q(s)\right)"),
            mass_dependent_width: sp.Symbol(R"\Gamma(s)"),
        }
    )
    update_docstring(
        relativistic_breit_wigner_with_ff,
        fR"""
    The general form of a relativistic Breit-Wigner with `.BlattWeisskopf` form
    factor is:

    .. math:: {sp.latex(rel_bw_with_ff)}
        :label: relativistic_breit_wigner_with_ff_general

    where :math:`\Gamma(s)` is defined by :eq:`coupled_width`, :math:`B_L(q)`
    is defined by :eq:`BlattWeisskopf`, and :math:`q^2(s)` is defined by
    :eq:`breakup_momentum_squared`.
    """,
    )


SCRIPT_NAME = __file__.split("/")[-1]
SCRIPT_NAME = ".".join(SCRIPT_NAME.split(".")[:-1])


def insert_math() -> None:
    definitions = dict(globals())
    for name, definition in definitions.items():
        module = inspect.getmodule(definition)
        if module is None:
            continue
        if module.__name__ not in {"__main__", SCRIPT_NAME}:
            continue
        if not inspect.isfunction(definition):
            continue
        if not name.startswith("render_"):
            continue
        function_arguments = inspect.signature(definition).parameters
        if len(function_arguments):
            raise ValueError(
                f"Local function {name} should not have a signature"
            )
        definition()
