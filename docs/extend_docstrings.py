# flake8: noqa
# pylint: disable=invalid-name

"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import inspect
from typing import Callable, Type, Union

import sympy as sp

from ampform.dynamics import (
    BlattWeisskopfSquared,
    _analytic_continuation,
    _phase_space_factor_hat,
    breakup_momentum_squared,
    coupled_width,
    phase_space_factor,
    phase_space_factor_ac,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)
from ampform.dynamics.math import ComplexSqrt


def update_docstring(
    class_type: Union[Callable, Type], appended_text: str
) -> None:
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


def render_blatt_weisskopf() -> None:
    L = sp.Symbol("L", integer=True)
    z = sp.Symbol("z", real=True)
    ff2 = BlattWeisskopfSquared(L, z)
    update_docstring(
        BlattWeisskopfSquared,
        f"""
    .. math:: {sp.latex(ff2)} = {sp.latex(ff2.doit())}
        :label: BlattWeisskopfSquared
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


def render_complex_sqrt() -> None:
    x = sp.Symbol("x", real=True)
    complex_sqrt = ComplexSqrt(x)
    update_docstring(
        ComplexSqrt,
        fR"""
    .. math:: {sp.latex(complex_sqrt)} = {sp.latex(complex_sqrt.evaluate())}
        :label: ComplexSqrt
    """,
    )


def render_coupled_width() -> None:
    L = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d")
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
    form_factor_sq = BlattWeisskopfSquared(L, z=q_squared * d ** 2)
    form_factor0_sq = BlattWeisskopfSquared(L, z=q0_squared * d ** 2)
    rho = phase_space_factor(s, m_a, m_b)
    rho0 = phase_space_factor(m0 ** 2, m_a, m_b)
    running_width = running_width.subs(
        {
            rho / rho0: sp.Symbol(R"\rho(s)") / sp.Symbol(R"\rho(m_{0})"),
            form_factor_sq: sp.Symbol("B_{L}^2(q)"),
            form_factor0_sq: sp.Symbol("B_{L}^2(q_{0})"),
        }
    )
    update_docstring(
        coupled_width,
        fR"""
    With that in mind, the "mass-dependent" width in a
    `.relativistic_breit_wigner_with_ff` becomes:

    .. math:: \Gamma(s) = {sp.latex(running_width)}
        :label: coupled_width

    where :math:`B_L^2(q)` is defined by :eq:`BlattWeisskopfSquared`,
    :math:`q(s)` is defined by :eq:`breakup_momentum_squared`, and
    :math:`\rho(s)` is (by default) defined by :eq:`phase_space_factor`.
    """,
    )


def render_phase_space_factor() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    rho = phase_space_factor(s, m_a, m_b)
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    rho = rho.subs({4 * q_squared: 4 * sp.Symbol("q^{2}(s)")})
    update_docstring(
        phase_space_factor,
        f"""

    .. math:: {sp.latex(rho)}
        :label: phase_space_factor

    with :math:`q^2(s)` defined as :eq:`breakup_momentum_squared`.
    """,
    )


def render_phase_space_factor_ac() -> None:
    s, m_a, m_b, rho_hat_symbol, q_squared_symbol = sp.symbols(
        R"s, m_a, m_b, \hat{\rho}, q^{2}(s)"
    )
    rho_analytic = _analytic_continuation(
        rho_hat_symbol, s, s_threshold=(m_a + m_b) ** 2
    )
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    rho_hat = _phase_space_factor_hat(s, m_a, m_b)
    rho_hat_subs = rho_hat.subs(4 * q_squared, 4 * q_squared_symbol)
    update_docstring(
        phase_space_factor_ac,
        fR"""
    .. math:: {sp.latex(rho_analytic)}
        :label: phase_space_factor_ac

    with :math:`\hat{{\rho}}` a slightly adapted :func:`phase_space_factor`
    that takes the absolute value of :func:`.breakup_momentum_squared`:

    .. math:: {sp.latex(rho_hat_subs)}
        :label: rho_hat
    """,
    )


def render_relativistic_breit_wigner() -> None:
    s, m0, w0 = sp.symbols("s m0 Gamma0")
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
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d")
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
    ff_sq = BlattWeisskopfSquared(L, z=q_squared * d ** 2)
    mass_dependent_width = coupled_width(s, m0, w0, m_a, m_b, L, d)
    rel_bw_with_ff = rel_bw_with_ff.subs(
        {
            2 * q_squared: 2 * sp.Symbol("q^{2}(s)"),
            ff_sq: sp.Symbol(R"B_{L}^2\left(q(s)\right)"),
            mass_dependent_width: sp.Symbol(R"\Gamma(s)"),
        }
    )
    update_docstring(
        relativistic_breit_wigner_with_ff,
        fR"""
    The general form of a relativistic Breit-Wigner with Blatt-Weisskopf form
    factor is:

    .. math:: {sp.latex(rel_bw_with_ff)}
        :label: relativistic_breit_wigner_with_ff

    where :math:`\Gamma(s)` is defined by :eq:`coupled_width`, :math:`B_L^2(q)`
    is defined by :eq:`BlattWeisskopfSquared`, and :math:`q(s)` is defined by
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
