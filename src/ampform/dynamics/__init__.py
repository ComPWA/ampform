# cspell:ignore Asner
# pylint: disable=protected-access, unbalanced-tuple-unpacking, unused-argument
"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/usage/dynamics` and
    :doc:`/usage/dynamics/analytic-continuation`
"""

from typing import Any, Optional

import sympy as sp
from sympy.printing.latex import LatexPrinter

from .decorator import (
    UnevaluatedExpression,
    implement_doit_method,
    verify_signature,
)
from .math import ComplexSqrt

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


@implement_doit_method()
class BlattWeisskopfSquared(UnevaluatedExpression):
    r"""Blatt-Weisskopf function :math:`B_L^2(z)`, up to :math:`L \leq 8`.

    Args:
        angular_momentum: Angular momentum :math:`L` of the decaying particle.

        z: Argument of the Blatt-Weisskopf function :math:`B_L^2(z)`. A usual
            choice is :math:`z = (d q)^2` with :math:`d` the impact parameter
            and :math:`q` the breakup-momentum (see
            :func:`breakup_momentum_squared`).

    Note that equal powers of :math:`z` appear in the nominator and the
    denominator, while some sources have nominator :math:`1`, instead of
    :math:`z^L`. Compare for instance :pdg-review:`2020; Resonances; p.6`, just
    before Equation (49.20).

    Each of these cases for :math:`L` has been taken from
    :cite:`pychyGekoppeltePartialwellenanalyseAnnihilationen2016`, p.59,
    :cite:`chungPartialWaveAnalysis1995`, p.415, and
    :cite:`chungFormulasAngularMomentumBarrier2015`. For a good overview of
    where to use these Blatt-Weisskopf functions, see
    :cite:`asnerDalitzPlotAnalysis2006`.

    See also :ref:`usage/dynamics/lineshapes:Form factor`.
    """
    is_commutative = True

    def __new__(  # pylint: disable=arguments-differ
        cls,
        angular_momentum: sp.Symbol,
        z: sp.Symbol,
        evaluate: bool = False,
        **hints: Any,
    ) -> "BlattWeisskopfSquared":
        args = sp.sympify((angular_momentum, z))
        if evaluate:
            # pylint: disable=no-member
            return sp.Expr.__new__(cls, *args, **hints).evaluate()
        return sp.Expr.__new__(cls, *args, **hints)

    def evaluate(self) -> sp.Expr:
        angular_momentum, z = self.args
        return sp.Piecewise(
            (
                1,
                sp.Eq(angular_momentum, 0),
            ),
            (
                2 * z / (z + 1),
                sp.Eq(angular_momentum, 1),
            ),
            (
                13 * z ** 2 / ((z - 3) * (z - 3) + 9 * z),
                sp.Eq(angular_momentum, 2),
            ),
            (
                (
                    277
                    * z ** 3
                    / (z * (z - 15) * (z - 15) + 9 * (2 * z - 5) * (2 * z - 5))
                ),
                sp.Eq(angular_momentum, 3),
            ),
            (
                (
                    12746
                    * z ** 4
                    / (
                        (z ** 2 - 45 * z + 105) * (z ** 2 - 45 * z + 105)
                        + 25 * z * (2 * z - 21) * (2 * z - 21)
                    )
                ),
                sp.Eq(angular_momentum, 4),
            ),
            (
                998881
                * z ** 5
                / (
                    z ** 5
                    + 15 * z ** 4
                    + 315 * z ** 3
                    + 6300 * z ** 2
                    + 99225 * z
                    + 893025
                ),
                sp.Eq(angular_momentum, 5),
            ),
            (
                118394977
                * z ** 6
                / (
                    z ** 6
                    + 21 * z ** 5
                    + 630 * z ** 4
                    + 18900 * z ** 3
                    + 496125 * z ** 2
                    + 9823275 * z
                    + 108056025
                ),
                sp.Eq(angular_momentum, 6),
            ),
            (
                19727003738
                * z ** 7
                / (
                    z ** 7
                    + 28 * z ** 6
                    + 1134 * z ** 5
                    + 47250 * z ** 4
                    + 1819125 * z ** 3
                    + 58939650 * z ** 2
                    + 1404728325 * z
                    + 18261468225
                ),
                sp.Eq(angular_momentum, 7),
            ),
            (
                4392846440677
                * z ** 8
                / (
                    z ** 8
                    + 36 * z ** 7
                    + 1890 * z ** 6
                    + 103950 * z ** 5
                    + 5457375 * z ** 4
                    + 255405150 * z ** 3
                    + 9833098275 * z ** 2
                    + 273922023375 * z
                    + 4108830350625
                ),
                sp.Eq(angular_momentum, 8),
            ),
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angular_momentum, z = tuple(map(printer._print, self.args))
        return fR"B_{{{angular_momentum}}}^2\left({z}\right)"


def relativistic_breit_wigner(
    s: sp.Symbol, mass0: sp.Symbol, gamma0: sp.Symbol
) -> sp.Expr:
    """Relativistic Breit-Wigner lineshape.

    See :ref:`usage/dynamics/lineshapes:_Without_ form factor` and
    :cite:`asnerDalitzPlotAnalysis2006`.
    """
    return gamma0 * mass0 / (mass0 ** 2 - s - gamma0 * mass0 * sp.I)


class PhaseSpaceFactor(Protocol):
    """Protocol that is used by :func:`.coupled_width`.

    Use this `~typing.Protocol` when defining other implementations of a phase
    space factor. Compare for instance :func:`.phase_space_factor` and
    :func:`.phase_space_factor_ac`.
    """

    def __call__(
        self, s: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
    ) -> sp.Expr:
        """Expected `~inspect.signature`."""
        ...


def phase_space_factor(
    s: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    """Standard phase-space factor, using :func:`breakup_momentum_squared`.

    See :pdg-review:`2020; Resonances; p.4`, Equation (49.8), with a slight
    adaptation: instead of a normal square root, this phase space factor make
    use of :eq:`ComplexSqrt` (`.ComplexSqrt`).
    """
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    denominator = _phase_space_factor_denominator(s)
    return ComplexSqrt(q_squared) / denominator


def phase_space_factor_ac(
    s: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    """Analytic continuation for the :func:`phase_space_factor`.

    See :pdg-review:`2014; Resonances; p.8` and
    :doc:`/usage/dynamics/analytic-continuation`.

    **Warning**: The PDG specifically derives this formula for a two-body decay
    *with equal masses*.
    """
    rho_hat = _phase_space_factor_hat(s, m_a, m_b)
    s_threshold = (m_a + m_b) ** 2
    return _analytic_continuation(rho_hat, s, s_threshold)


def _phase_space_factor_hat(
    s: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    """Phase space factor used in the analytic continuation."""
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    denominator = _phase_space_factor_denominator(s)
    return sp.sqrt(sp.Abs(q_squared)) / denominator


def _phase_space_factor_denominator(s: sp.Symbol) -> sp.Expr:
    return 8 * sp.pi * sp.sqrt(s)


def _analytic_continuation(
    rho: sp.Symbol, s: sp.Symbol, s_threshold: sp.Symbol
) -> sp.Expr:
    return sp.Piecewise(
        (
            sp.I * rho / sp.pi * sp.log(sp.Abs((1 + rho) / (1 - rho))),
            s < 0,
        ),
        (
            rho + sp.I * rho / sp.pi * sp.log(sp.Abs((1 + rho) / (1 - rho))),
            s > s_threshold,
        ),
        (
            2 * sp.I * rho / sp.pi * sp.atan(1 / rho),
            True,
        ),
    )


def coupled_width(  # pylint: disable=too-many-arguments
    s: sp.Symbol,
    mass0: sp.Symbol,
    gamma0: sp.Symbol,
    m_a: sp.Symbol,
    m_b: sp.Symbol,
    angular_momentum: sp.Symbol,
    meson_radius: sp.Symbol,
    phsp_factor: Optional[PhaseSpaceFactor] = None,
) -> sp.Expr:
    r"""Mass-dependent width, coupled to the pole position of the resonance.

    See :pdg-review:`2020; Resonances; p.6` and
    :cite:`asnerDalitzPlotAnalysis2006`, equation (6). Default value for
    :code:`phsp_factor` is :meth:`phase_space_factor`.

    Note that the `.BlattWeisskopfSquared` of AmpForm is normalized in the
    sense that equal powers of :math:`z` appear in the nominator and the
    denominator, while the definition in the PDG (as well as some other
    sources), always have :math:`1` in the nominator of the Blatt-Weisskopf. In
    that case, one needs an additional factor :math:`\left(q/q_0\right)^{2L}`
    in the definition for :math:`\Gamma(m)`.
    """
    if phsp_factor is not None:
        verify_signature(phsp_factor, PhaseSpaceFactor)
    else:
        phsp_factor = phase_space_factor
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    q0_squared = breakup_momentum_squared(mass0 ** 2, m_a, m_b)
    form_factor_sq = BlattWeisskopfSquared(
        angular_momentum, z=q_squared * meson_radius ** 2
    )
    form_factor0_sq = BlattWeisskopfSquared(
        angular_momentum, z=q0_squared * meson_radius ** 2
    )
    rho = phsp_factor(s, m_a, m_b)
    rho0 = phsp_factor(mass0 ** 2, m_a, m_b)
    return gamma0 * (form_factor_sq / form_factor0_sq) * (rho / rho0)


def relativistic_breit_wigner_with_ff(  # pylint: disable=too-many-arguments
    s: sp.Symbol,
    mass0: sp.Symbol,
    gamma0: sp.Symbol,
    m_a: sp.Symbol,
    m_b: sp.Symbol,
    angular_momentum: sp.Symbol,
    meson_radius: sp.Symbol,
    phsp_factor: Optional[PhaseSpaceFactor] = None,
) -> sp.Expr:
    """Relativistic Breit-Wigner with `.BlattWeisskopfSquared` factor.

    Note that this lineshape is set to zero for :math:`s < (m_a + m_b)^2` as a
    continuation of the `.BlattWeisskopfSquared` damping factor behavior at
    :math:`s = (m_a + m_b)^2`.

    See :ref:`usage/dynamics/lineshapes:_With_ form factor` and
    :pdg-review:`2020; Resonances; p.6`.
    """
    q_squared = breakup_momentum_squared(s, m_a, m_b)
    ff_squared = BlattWeisskopfSquared(
        angular_momentum, z=q_squared * meson_radius ** 2
    )
    s_threshold = (m_a + m_b) ** 2
    form_factor = sp.Piecewise(
        (0, sp.And(s < s_threshold, sp.Ne(angular_momentum, 0))),
        (sp.sqrt(ff_squared), True),
    )
    mass_dependent_width = coupled_width(
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius, phsp_factor
    )
    return (mass0 * gamma0 * form_factor) / (
        mass0 ** 2 - s - mass_dependent_width * mass0 * sp.I
    )


def breakup_momentum_squared(
    s: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    r"""Squared value of the two-body breakup-up momentum.

    For a two-body decay :math:`R \to ab`, the *break-up momentum* is the
    absolute value of the momentum of both :math:`a` and :math:`b` in the rest
    frame of :math:`R`.

    Args:
        s: :ref:`Mandelstam variable <pwa:mandelstam-variables>` :math:`s`.
            Commonly, this is just :math:`s = m_R^2`, with :math:`m_R` the
            invariant mass of decaying particle :math:`R`.

        m_a: Mass of decay product :math:`a`.
        m_b: Mass of decay product :math:`b`.

    See :pdg-review:`2020; Kinematics; p.3`.
    """
    return (s - (m_a + m_b) ** 2) * (s - (m_a - m_b) ** 2) / (4 * s)
