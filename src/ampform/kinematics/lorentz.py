"""Symbolic implementations for Lorentz vectors and boosts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import sympy as sp

from ampform.helicity.decay import determine_attached_final_state, list_decay_chain_ids
from ampform.sympy import (
    NumPyPrintable,
    UnevaluatedExpression,
    _implement_latex_subscript,
    create_expression,
    implement_doit_method,
    make_commutative,
)
from ampform.sympy._array_expressions import (
    ArrayAxisSum,
    ArrayMultiplication,
    ArraySlice,
    ArraySum,
    ArraySymbol,
)
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from qrules.topology import Topology
    from sympy.printing.latex import LatexPrinter
    from sympy.printing.numpy import NumPyPrinter


def create_four_momentum_symbols(topology: Topology) -> FourMomenta:
    """Create a set of array-symbols for a `~qrules.topology.Topology`.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> create_four_momentum_symbols(topologies[0])
    {0: p0, 1: p1, 2: p2}
    """
    final_state_ids = sorted(topology.outgoing_edge_ids)
    return {i: create_four_momentum_symbol(i) for i in final_state_ids}


def create_four_momentum_symbol(index: int) -> FourMomentumSymbol:
    return FourMomentumSymbol(f"p{index}", shape=[])


FourMomenta = Dict[int, "FourMomentumSymbol"]
"""A mapping of state IDs to their corresponding `.FourMomentumSymbol`.

It's best to create a `dict` of `.FourMomenta` with
:func:`create_four_momentum_symbols`.
"""
FourMomentumSymbol = ArraySymbol
r"""Array-`~sympy.core.symbol.Symbol` that represents an array of four-momenta.

The array is assumed to be of shape :math:`n\times 4` with :math:`n` the number of
events. The four-momenta are assumed to be in the order :math:`\left(E,\vec{p}\right)`.
See also `Energy`, `FourMomentumX`, `FourMomentumY`, and `FourMomentumZ`.
"""


# for numpy broadcasting
ArraySlice = make_commutative(ArraySlice)  # type: ignore[misc]


@implement_doit_method
@make_commutative
class Energy(UnevaluatedExpression):
    """Represents the energy-component of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> Energy:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 0))

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self._momentum)
        return Rf"E\left({momentum}\right)"


@_implement_latex_subscript(subscript="x")
@implement_doit_method
@make_commutative
class FourMomentumX(UnevaluatedExpression):
    """Component :math:`x` of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> FourMomentumX:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 1))


@_implement_latex_subscript(subscript="y")
@implement_doit_method
@make_commutative
class FourMomentumY(UnevaluatedExpression):
    """Component :math:`y` of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> FourMomentumY:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 2))


@_implement_latex_subscript(subscript="z")
@implement_doit_method
@make_commutative
class FourMomentumZ(UnevaluatedExpression):
    """Component :math:`z` of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> FourMomentumZ:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 3))


@implement_doit_method
@make_commutative
class ThreeMomentum(UnevaluatedExpression, NumPyPrintable):
    """Spatial components of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> ThreeMomentum:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), slice(1, None)))

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self._momentum)
        return Rf"\vec{{{momentum}}}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        return printer._print(self.evaluate())


@implement_doit_method
@make_commutative
class EuclideanNorm(UnevaluatedExpression, NumPyPrintable):
    """Take the euclidean norm of an array over axis 1."""

    def __new__(cls, vector: sp.Basic, **hints) -> EuclideanNorm:
        return create_expression(cls, vector, **hints)

    @property
    def _vector(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArraySlice:
        return sp.sqrt(EuclideanNormSquared(self._vector))

    def _latex(self, printer: LatexPrinter, *args) -> str:
        vector = printer._print(self._vector)
        return Rf"\left|{vector}\right|"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        return printer._print(self.evaluate(), *args)


@implement_doit_method
@make_commutative
class EuclideanNormSquared(UnevaluatedExpression):
    """Take the squared euclidean norm of an array over axis 1."""

    def __new__(cls, vector: sp.Basic, **hints) -> EuclideanNormSquared:
        return create_expression(cls, vector, **hints)

    @property
    def _vector(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ArrayAxisSum:
        return ArrayAxisSum(self._vector**2, axis=1)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        vector = printer._print(self._vector, *args)
        return Rf"\left|{vector}\right|^{{2}}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        return printer._print(self.evaluate(), *args)


def three_momentum_norm(momentum: sp.Basic) -> EuclideanNorm:
    return EuclideanNorm(ThreeMomentum(momentum))


@implement_doit_method
@make_commutative
class InvariantMass(UnevaluatedExpression):
    """Invariant mass of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> InvariantMass:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> ComplexSqrt:
        p = self._momentum
        p_xyz = ThreeMomentum(p)
        return ComplexSqrt(Energy(p) ** 2 - EuclideanNorm(p_xyz) ** 2)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self._momentum)
        return f"m_{{{momentum}}}"


@implement_doit_method
@make_commutative
class NegativeMomentum(UnevaluatedExpression):
    r"""Invert the spatial components of a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> NegativeMomentum:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    def evaluate(self) -> sp.Expr:
        p = self._momentum
        eta = MinkowskiMetric(p)
        return ArrayMultiplication(eta, p)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self._momentum)
        return Rf"-\left({momentum}\right)"


class MinkowskiMetric(NumPyPrintable):
    r"""Minkowski metric :math:`\eta = (1, -1, -1, -1)`."""

    def __new__(cls, momentum: sp.Basic, **hints) -> MinkowskiMetric:
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> MinkowskiMetric:
        return self.args[0]  # type: ignore[return-value]

    def as_explicit(self) -> sp.MutableDenseMatrix:
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ])

    def _latex(self, printer: LatexPrinter, *args) -> str:
        return R"\boldsymbol{\eta}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].update({"array", "ones", "zeros"})
        momentum = printer._print(self._momentum)
        n_events = f"len({momentum})"
        zeros = f"zeros({n_events})"
        ones = f"ones({n_events})"
        return f"""array(
            [
                [{ones}, {zeros}, {zeros}, {zeros}],
                [{zeros}, -{ones}, {zeros}, {zeros}],
                [{zeros}, {zeros}, -{ones}, {zeros}],
                [{zeros}, {zeros}, {zeros}, -{ones}],
            ]
        ).transpose((2, 0, 1))"""


@implement_doit_method
class BoostZMatrix(UnevaluatedExpression):
    r"""Represents a Lorentz boost matrix in the :math:`z`-direction.

    Args:
        beta: Velocity in the :math:`z`-direction, :math:`\beta=p_z/E`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`beta`.
    """

    def __new__(
        cls, beta: sp.Basic, n_events: sp.Expr | None = None, **kwargs
    ) -> BoostZMatrix:
        if n_events is None:
            n_events = ArraySize(beta)
        return create_expression(cls, beta, n_events, **kwargs)

    def as_explicit(self) -> sp.MutableDenseMatrix:
        beta = self.args[0]
        gamma = 1 / ComplexSqrt(1 - beta**2)  # type: ignore[operator]
        return sp.Matrix([
            [gamma, 0, 0, -gamma * beta],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-gamma * beta, 0, 0, gamma],
        ])

    def evaluate(self) -> _BoostZMatrixImplementation:
        beta = self.args[0]
        gamma = 1 / sp.sqrt(1 - beta**2)  # type: ignore[operator]
        n_events = self.args[1]
        return _BoostZMatrixImplementation(
            beta=beta,
            gamma=gamma,
            gamma_beta=gamma * beta,
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        return printer._print(self.evaluate(), *args)


class _BoostZMatrixImplementation(NumPyPrintable):
    def __new__(
        cls,
        beta: sp.Basic,
        gamma: sp.Basic,
        gamma_beta: sp.Basic,
        ones: _OnesArray,
        zeros: _ZerosArray,
        **hints,
    ) -> _BoostZMatrixImplementation:
        return create_expression(cls, beta, gamma, gamma_beta, ones, zeros, **hints)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        beta = printer._print(self.args[0])
        return Rf"\boldsymbol{{B_z}}\left({beta}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("array")
        _, gamma, gamma_beta, ones, zeros = map(printer._print, self.args)
        return f"""array(
            [
                [{gamma}, {zeros}, {zeros}, -{gamma_beta}],
                [{zeros}, {ones}, {zeros}, {zeros}],
                [{zeros}, {zeros}, {ones}, {zeros}],
                [-{gamma_beta}, {zeros}, {zeros}, {gamma}],
            ]
        ).transpose((2, 0, 1))"""


@implement_doit_method
class BoostMatrix(UnevaluatedExpression):
    r"""Compute a rank-3 Lorentz boost matrix from a `.FourMomentumSymbol`."""

    def __new__(cls, momentum: sp.Basic, **kwargs) -> BoostMatrix:
        return create_expression(cls, momentum, **kwargs)

    def as_explicit(self) -> sp.MutableDenseMatrix:
        momentum = self.args[0]
        energy = Energy(momentum)
        beta_sq = EuclideanNormSquared(ThreeMomentum(momentum)) / energy**2
        beta_x = FourMomentumX(momentum) / energy
        beta_y = FourMomentumY(momentum) / energy
        beta_z = FourMomentumZ(momentum) / energy
        g = 1 / sp.sqrt(1 - beta_sq)
        return sp.Matrix([
            [g, -g * beta_x, -g * beta_y, -g * beta_z],
            [
                -g * beta_x,
                1 + (g - 1) * beta_x**2 / beta_sq,
                (g - 1) * beta_y * beta_x / beta_sq,
                (g - 1) * beta_z * beta_x / beta_sq,
            ],
            [
                -g * beta_y,
                (g - 1) * beta_x * beta_y / beta_sq,
                1 + (g - 1) * beta_y**2 / beta_sq,
                (g - 1) * beta_z * beta_y / beta_sq,
            ],
            [
                -g * beta_z,
                (g - 1) * beta_x * beta_z / beta_sq,
                (g - 1) * beta_y * beta_z / beta_sq,
                1 + (g - 1) * beta_z**2 / beta_sq,
            ],
        ])

    def evaluate(self) -> _BoostMatrixImplementation:
        momentum = self.args[0]
        energy = Energy(momentum)
        beta_sq = EuclideanNormSquared(ThreeMomentum(momentum)) / energy**2
        beta_x = FourMomentumX(momentum) / energy
        beta_y = FourMomentumY(momentum) / energy
        beta_z = FourMomentumZ(momentum) / energy
        gamma = 1 / sp.sqrt(1 - beta_sq)
        return _BoostMatrixImplementation(
            momentum,
            b00=gamma,
            b01=-gamma * beta_x,
            b02=-gamma * beta_y,
            b03=-gamma * beta_z,
            b11=1 + (gamma - 1) * beta_x**2 / beta_sq,
            b12=(gamma - 1) * beta_x * beta_y / beta_sq,
            b13=(gamma - 1) * beta_x * beta_z / beta_sq,
            b22=1 + (gamma - 1) * beta_y**2 / beta_sq,
            b23=(gamma - 1) * beta_y * beta_z / beta_sq,
            b33=1 + (gamma - 1) * beta_z**2 / beta_sq,
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self.args[0])
        return Rf"\boldsymbol{{B}}\left({momentum}\right)"


class _BoostMatrixImplementation(NumPyPrintable):
    def __new__(
        cls,
        momentum: sp.Basic,
        b00: sp.Basic,
        b01: sp.Basic,
        b02: sp.Basic,
        b03: sp.Basic,
        b11: sp.Basic,
        b12: sp.Basic,
        b13: sp.Basic,
        b22: sp.Basic,
        b23: sp.Basic,
        b33: sp.Basic,
        **kwargs,
    ) -> _BoostMatrixImplementation:
        return create_expression(
            cls,
            momentum,
            b00,
            b01,
            b02,
            b03,
            b11,
            b12,
            b13,
            b22,
            b23,
            b33,
            **kwargs,
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        momentum = printer._print(self.args[0])
        return Rf"\boldsymbol{{B}}\left({momentum}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        _, b00, b01, b02, b03, b11, b12, b13, b22, b23, b33 = self.args
        return f"""array(
            [
                [{b00}, {b01}, {b02}, {b03}],
                [{b01}, {b11}, {b12}, {b13}],
                [{b02}, {b12}, {b22}, {b23}],
                [{b03}, {b13}, {b23}, {b33}],
            ]
        ).transpose((2, 0, 1))"""


@implement_doit_method
class RotationYMatrix(UnevaluatedExpression):
    r"""Rotation matrix around the :math:`y`-axis for a `.FourMomentumSymbol`.

    Args:
        angle: Angle with which to rotate, see e.g. `.Phi` and `.Theta`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`angle`.
    """

    def __new__(
        cls, angle: sp.Basic, n_events: sp.Expr | None = None, **hints
    ) -> RotationYMatrix:
        if n_events is None:
            n_events = ArraySize(angle)
        return create_expression(cls, angle, n_events, **hints)

    def as_explicit(self) -> sp.MutableDenseMatrix:
        angle = self.args[0]
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(angle), 0, sp.sin(angle)],
            [0, 0, 1, 0],
            [0, -sp.sin(angle), 0, sp.cos(angle)],
        ])

    def evaluate(self) -> _RotationYMatrixImplementation:
        angle = self.args[0]
        n_events = self.args[1]
        return _RotationYMatrixImplementation(
            angle=angle,
            cos_angle=sp.cos(angle),
            sin_angle=sp.sin(angle),
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        return printer._print(self.evaluate(), *args)


class _RotationYMatrixImplementation(NumPyPrintable):
    def __new__(
        cls,
        angle: sp.Basic,
        cos_angle: sp.Basic,
        sin_angle: sp.Basic,
        ones: _OnesArray,
        zeros: _ZerosArray,
        **hints,
    ) -> _RotationYMatrixImplementation:
        return create_expression(cls, angle, cos_angle, sin_angle, ones, zeros, **hints)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        angle, *_ = self.args
        angle_latex = printer._print(angle)
        return Rf"\boldsymbol{{R_y}}\left({angle_latex}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("array")
        _, cos_angle, sin_angle, ones, zeros = map(printer._print, self.args)
        return f"""array(
            [
                [{ones}, {zeros}, {zeros}, {zeros}],
                [{zeros}, {cos_angle}, {zeros}, {sin_angle}],
                [{zeros}, {zeros}, {ones}, {zeros}],
                [{zeros}, -{sin_angle}, {zeros}, {cos_angle}],
            ]
        ).transpose((2, 0, 1))"""


@implement_doit_method
class RotationZMatrix(UnevaluatedExpression):
    r"""Rotation matrix around the :math:`z`-axis for a `.FourMomentumSymbol`.

    Args:
        angle: Angle with which to rotate, see e.g. `.Phi` and `.Theta`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`angle`.
    """

    def __new__(
        cls, angle: sp.Basic, n_events: sp.Expr | None = None, **hints
    ) -> RotationZMatrix:
        if n_events is None:
            n_events = ArraySize(angle)
        return create_expression(cls, angle, n_events, **hints)

    def as_explicit(self) -> sp.MutableDenseMatrix:
        angle = self.args[0]
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle), 0],
            [0, sp.sin(angle), sp.cos(angle), 0],
            [0, 0, 0, 1],
        ])

    def evaluate(self) -> _RotationZMatrixImplementation:
        angle = self.args[0]
        n_events = self.args[1]
        return _RotationZMatrixImplementation(
            angle=angle,
            cos_angle=sp.cos(angle),
            sin_angle=sp.sin(angle),
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        return printer._print(self.evaluate(), *args)


class _RotationZMatrixImplementation(NumPyPrintable):
    def __new__(
        cls,
        angle: sp.Basic,
        cos_angle: sp.Basic,
        sin_angle: sp.Basic,
        ones: _OnesArray,
        zeros: _ZerosArray,
        **hints,
    ) -> _RotationZMatrixImplementation:
        return create_expression(cls, angle, cos_angle, sin_angle, ones, zeros, **hints)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        angle, *_ = self.args
        angle_latex = printer._print(angle)
        return Rf"\boldsymbol{{R_z}}\left({angle_latex}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("array")
        _, cos_angle, sin_angle, ones, zeros = map(printer._print, self.args)
        return f"""array(
            [
                [{ones}, {zeros}, {zeros}, {zeros}],
                [{zeros}, {cos_angle}, -{sin_angle}, {zeros}],
                [{zeros}, {sin_angle}, {cos_angle}, {zeros}],
                [{zeros}, {zeros}, {zeros}, {ones}],
            ]
        ).transpose((2, 0, 1))"""


class _OnesArray(NumPyPrintable):
    def __new__(cls, shape, **kwargs) -> _OnesArray:
        return create_expression(cls, shape, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("ones")
        shape = printer._print(self.args[0])
        return f"ones({shape})"


class _ZerosArray(NumPyPrintable):
    def __new__(cls, shape, **kwargs) -> _ZerosArray:
        return create_expression(cls, shape, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("zeros")
        shape = printer._print(self.args[0])
        return f"zeros({shape})"


class ArraySize(NumPyPrintable):
    """Symbolic expression for getting the size of a numerical array."""

    def __new__(cls, array: sp.Basic, **kwargs) -> ArraySize:
        return create_expression(cls, array, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        shape = printer._print(self.args[0])
        return f"len({shape})"


def compute_boost_chain(
    topology: Topology, momenta: FourMomenta, state_id: int
) -> list[BoostMatrix]:
    boost_matrices = []
    decay_chain_state_ids = __get_boost_chain_ids(topology, state_id)
    boosted_momenta: dict[int, sp.Expr] = {
        i: get_four_momentum_sum(topology, momenta, i) for i in decay_chain_state_ids
    }
    for current_state_id in decay_chain_state_ids:
        current_momentum = boosted_momenta[current_state_id]
        boost = BoostMatrix(current_momentum)
        boosted_momenta = {
            i: ArrayMultiplication(boost, p) for i, p in boosted_momenta.items()
        }
        boost_matrices.append(boost)
    return boost_matrices


def __get_boost_chain_ids(topology: Topology, state_id: int) -> list[int]:
    """Get the state IDs from first resonance to this final state.

    >>> from qrules.topology import create_isobar_topologies
    >>> topology = create_isobar_topologies(3)[0]
    >>> __get_boost_chain_ids(topology, state_id=0)
    [0]
    >>> __get_boost_chain_ids(topology, state_id=1)
    [3, 1]
    >>> __get_boost_chain_ids(topology, state_id=2)
    [3, 2]
    """
    decay_chain_state_ids = list(reversed(list_decay_chain_ids(topology, state_id)))
    initial_state_id = next(iter(topology.incoming_edge_ids))
    decay_chain_state_ids.remove(initial_state_id)
    return decay_chain_state_ids


def get_four_momentum_sum(
    topology: Topology, momenta: FourMomenta, state_id: int
) -> ArraySum | FourMomentumSymbol:
    """Get the `.FourMomentumSymbol` or sum of momenta for **any** edge ID.

    If the edge ID is a final state ID, return its `.FourMomentumSymbol`. If it's an
    intermediate edge ID, return the sum of the momenta of the final states to which it
    decays.

    >>> from qrules.topology import create_isobar_topologies
    >>> topology = create_isobar_topologies(3)[0]
    >>> momenta = create_four_momentum_symbols(topology)
    >>> get_four_momentum_sum(topology, momenta, state_id=0)
    p0
    >>> get_four_momentum_sum(topology, momenta, state_id=3)
    p1 + p2
    """
    if state_id in topology.outgoing_edge_ids:
        return momenta[state_id]
    sub_momenta_ids = determine_attached_final_state(topology, state_id)
    return ArraySum(*[momenta[i] for i in sub_momenta_ids])


def compute_invariant_masses(
    four_momenta: FourMomenta, topology: Topology
) -> dict[sp.Symbol, sp.Expr]:
    """Compute the invariant masses for all final state combinations."""
    if topology.outgoing_edge_ids != set(four_momenta):
        msg = (
            f"Momentum IDs {set(four_momenta)} do not match final state edge IDs"
            f" {set(topology.outgoing_edge_ids)}"
        )
        raise ValueError(msg)
    invariant_masses: dict[sp.Symbol, sp.Expr] = {}
    for state_id in topology.edges:
        attached_state_ids = determine_attached_final_state(topology, state_id)
        total_momentum = ArraySum(*[four_momenta[i] for i in attached_state_ids])
        expr = InvariantMass(total_momentum)
        symbol = get_invariant_mass_symbol(topology, state_id)
        invariant_masses[symbol] = expr
    return invariant_masses


def get_invariant_mass_symbol(topology: Topology, state_id: int) -> sp.Symbol:
    """Generate an invariant mass label for a state (edge on a topology).

    Example
    -------
    In the case shown in Figure :ref:`one-to-five-topology-0`, the invariant mass of
    state :math:`5` is :math:`m_{034}`, because :math:`p_5=p_0+p_3+p_4`:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> get_invariant_mass_symbol(topologies[0], state_id=5)
    m_034

    Naturally, the 'invariant' mass label for a final state is just the mass of the
    state itself:

    >>> get_invariant_mass_symbol(topologies[0], state_id=1)
    m_1
    """
    final_state_ids = determine_attached_final_state(topology, state_id)
    mass_name = f"m_{''.join(map(str, sorted(final_state_ids)))}"
    return sp.Symbol(mass_name, nonnegative=True)
