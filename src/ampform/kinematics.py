# cspell:ignore einsum
# pylint: disable=arguments-differ,no-member,protected-access,unused-argument
"""Classes and functions for relativistic four-momentum kinematics."""

import itertools
import sys
from collections import abc
from functools import singledispatch
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import attrs
import sympy as sp
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter

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

if TYPE_CHECKING:  # pragma: no cover
    if sys.version_info < (3, 10):
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias


class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.create_expressions` method forms the bridge between four-momentum
    data for the decay you are studying and the kinematic variables that are in
    the `.HelicityModel`. These are invariant mass (see
    :func:`.get_invariant_mass_label`) and the :math:`\theta` and :math:`\phi`
    helicity angles (see :func:`.get_helicity_angle_label`).
    """

    def __init__(
        self,
        transitions: Union[
            ReactionInfo, Iterable[Union[Topology, StateTransition]]
        ],
    ) -> None:
        self.__topologies = _extract_topologies(transitions)
        for topology in self.__topologies:
            _assert_isobar_topology(topology)

    def register_transition(self, transition: StateTransition) -> None:
        topology = _get_topology(transition)
        self.register_topology(topology)

    def register_topology(self, topology: Topology) -> None:
        _assert_isobar_topology(topology)
        if self.__topologies:
            existing = next(iter(self.__topologies))
            if topology.incoming_edge_ids != existing.incoming_edge_ids:
                raise ValueError(
                    "Initial state ID mismatch those of existing topologies"
                )
            if topology.outgoing_edge_ids != existing.outgoing_edge_ids:
                raise ValueError(
                    "Final state IDs mismatch those of existing topologies"
                )
        self.__topologies.add(topology)

    @property
    def registered_topologies(self) -> FrozenSet[Topology]:
        return frozenset(self.__topologies)

    def permutate_registered_topologies(self) -> None:
        """Register outgoing edge permutations of all `registered_topologies`.

        See :ref:`usage/amplitude:Extend kinematic variables`.
        """
        for topology in set(self.__topologies):
            final_state_ids = topology.outgoing_edge_ids
            for permutation in itertools.permutations(final_state_ids):
                id_mapping = dict(zip(topology.outgoing_edge_ids, permutation))
                permuted_topology = attrs.evolve(
                    topology,
                    edges={
                        id_mapping.get(i, i): edge
                        for i, edge in topology.edges.items()
                    },
                )
                self.__topologies.add(permuted_topology)

    def create_expressions(self) -> Dict[str, sp.Expr]:
        output = {}
        for topology in self.__topologies:
            four_momenta = create_four_momentum_symbols(topology)
            output.update(compute_helicity_angles(four_momenta, topology))
            output.update(compute_invariant_masses(four_momenta, topology))
        return output


@singledispatch
def _extract_topologies(
    obj: Union[ReactionInfo, Iterable[Union[Topology, StateTransition]]]
) -> Set[Topology]:
    raise TypeError(f"Cannot extract topologies from a {type(obj).__name__}")


@_extract_topologies.register(ReactionInfo)
def _(transitions: ReactionInfo) -> Set[Topology]:
    return _extract_topologies(transitions.transitions)


@_extract_topologies.register(abc.Iterable)
def _(transitions: abc.Iterable) -> Set[Topology]:
    return {_get_topology(t) for t in transitions}


@singledispatch
def _get_topology(obj: Any) -> Topology:
    raise TypeError(
        f"Cannot create a {Topology.__name__} from a {type(obj).__name__}"
    )


@_get_topology.register(Topology)
def _(obj: Topology) -> Topology:
    return obj


@_get_topology.register(StateTransition)
def _(obj: StateTransition) -> Topology:
    return obj.topology


def create_four_momentum_symbols(topology: Topology) -> "FourMomenta":
    """Create a set of array-symbols for a `~qrules.topology.Topology`.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> create_four_momentum_symbols(topologies[0])
    {0: p0, 1: p1, 2: p2}
    """
    n_final_states = len(topology.outgoing_edge_ids)
    return {i: FourMomentumSymbol(f"p{i}") for i in range(n_final_states)}


FourMomenta = Dict[int, "FourMomentumSymbol"]
"""A mapping of state IDs to their corresponding `FourMomentumSymbol`.

It's best to create a `dict` of `FourMomenta` with
:func:`create_four_momentum_symbols`.
"""
FourMomentumSymbol: "TypeAlias" = ArraySymbol
r"""Array-`~sympy.core.symbol.Symbol` that represents an array of four-momenta.

The array is assumed to be of shape :math:`n\times 4` with :math:`n` the number
of events. The four-momenta are assumed to be in the order
:math:`\left(E,\vec{p}\right)`. See also `Energy`, `FourMomentumX`,
`FourMomentumY`, and `FourMomentumZ`.
"""


# for numpy broadcasting
ArraySlice = make_commutative(ArraySlice)  # type: ignore[misc]


@implement_doit_method
@make_commutative
class Energy(UnevaluatedExpression):
    """Represents the energy-component of a `FourMomentumSymbol`."""

    def __new__(cls, momentum: "FourMomentumSymbol", **hints: Any) -> "Energy":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 0))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self._momentum)
        return Rf"E\left({momentum}\right)"


@_implement_latex_subscript(subscript="x")
@implement_doit_method
@make_commutative
class FourMomentumX(UnevaluatedExpression):
    """Component :math:`x` of a `FourMomentumSymbol`."""

    def __new__(
        cls, momentum: "FourMomentumSymbol", **hints: Any
    ) -> "FourMomentumX":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 1))


@_implement_latex_subscript(subscript="y")
@implement_doit_method
@make_commutative
class FourMomentumY(UnevaluatedExpression):
    """Component :math:`y` of a `FourMomentumSymbol`."""

    def __new__(
        cls, momentum: "FourMomentumSymbol", **hints: Any
    ) -> "FourMomentumY":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 2))


@_implement_latex_subscript(subscript="z")
@implement_doit_method
@make_commutative
class FourMomentumZ(UnevaluatedExpression):
    """Component :math:`z` of a `FourMomentumSymbol`."""

    def __new__(
        cls, momentum: "FourMomentumSymbol", **hints: Any
    ) -> "FourMomentumZ":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self._momentum, (slice(None), 3))


@implement_doit_method
@make_commutative
class ThreeMomentumNorm(NumPyPrintable, UnevaluatedExpression):
    """Norm of the three-momentum of a `FourMomentumSymbol`."""

    def __new__(
        cls, momentum: "FourMomentumSymbol", **hints: Any
    ) -> "ThreeMomentumNorm":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        three_momentum = ArraySlice(
            self._momentum, (slice(None), slice(1, None))
        )
        norm_squared = ArrayAxisSum(three_momentum**2, axis=1)
        return sp.sqrt(norm_squared)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self._momentum)
        return Rf"\left|\vec{{{momentum}}}\right|"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        return printer._print(self.evaluate())


@implement_doit_method
@make_commutative
class InvariantMass(UnevaluatedExpression):
    """Invariant mass of a `FourMomentumSymbol`."""

    def __new__(cls, momentum: "FourMomentumSymbol", **hints: Any) -> "Energy":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> ArraySlice:
        p = self._momentum
        return ComplexSqrt(Energy(p) ** 2 - ThreeMomentumNorm(p) ** 2)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self._momentum)
        return f"m_{{{momentum}}}"


@implement_doit_method
@make_commutative
class Phi(UnevaluatedExpression):
    r"""Azimuthal angle :math:`\phi` of a `FourMomentumSymbol`."""

    def __new__(cls, momentum: "FourMomentumSymbol", **hints: Any) -> "Phi":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> sp.Expr:
        p = self._momentum
        return sp.atan2(FourMomentumY(p), FourMomentumX(p))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self._momentum)
        return Rf"\phi\left({momentum}\right)"


@implement_doit_method
@make_commutative
class Theta(UnevaluatedExpression):
    r"""Polar (elevation) angle :math:`\theta` of a `FourMomentumSymbol`."""

    def __new__(cls, momentum: "FourMomentumSymbol", **hints: Any) -> "Theta":
        return create_expression(cls, momentum, **hints)

    @property
    def _momentum(self) -> "FourMomentumSymbol":
        return self.args[0]

    def evaluate(self) -> sp.Expr:
        p = self._momentum
        return sp.acos(FourMomentumZ(p) / ThreeMomentumNorm(p))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self._momentum)
        return Rf"\theta\left({momentum}\right)"


@implement_doit_method
class BoostZMatrix(UnevaluatedExpression):
    r"""Represents a Lorentz boost matrix in the :math:`z`-direction.

    Args:
        beta: Velocity in the :math:`z`-direction, :math:`\beta=p_z/E`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`beta`.
    """

    def __new__(
        cls, beta: sp.Expr, n_events: Optional[sp.Symbol] = None, **kwargs: Any
    ) -> "BoostZMatrix":
        if n_events is None:
            n_events = _ArraySize(beta)
        return create_expression(cls, beta, n_events, **kwargs)

    def as_explicit(self) -> sp.Expr:
        beta = self.args[0]
        gamma = 1 / sp.sqrt(1 - beta**2)
        return sp.Matrix(
            [
                [gamma, 0, 0, -gamma * beta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-gamma * beta, 0, 0, gamma],
            ]
        )

    def evaluate(self) -> "_BoostZMatrixImplementation":
        beta = self.args[0]
        gamma = 1 / sp.sqrt(1 - beta**2)
        n_events = self.args[1]
        return _BoostZMatrixImplementation(
            beta=beta,
            gamma=gamma,
            gamma_beta=gamma * beta,
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        return printer._print(self.evaluate(), *args)


class _BoostZMatrixImplementation(NumPyPrintable):
    def __new__(  # pylint: disable=too-many-arguments
        cls,
        beta: sp.Expr,
        gamma: sp.Expr,
        gamma_beta: sp.Expr,
        ones: "_OnesArray",
        zeros: "_ZerosArray",
        **hints: Any,
    ) -> "_BoostZMatrixImplementation":
        return create_expression(
            cls, beta, gamma, gamma_beta, ones, zeros, **hints
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        beta = printer._print(self.args[0])
        return Rf"\boldsymbol{{B_z}}\left({beta}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
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
class RotationYMatrix(UnevaluatedExpression):
    r"""Rotation matrix around the :math:`y`-axis for a `FourMomentumSymbol`.

    Args:
        angle: Angle with which to rotate, see e.g. `Phi` and `Theta`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`angle`.
    """

    def __new__(
        cls, angle: sp.Expr, n_events: Optional[sp.Symbol] = None, **hints: Any
    ) -> "RotationYMatrix":
        if n_events is None:
            n_events = _ArraySize(angle)
        return create_expression(cls, angle, n_events, **hints)

    def as_explicit(self) -> sp.Expr:
        angle = self.args[0]
        return sp.Matrix(
            [
                [1, 0, 0, 0],
                [0, sp.cos(angle), 0, sp.sin(angle)],
                [0, 0, 1, 0],
                [0, -sp.sin(angle), 0, sp.cos(angle)],
            ]
        )

    def evaluate(self) -> "_RotationYMatrixImplementation":
        angle = self.args[0]
        n_events = self.args[1]
        return _RotationYMatrixImplementation(
            angle=angle,
            cos_angle=sp.cos(angle),
            sin_angle=sp.sin(angle),
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        return printer._print(self.evaluate(), *args)


class _RotationYMatrixImplementation(NumPyPrintable):
    def __new__(  # pylint: disable=too-many-arguments
        cls,
        angle: sp.Expr,
        cos_angle: sp.Expr,
        sin_angle: sp.Expr,
        ones: "_OnesArray",
        zeros: "_ZerosArray",
        **hints: Any,
    ) -> "_RotationYMatrixImplementation":
        return create_expression(
            cls, angle, cos_angle, sin_angle, ones, zeros, **hints
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angle, *_ = self.args
        angle = printer._print(angle)
        return Rf"\boldsymbol{{R_y}}\left({angle}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
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
    r"""Rotation matrix around the :math:`z`-axis for a `FourMomentumSymbol`.

    Args:
        angle: Angle with which to rotate, see e.g. `Phi` and `Theta`.
        n_events: Number of events :math:`n` for this matrix array of shape
            :math:`n\times4\times4`. Defaults to the `len` of :code:`angle`.
    """

    def __new__(
        cls, angle: sp.Expr, n_events: Optional[sp.Symbol] = None, **hints: Any
    ) -> "RotationZMatrix":
        if n_events is None:
            n_events = _ArraySize(angle)
        return create_expression(cls, angle, n_events, **hints)

    def as_explicit(self) -> sp.Expr:
        angle = self.args[0]
        return sp.Matrix(
            [
                [1, 0, 0, 0],
                [0, sp.cos(angle), -sp.sin(angle), 0],
                [0, sp.sin(angle), sp.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )

    def evaluate(self) -> "_RotationZMatrixImplementation":
        angle = self.args[0]
        n_events = self.args[1]
        return _RotationZMatrixImplementation(
            angle=angle,
            cos_angle=sp.cos(angle),
            sin_angle=sp.sin(angle),
            ones=_OnesArray(n_events),
            zeros=_ZerosArray(n_events),
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        return printer._print(self.evaluate(), *args)


class _RotationZMatrixImplementation(NumPyPrintable):
    def __new__(  # pylint: disable=too-many-arguments
        cls,
        angle: sp.Expr,
        cos_angle: sp.Expr,
        sin_angle: sp.Expr,
        ones: "_OnesArray",
        zeros: "_ZerosArray",
        **hints: Any,
    ) -> "_RotationZMatrixImplementation":
        return create_expression(
            cls, angle, cos_angle, sin_angle, ones, zeros, **hints
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angle, *_ = self.args
        angle = printer._print(angle)
        return Rf"\boldsymbol{{R_z}}\left({angle}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
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
    def __new__(
        cls, shape: Union[int, Sequence[int]], **kwargs: Any
    ) -> "_OnesArray":
        return create_expression(cls, shape, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("ones")
        shape = printer._print(self.args[0])
        return f"ones({shape})"


class _ZerosArray(NumPyPrintable):
    def __new__(
        cls, shape: Union[int, Sequence[int]], **kwargs: Any
    ) -> "_ZerosArray":
        return create_expression(cls, shape, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("zeros")
        shape = printer._print(self.args[0])
        return f"zeros({shape})"


class _ArraySize(NumPyPrintable):
    def __new__(cls, array: sp.Basic, **kwargs: Any) -> "_ArraySize":
        return create_expression(cls, array, **kwargs)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        shape = printer._print(self.args[0])
        return f"len({shape})"


def compute_helicity_angles(
    four_momenta: "FourMomenta", topology: Topology
) -> Dict[str, sp.Expr]:
    """Formulate expressions for all helicity angles in a topology.

    Formulate expressions (`~sympy.core.expr.Expr`) for all helicity angles
    appearing in a given `~qrules.topology.Topology`. The expressions are given
    in terms of `FourMomenta` The expressions returned as values in a
    `dict`, where the keys are defined by :func:`get_helicity_angle_label`.

    Example
    -------
    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> topology = topologies[0]
    >>> four_momenta = create_four_momentum_symbols(topology)
    >>> angles = compute_helicity_angles(four_momenta, topology)
    >>> angles["theta_1+2"]
    Theta(p1 + p2)
    """
    if topology.outgoing_edge_ids != set(four_momenta):
        raise ValueError(
            f"Momentum IDs {set(four_momenta)} do not match "
            f"final state edge IDs {set(topology.outgoing_edge_ids)}"
        )

    n_events = _get_number_of_events(four_momenta)

    def __recursive_helicity_angles(  # pylint: disable=too-many-locals
        four_momenta: FourMomenta, node_id: int
    ) -> Dict[str, sp.Expr]:
        helicity_angles: Dict[str, sp.Expr] = {}
        child_state_ids = sorted(
            topology.get_edge_ids_outgoing_from_node(node_id)
        )
        if all(
            topology.edges[i].ending_node_id is None for i in child_state_ids
        ):
            state_id = child_state_ids[0]
            four_momentum = four_momenta[state_id]
            phi_label, theta_label = get_helicity_angle_label(
                topology, state_id
            )
            helicity_angles[phi_label] = Phi(four_momentum)
            helicity_angles[theta_label] = Theta(four_momentum)
        for state_id in child_state_ids:
            edge = topology.edges[state_id]
            if edge.ending_node_id is not None:
                # recursively determine all momenta ids in the list
                sub_momenta_ids = determine_attached_final_state(
                    topology, state_id
                )
                if len(sub_momenta_ids) > 1:
                    # add all of these momenta together -> defines new subsystem
                    four_momentum = ArraySum(
                        *[four_momenta[i] for i in sub_momenta_ids]
                    )

                    # boost all of those momenta into this new subsystem
                    phi = Phi(four_momentum)
                    theta = Theta(four_momentum)
                    p3_norm = ThreeMomentumNorm(four_momentum)
                    beta = p3_norm / Energy(four_momentum)
                    new_momentum_pool = {
                        k: ArrayMultiplication(
                            BoostZMatrix(beta, n_events),
                            RotationYMatrix(-theta, n_events),
                            RotationZMatrix(-phi, n_events),
                            p,
                        )
                        for k, p in four_momenta.items()
                        if k in sub_momenta_ids
                    }

                    # register current angle variables
                    phi_label, theta_label = get_helicity_angle_label(
                        topology, state_id
                    )
                    helicity_angles[phi_label] = Phi(four_momentum)
                    helicity_angles[theta_label] = Theta(four_momentum)

                    # call next recursion
                    angles = __recursive_helicity_angles(
                        new_momentum_pool,
                        edge.ending_node_id,
                    )
                    helicity_angles.update(angles)

        return helicity_angles

    initial_state_id = next(iter(topology.incoming_edge_ids))
    initial_state_edge = topology.edges[initial_state_id]
    assert initial_state_edge.ending_node_id is not None
    return __recursive_helicity_angles(
        four_momenta, initial_state_edge.ending_node_id
    )


def _get_number_of_events(
    four_momenta: "FourMomenta",
) -> "_ArraySize":
    sorted_momentum_symbols = sorted(four_momenta.values(), key=str)
    return _ArraySize(sorted_momentum_symbols[0])


def compute_invariant_masses(
    four_momenta: "FourMomenta", topology: Topology
) -> Dict[str, sp.Expr]:
    """Compute the invariant masses for all final state combinations."""
    if topology.outgoing_edge_ids != set(four_momenta):
        raise ValueError(
            f"Momentum IDs {set(four_momenta)} do not match "
            f"final state edge IDs {set(topology.outgoing_edge_ids)}"
        )
    invariant_masses = {}
    for state_id in topology.edges:
        attached_state_ids = determine_attached_final_state(topology, state_id)
        total_momentum = ArraySum(
            *[four_momenta[i] for i in attached_state_ids]
        )
        invariant_mass = InvariantMass(total_momentum)
        name = get_invariant_mass_label(topology, state_id)
        invariant_masses[name] = invariant_mass
    return invariant_masses


def get_helicity_angle_label(
    topology: Topology, state_id: int
) -> Tuple[str, str]:
    """Generate labels that can be used to identify helicity angles.

    The generated subscripts describe the decay sequence from the right to the
    left, separated by commas. Resonance edge IDs are expressed as a sum of the
    final state IDs that lie below them (see
    :func:`.determine_attached_final_state`). The generated label does not
    state the top-most edge (the initial state).

    Example
    -------
    The following two allowed isobar topologies for a **1-to-5-body** decay
    illustrates how the naming scheme results in a unique label for each of the
    **eight edges** in the decay topology. Note that label only uses final
    state IDs, but still reflects the internal decay topology.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> topology = topologies[0]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     phi_label, theta_label = get_helicity_angle_label(topology, i)
    ...     print(f"{i}: '{phi_label}'")
    0: 'phi_0,0+3+4'
    1: 'phi_1,1+2'
    2: 'phi_2,1+2'
    3: 'phi_3,3+4,0+3+4'
    4: 'phi_4,3+4,0+3+4'
    5: 'phi_0+3+4'
    6: 'phi_1+2'
    7: 'phi_3+4,0+3+4'
    >>> topology = topologies[1]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     phi_label, theta_label = get_helicity_angle_label(topology, i)
    ...     print(f"{i}: '{phi_label}'")
    0: 'phi_0,0+1'
    1: 'phi_1,0+1'
    2: 'phi_2,2+3+4'
    3: 'phi_3,3+4,2+3+4'
    4: 'phi_4,3+4,2+3+4'
    5: 'phi_0+1'
    6: 'phi_2+3+4'
    7: 'phi_3+4,2+3+4'

    Some labels explained:

    - :code:`phi_1+2`: **edge 6** on the *left* topology, because for this
      topology, we have :math:`p_6=p_1+p_2`.
    - :code:`phi_2+3+4`: **edge 6** *right*, because for this topology,
      :math:`p_6=p_2+p_3+p_4`.
    - :code:`phi_1,1+2`: **edge 1** *left*, because 1 decays from
      :math:`p_6=p_1+p_2`.
    - :code:`phi_1,0+1`: **edge 1** *right*, because it decays from
      :math:`p_5=p_0+p_1`.
    - :code:`phi_4,3+4,2+3+4`: **edge 4** *right*, because it decays from edge
      7 (:math:`p_7=p_3+p_4`), which comes from edge 6
      (:math:`p_7=p_2+p_3+p_4`).

    As noted, the top-most parent (initial state) is not listed in the label.
    """
    _assert_isobar_topology(topology)

    def recursive_label(topology: Topology, state_id: int) -> str:
        edge = topology.edges[state_id]
        if edge.ending_node_id is None:
            label = f"{state_id}"
        else:
            attached_final_state_ids = determine_attached_final_state(
                topology, state_id
            )
            label = "+".join(map(str, attached_final_state_ids))
        if edge.originating_node_id is not None:
            incoming_state_ids = topology.get_edge_ids_ingoing_to_node(
                edge.originating_node_id
            )
            state_id = next(iter(incoming_state_ids))
            if state_id not in topology.incoming_edge_ids:
                label += f",{recursive_label(topology, state_id)}"
        return label

    label = recursive_label(topology, state_id)
    return f"phi_{label}", f"theta_{label}"


def get_invariant_mass_label(topology: Topology, state_id: int) -> str:
    """Generate an invariant mass label for a state (edge on a topology).

    Example
    -------
    In the case shown in Figure :ref:`one-to-five-topology-0`, the invariant
    mass of state :math:`5` is :math:`m_{034}`, because
    :math:`p_5=p_0+p_3+p_4`:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> get_invariant_mass_label(topologies[0], state_id=5)
    'm_034'

    Naturally, the 'invariant' mass label for a final state is just the mass of the
    state itself:

    >>> get_invariant_mass_label(topologies[0], state_id=1)
    'm_1'
    """
    final_state_ids = determine_attached_final_state(topology, state_id)
    return f"m_{''.join(map(str, sorted(final_state_ids)))}"


def _assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        _assert_two_body_decay(topology, node_id)


def _assert_two_body_decay(topology: Topology, node_id: int) -> None:
    parent_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
    if len(parent_state_ids) != 1:
        raise ValueError(
            f"Node {node_id} has {len(parent_state_ids)} parent states,"
            " so this is not an isobar decay"
        )
    child_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
    if len(child_state_ids) != 2:
        raise ValueError(
            f"Node {node_id} decays to {len(child_state_ids)} states,"
            " so this is not an isobar decay"
        )


def determine_attached_final_state(
    topology: Topology, state_id: int
) -> List[int]:
    """Determine all final state particles of a transition.

    These are attached downward (forward in time) for a given edge (resembling
    the root).

    Example
    -------
    For **edge 5** in Figure :ref:`one-to-five-topology-0`, we get:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> determine_attached_final_state(topologies[0], state_id=5)
    [0, 3, 4]
    """
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return [state_id]
    return sorted(
        topology.get_originating_final_state_edge_ids(edge.ending_node_id)
    )
