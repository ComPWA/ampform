# cspell:ignore einsum
# pylint: disable=arguments-differ,protected-access,unused-argument
"""Kinematics of an amplitude model in the helicity formalism."""

import functools
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import attr
import sympy as sp
from attr.validators import instance_of
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition
from sympy.printing.conventions import split_super_sub
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.printer import Printer

from ampform.sympy import (
    UnevaluatedExpression,
    create_expression,
    implement_doit_method,
    make_commutative,
)
from ampform.sympy._array_expressions import ArraySlice, ArraySymbol
from ampform.sympy.math import ComplexSqrt

FourMomentumSymbols = Dict[int, ArraySymbol]


# for numpy broadcasting
ArraySlice = make_commutative()(ArraySlice)  # type: ignore[misc]


@attr.s(on_setattr=attr.setters.frozen)
class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.create_expressions` method forms the bridge between four-momentum
    data for the decay you are studying and the kinematic variables that are in
    the `.HelicityModel`. These are invariant mass (see
    :func:`.get_invariant_mass_label`) and the :math:`\theta` and :math:`\phi`
    helicity angles (see :func:`.get_helicity_angle_label`).
    """

    reaction_info: ReactionInfo = attr.ib(validator=instance_of(ReactionInfo))
    registered_topologies: Set[Topology] = attr.ib(
        factory=set, init=False, repr=False
    )

    def register_transition(self, transition: StateTransition) -> None:
        if set(self.reaction_info.initial_state) != set(
            transition.initial_states
        ):
            raise ValueError("Transition has mismatching initial state IDs")
        if set(self.reaction_info.final_state) != set(transition.final_states):
            raise ValueError("Transition has mismatching final state IDs")
        for state_id in self.reaction_info.final_state:
            particle = self.reaction_info.initial_state[state_id]
            state = transition.initial_states[state_id]
            if particle != state.particle:
                raise ValueError(
                    "Transition has different initial particle at"
                    f" {state_id}.",
                    f" Expecting: {particle.name}"
                    f" In added transition: {state.particle.name}",
                )
        self.register_topology(transition.topology)

    def register_topology(self, topology: Topology) -> None:
        _assert_isobar_topology(topology)
        if len(self.registered_topologies) == 0:
            object.__setattr__(
                self,
                "final_state_ids",
                tuple(sorted(topology.outgoing_edge_ids)),
            )
        if len(topology.incoming_edge_ids) != 1:
            raise ValueError(
                f"Topology has {len(topology.incoming_edge_ids)} incoming"
                " edges, so is not isobar"
            )
        if len(self.registered_topologies) != 0:
            existing_topology = next(iter(self.registered_topologies))
            if (
                (
                    topology.incoming_edge_ids
                    != existing_topology.incoming_edge_ids
                )
                or (
                    topology.outgoing_edge_ids
                    != existing_topology.outgoing_edge_ids
                )
                or (
                    topology.outgoing_edge_ids
                    != existing_topology.outgoing_edge_ids
                )
                or (topology.nodes != existing_topology.nodes)
            ):
                raise ValueError("Edge or node IDs of topology do not match")
        self.registered_topologies.add(topology)

    def permutate_registered_topologies(self) -> None:
        """Register permutations of all `registered_topologies`.

        See :ref:`usage/amplitude:Extend kinematic variables`.
        """
        for topology in set(self.registered_topologies):
            final_state_ids = topology.outgoing_edge_ids
            for permutation in itertools.permutations(final_state_ids):
                id_mapping = dict(zip(topology.outgoing_edge_ids, permutation))
                permuted_topology = attr.evolve(
                    topology,
                    edges={
                        id_mapping.get(i, i): edge
                        for i, edge in topology.edges.items()
                    },
                )
                self.register_topology(permuted_topology)

    def create_expressions(self) -> Dict[str, sp.Expr]:
        output = {}
        for topology in self.registered_topologies:
            four_momenta = create_four_momentum_symbols(topology)
            output.update(compute_helicity_angles(four_momenta, topology))
            output.update(compute_invariant_masses(four_momenta, topology))
        return output


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


def compute_helicity_angles(
    four_momenta: FourMomentumSymbols, topology: Topology
) -> Dict[str, sp.Expr]:
    if topology.outgoing_edge_ids != set(four_momenta):
        raise ValueError(
            f"Momentum IDs {set(four_momenta)} do not match "
            f"final state edge IDs {set(topology.outgoing_edge_ids)}"
        )

    def __recursive_helicity_angles(  # pylint: disable=too-many-locals
        four_momenta: FourMomentumSymbols, node_id: int
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
                            BoostZ(beta),
                            RotationY(-theta),
                            RotationZ(-phi),
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


def compute_invariant_masses(
    four_momenta: FourMomentumSymbols, topology: Topology
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


class ArraySum(sp.Expr):
    precedence = PRECEDENCE["Add"]
    terms: Tuple[sp.Basic, ...] = property(lambda self: self.args)  # type: ignore[assignment]

    def __new__(cls, *terms: sp.Basic, **hints: Any) -> "Energy":
        return create_expression(cls, *terms, **hints)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        if all(
            map(lambda i: isinstance(i, (sp.Symbol, ArraySymbol)), self.terms)
        ):
            names = set(map(_strip_subscript_superscript, self.terms))
            if len(names) == 1:
                name = next(iter(names))
                subscript = "".join(map(_get_subscript, self.terms))
                return f"{{{name}}}_{{{subscript}}}"
        return printer._print_ArraySum(self)


def _print_array_sum(self: Printer, expr: ArraySum) -> str:
    terms = map(self._print, expr.terms)
    return " + ".join(terms)


Printer._print_ArraySum = _print_array_sum


def _get_subscript(symbol: sp.Symbol) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _get_subscript(sp.Symbol("p1"))
    '1'
    >>> _get_subscript(sp.Symbol("p^2_{0,0}"))
    '0,0'
    """
    if isinstance(symbol, sp.Basic):
        text = sp.latex(symbol)
    else:
        text = symbol
    _, _, subscripts = split_super_sub(text)
    stripped_subscripts: Iterable[str] = map(
        lambda s: s.strip("{").strip("}"), subscripts
    )
    return " ".join(stripped_subscripts)


def _strip_subscript_superscript(symbol: sp.Symbol) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _strip_subscript_superscript(sp.Symbol("p1"))
    'p'
    >>> _strip_subscript_superscript(sp.Symbol("p^2_{0,0}"))
    'p'
    """
    if isinstance(symbol, sp.Basic):
        text = sp.latex(symbol)
    else:
        text = symbol
    name, _, _ = split_super_sub(text)
    return name


@make_commutative()
class ArrayAxisSum(sp.Expr):
    array: ArraySymbol = property(lambda self: self.args[0])
    axis: Optional[int] = property(lambda self: self.args[1])  # type: ignore[assignment]

    def __new__(
        cls, array: ArraySymbol, axis: Optional[int] = None, **hints: Any
    ) -> "ArrayAxisSum":
        if axis is not None and not isinstance(axis, (int, sp.Integer)):
            raise TypeError("Only single digits allowed for axis")
        return create_expression(cls, array, axis, **hints)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        array = printer._print(self.array)
        if self.axis is None:
            return fR"\sum{{{array}}}"
        axis = printer._print(self.axis)
        return fR"\sum_{{\mathrm{{axis{axis}}}}}{{{array}}}"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("sum")
        array = printer._print(self.array)
        axis = printer._print(self.axis)
        return f"sum({array}, axis={axis})"


class ArrayMultiplication(sp.Expr):
    tensors: List[sp.Expr] = property(lambda self: self.args)  # type: ignore[assignment]

    def __new__(cls, *tensors: sp.Expr, **hints: Any) -> "ArrayMultiplication":
        return create_expression(cls, *tensors, **hints)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        tensors = map(printer._print, self.tensors)
        return " ".join(tensors)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        def multiply(matrix: sp.Expr, vector: sp.Expr) -> str:
            return f'einsum("...ij,...j->...i", {matrix}, {vector})'

        def recursive_multiply(tensors: Sequence[sp.Expr]) -> str:
            if len(tensors) < 2:
                raise ValueError("Need at least two tensors")
            if len(tensors) == 2:
                return multiply(tensors[0], tensors[1])
            return multiply(tensors[0], recursive_multiply(tensors[1:]))

        printer.module_imports[printer._module].update({"einsum", "transpose"})
        tensors = list(map(printer._print, self.args))
        if len(tensors) == 0:
            return ""
        if len(tensors) == 1:
            return tensors[0]
        return recursive_multiply(tensors)


class BoostZ(sp.Expr):
    beta: sp.Expr = property(lambda self: self.args[0])

    def __new__(cls, beta: sp.Expr, **kwargs: Any) -> "BoostZ":
        return create_expression(cls, beta, **kwargs)

    def as_explicit(self) -> sp.Expr:
        beta = self.beta
        gamma = 1 / sp.sqrt(1 - beta ** 2)
        return sp.Matrix(
            [
                [gamma, 0, 0, -gamma * beta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-gamma * beta, 0, 0, gamma],
            ]
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        beta = printer._print(self.beta)
        return fR"\boldsymbol{{B_z}}\left({beta}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].update(
            {"array", "ones", "zeros", "sqrt"}
        )
        beta = printer._print(self.beta)
        gamma = f"1 / sqrt(1 - ({beta}) ** 2)"
        n_events = f"len({beta})"
        zeros = f"zeros({n_events})"
        ones = f"ones({n_events})"
        return f"""array(
            [
                [{gamma}, {zeros}, {zeros}, -{gamma} * {beta}],
                [{zeros}, {ones}, {zeros}, {zeros}],
                [{zeros}, {zeros}, {ones}, {zeros}],
                [-{gamma} * {beta}, {zeros}, {zeros}, {gamma}],
            ]
        ).transpose((2, 0, 1))"""


class RotationY(sp.Expr):
    angle: sp.Expr = property(lambda self: self.args[0])

    def __new__(cls, angle: sp.Expr, **hints: Any) -> "RotationY":
        return create_expression(cls, angle, **hints)

    def as_explicit(self) -> sp.Expr:
        angle = self.angle
        return sp.Matrix(
            [
                [1, 0, 0, 0],
                [0, sp.cos(angle), 0, sp.sin(angle)],
                [0, 0, 1, 0],
                [0, -sp.sin(angle), 0, sp.cos(angle)],
            ]
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angle, *_ = self.args
        angle = printer._print(angle)
        return fR"\boldsymbol{{R_y}}\left({angle}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].update(
            {"array", "cos", "ones", "zeros", "sin"}
        )
        angle = printer._print(self.angle)
        n_events = f"len({angle})"
        zeros = f"zeros({n_events})"
        ones = f"ones({n_events})"
        return f"""array(
            [
                [{ones}, {zeros}, {zeros}, {zeros}],
                [{zeros}, cos({angle}), {zeros}, sin({angle})],
                [{zeros}, {zeros}, {ones}, {zeros}],
                [{zeros}, -sin({angle}), {zeros}, cos({angle})],
            ]
        ).transpose((2, 0, 1))"""


class RotationZ(sp.Expr):
    angle: sp.Expr = property(lambda self: self.args[0])

    def __new__(cls, angle: sp.Symbol, **hints: Any) -> "RotationZ":
        return create_expression(cls, angle, **hints)

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

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angle, *_ = self.args
        angle = printer._print(angle)
        return fR"\boldsymbol{{R_z}}\left({angle}\right)"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].update(
            {"array", "cos", "ones", "zeros", "sin"}
        )
        angle = printer._print(self.angle)
        n_events = f"len({angle})"
        zeros = f"zeros({n_events})"
        ones = f"ones({n_events})"
        return f"""array(
            [
                [{ones}, {zeros}, {zeros}, {zeros}],
                [{zeros}, cos({angle}), -sin({angle}), {zeros}],
                [{zeros}, sin({angle}), cos({angle}), {zeros}],
                [{zeros}, {zeros}, {zeros}, {ones}],
            ]
        ).transpose((2, 0, 1))"""


class HasMomentum:
    # pylint: disable=no-member
    momentum: ArraySymbol = property(lambda self: self.args[0])


def implement_latex_subscript(
    subscript: str,
) -> Callable[[Type[UnevaluatedExpression]], Type[UnevaluatedExpression]]:
    def decorator(
        decorated_class: Type[UnevaluatedExpression],
    ) -> Type[UnevaluatedExpression]:
        @functools.wraps(decorated_class.doit)
        def _latex(
            self: HasMomentum, printer: LatexPrinter, *args: Any
        ) -> str:
            momentum = printer._print(self.momentum)
            if printer._needs_mul_brackets(self.momentum):
                momentum = fR"\left({momentum}\right)"
            else:
                momentum = fR"{{{momentum}}}"
            return f"{momentum}_{subscript}"

        decorated_class._latex = _latex  # type: ignore[assignment]
        return decorated_class

    return decorator


@implement_doit_method
@make_commutative()
class Energy(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "Energy":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self.momentum, (slice(None), 0))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self.momentum)
        return fR"E\left({momentum}\right)"


@implement_latex_subscript(subscript="x")
@implement_doit_method
@make_commutative()
class FourMomentumX(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "FourMomentumX":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self.momentum, (slice(None), 1))


@implement_latex_subscript(subscript="y")
@implement_doit_method
@make_commutative()
class FourMomentumY(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "FourMomentumY":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self.momentum, (slice(None), 2))


@implement_latex_subscript(subscript="z")
@implement_doit_method
@make_commutative()
class FourMomentumZ(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "FourMomentumZ":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        return ArraySlice(self.momentum, (slice(None), 3))


@implement_doit_method
@make_commutative()
class ThreeMomentumNorm(HasMomentum, UnevaluatedExpression):
    def __new__(
        cls, momentum: ArraySymbol, **hints: Any
    ) -> "ThreeMomentumNorm":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        three_momentum = ArraySlice(
            self.momentum, (slice(None), slice(1, None))
        )
        norm_squared = ArrayAxisSum(three_momentum ** 2, axis=1)
        return sp.sqrt(norm_squared)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self.momentum)
        return fR"\left|\vec{{{momentum}}}\right|"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        return printer._print(self.evaluate())


@implement_doit_method
@make_commutative()
class InvariantMass(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "Energy":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> ArraySlice:
        p = self.momentum
        return ComplexSqrt(Energy(p) ** 2 - ThreeMomentumNorm(p) ** 2)

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self.momentum)
        return f"m_{{{momentum}}}"


@implement_doit_method
@make_commutative()
class Phi(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "Phi":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> sp.Expr:
        p = self.momentum
        return sp.atan2(FourMomentumY(p), FourMomentumX(p))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self.momentum)
        return fR"\phi\left({momentum}\right)"


@implement_doit_method
@make_commutative()
class Theta(HasMomentum, UnevaluatedExpression):
    def __new__(cls, momentum: ArraySymbol, **hints: Any) -> "Theta":
        return create_expression(cls, momentum, **hints)

    def evaluate(self) -> sp.Expr:
        p = self.momentum
        return sp.acos(FourMomentumZ(p) / ThreeMomentumNorm(p))

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        momentum = printer._print(self.momentum)
        return fR"\theta\left({momentum}\right)"


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


def create_four_momentum_symbols(topology: Topology) -> FourMomentumSymbols:
    n_final_states = len(topology.outgoing_edge_ids)
    return {i: ArraySymbol(f"p{i}") for i in range(n_final_states)}


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
