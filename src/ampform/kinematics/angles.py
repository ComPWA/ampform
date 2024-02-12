"""Angle computations for (boosted) :mod:`.lorentz` vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import sympy as sp

from ampform.helicity.decay import (
    determine_attached_final_state,
    get_sibling_state_id,
    is_opposite_helicity_state,
)
from ampform.helicity.naming import get_helicity_angle_symbols, get_helicity_suffix
from ampform.kinematics.lorentz import (
    ArraySize,
    BoostMatrix,
    BoostZMatrix,
    Energy,
    FourMomenta,
    FourMomentumX,
    FourMomentumY,
    FourMomentumZ,
    NegativeMomentum,
    RotationYMatrix,
    RotationZMatrix,
    compute_boost_chain,
    three_momentum_norm,
)
from ampform.sympy import unevaluated
from ampform.sympy._array_expressions import (
    ArrayMultiplication,
    ArraySlice,
    ArraySum,
    MatrixMultiplication,
)

if TYPE_CHECKING:
    from qrules.topology import Topology


@unevaluated
class Phi(sp.Expr):
    r"""Azimuthal angle :math:`\phi` of a `.FourMomentumSymbol`."""

    momentum: sp.Basic
    _latex_repr_ = R"\phi\left({momentum}\right)"

    def evaluate(self) -> sp.Expr:
        p = self.momentum
        return sp.atan2(FourMomentumY(p), FourMomentumX(p))


@unevaluated
class Theta(sp.Expr):
    r"""Polar (elevation) angle :math:`\theta` of a `.FourMomentumSymbol`."""

    momentum: sp.Basic
    _latex_repr_ = R"\theta\left({momentum}\right)"

    def evaluate(self) -> sp.Expr:
        p = self.momentum
        return sp.acos(FourMomentumZ(p) / three_momentum_norm(p))


def compute_helicity_angles(
    four_momenta: Mapping[int, sp.Expr], topology: Topology
) -> dict[sp.Symbol, sp.Expr]:
    """Formulate expressions for all helicity angles in a topology.

    Formulate expressions (`~sympy.core.expr.Expr`) for all helicity angles appearing in
    a given `~qrules.topology.Topology`. The expressions are given in terms of
    `.FourMomenta` The expressions returned as values in a `dict`, where the keys are
    defined by :func:`.get_helicity_angle_symbols`.

    Example
    -------
    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> topology = topologies[0]
    >>> from ampform.kinematics.lorentz import create_four_momentum_symbols
    >>> four_momenta = create_four_momentum_symbols(topology)
    >>> angles = compute_helicity_angles(four_momenta, topology)
    >>> theta_symbol = sp.Symbol("theta_0", real=True)
    >>> angles[theta_symbol]
    Theta(p1 + p2)
    """
    if topology.outgoing_edge_ids != set(four_momenta):
        msg = (
            f"Momentum IDs {set(four_momenta)} do not match final state edge IDs"
            f" {set(topology.outgoing_edge_ids)}"
        )
        raise ValueError(msg)

    n_events = _get_number_of_events(four_momenta)

    def __recursive_helicity_angles(
        four_momenta: Mapping[int, sp.Expr], node_id: int
    ) -> dict[sp.Symbol, sp.Expr]:
        helicity_angles: dict[sp.Symbol, sp.Expr] = {}
        child_state_ids = sorted(topology.get_edge_ids_outgoing_from_node(node_id))
        if all(topology.edges[i].ending_node_id is None for i in child_state_ids):
            state_id = child_state_ids[0]
            if is_opposite_helicity_state(topology, state_id):
                state_id = child_state_ids[1]
            four_momentum: sp.Expr = four_momenta[state_id]
            phi, theta = get_helicity_angle_symbols(topology, state_id)
            helicity_angles[phi] = Phi(four_momentum)
            helicity_angles[theta] = Theta(four_momentum)
        for state_id in child_state_ids:
            edge = topology.edges[state_id]
            if edge.ending_node_id is not None:
                # recursively determine all momenta ids in the list
                sub_momenta_ids = determine_attached_final_state(topology, state_id)
                if len(sub_momenta_ids) > 1:
                    # add all of these momenta together -> defines new subsystem
                    four_momentum = ArraySum(*[
                        four_momenta[i] for i in sub_momenta_ids
                    ])

                    # boost all of those momenta into this new subsystem
                    phi_expr = Phi(four_momentum)
                    theta_expr = Theta(four_momentum)
                    p3_norm = three_momentum_norm(four_momentum)
                    beta = p3_norm / Energy(four_momentum)
                    new_momentum_pool: dict[int, sp.Expr] = {
                        k: ArrayMultiplication(
                            BoostZMatrix(beta, n_events),
                            RotationYMatrix(-theta_expr, n_events),
                            RotationZMatrix(-phi_expr, n_events),
                            p,
                        )
                        for k, p in four_momenta.items()
                        if k in sub_momenta_ids
                    }

                    # register current angle variables
                    if is_opposite_helicity_state(topology, state_id):
                        state_id = get_sibling_state_id(topology, state_id)
                    phi, theta = get_helicity_angle_symbols(topology, state_id)
                    helicity_angles[phi] = Phi(four_momentum)
                    helicity_angles[theta] = Theta(four_momentum)

                    # call next recursion
                    angles = __recursive_helicity_angles(
                        new_momentum_pool,
                        edge.ending_node_id,
                    )
                    helicity_angles.update(angles)

        return helicity_angles

    initial_state_id = next(iter(topology.incoming_edge_ids))
    initial_state_edge = topology.edges[initial_state_id]
    if initial_state_edge.ending_node_id is None:
        msg = "Edge does not end in a node"
        raise ValueError(msg)
    return __recursive_helicity_angles(four_momenta, initial_state_edge.ending_node_id)


def _get_number_of_events(four_momenta: Mapping[int, sp.Expr]) -> ArraySize:
    sorted_momentum_symbols = sorted(four_momenta.values(), key=str)
    return ArraySize(sorted_momentum_symbols[0])


def compute_wigner_angles(
    topology: Topology, momenta: FourMomenta, state_id: int
) -> dict[sp.Symbol, sp.Expr]:
    """Create an `~sympy.core.expr.Expr` for each angle in a Wigner rotation.

    Implementation of (B.2-4) in :cite:`marangottoHelicityAmplitudesGeneric2020`, with
    :math:`x'_z` etc. taken from the result of :func:`compute_wigner_rotation_matrix`.
    See also `Wigner rotations <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
    """
    wigner_rotation_matrix = compute_wigner_rotation_matrix(topology, momenta, state_id)
    x_z = ArraySlice(wigner_rotation_matrix, (slice(None), 1, 3))
    y_z = ArraySlice(wigner_rotation_matrix, (slice(None), 2, 3))
    z_x = ArraySlice(wigner_rotation_matrix, (slice(None), 3, 1))
    z_y = ArraySlice(wigner_rotation_matrix, (slice(None), 3, 2))
    z_z = ArraySlice(wigner_rotation_matrix, (slice(None), 3, 3))
    suffix = get_helicity_suffix(topology, state_id)
    alpha, beta, gamma = sp.symbols(
        f"alpha{suffix} beta{suffix} gamma{suffix}", real=True
    )
    return {
        alpha: sp.atan2(z_y, z_x),
        beta: sp.acos(z_z),
        gamma: sp.atan2(y_z, -x_z),
    }


def compute_wigner_rotation_matrix(
    topology: Topology, momenta: FourMomenta, state_id: int
) -> MatrixMultiplication:
    """Compute a Wigner rotation matrix.

    Implementation of Eq. (36) in :cite:`marangottoHelicityAmplitudesGeneric2020`. See
    also `Wigner rotations <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
    """
    momentum = momenta[state_id]
    inverted_direct_boost = BoostMatrix(NegativeMomentum(momentum))
    boost_chain = compute_boost_chain(topology, momenta, state_id)
    return MatrixMultiplication(inverted_direct_boost, *boost_chain)
