"""Angle computations for (boosted) :mod:`.lorentz` vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from ampform.kinematics.phasespace import Kallen
from ampform.sympy import unevaluated
from ampform.sympy._array_expressions import (
    ArrayMultiplication,
    ArraySlice,
    ArraySum,
    MatrixMultiplication,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    Implementation of (B.2-4) in :cite:`Marangotto:2019ucc`, with :math:`x'_z` etc.
    taken from the result of :func:`compute_wigner_rotation_matrix`. See also `Wigner
    rotations <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
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

    Implementation of Eq. (36) in :cite:`Marangotto:2019ucc`. See also `Wigner rotations
    <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
    """
    momentum = momenta[state_id]
    inverted_direct_boost = BoostMatrix(NegativeMomentum(momentum))
    boost_chain = compute_boost_chain(topology, momenta, state_id)
    return MatrixMultiplication(inverted_direct_boost, *boost_chain)


def formulate_scattering_angle(
    state_id: int, sibling_id: int
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate the scattering angle in the rest frame of the resonance.

    Compute the :math:`\theta_{ij}` scattering angle as formulated in `Eq (A1) in the
    DPD paper <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.101.034033#page=9>`_
    :cite:`Marangotto:2019ucc`. The angle is that between particle :math:`i` and
    spectator particle :math:`k` in the rest frame of the isobar resonance :math:`(ij)`.
    """
    if not {state_id, sibling_id} <= {1, 2, 3}:
        msg = "Child IDs need to be one of 1, 2, 3"
        raise ValueError(msg)
    if {state_id, sibling_id} in {(2, 1), (3, 2), (1, 3)}:
        msg = f"Cannot compute scattering angle θ{state_id}{sibling_id}"
        raise NotImplementedError(msg)
    if state_id == sibling_id:
        msg = f"IDs of the decay products cannot be equal: {state_id}"
        raise ValueError(msg)
    symbol = sp.Symbol(f"theta_{state_id}{sibling_id}", real=True)
    spectator_id = next(iter({1, 2, 3} - {state_id, sibling_id}))
    m0 = sp.Symbol("m_0", nonnegative=True)
    mi = sp.Symbol(f"m_{state_id}", nonnegative=True)
    mj = sp.Symbol(f"m_{sibling_id}", nonnegative=True)
    mk = sp.Symbol(f"m_{spectator_id}", nonnegative=True)
    sj = sp.Symbol(f"m_{__get_id_complement(sibling_id)}", nonnegative=True) ** 2
    sk = sp.Symbol(f"m_{__get_id_complement(spectator_id)}", nonnegative=True) ** 2
    theta = sp.acos(
        (2 * sk * (sj - mk**2 - mi**2) - (sk + mi**2 - mj**2) * (m0**2 - sk - mk**2))
        / (sp.sqrt(Kallen(m0**2, mk**2, sk)) * sp.sqrt(Kallen(sk, mi**2, mj**2)))
    )
    return symbol, theta


def formulate_theta_hat_angle(
    isobar_id: int, aligned_subsystem: int
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate an expression for :math:`\hat\theta_{i(j)}`."""
    allowed_ids = {1, 2, 3}
    if not {isobar_id, aligned_subsystem} <= allowed_ids:
        msg = f"Child IDs need to be one of {', '.join(map(str, allowed_ids))}"
        raise ValueError(msg)
    symbol = sp.Symbol(Rf"\hat\theta_{isobar_id}({aligned_subsystem})", real=True)
    if isobar_id == aligned_subsystem:
        return symbol, sp.S.Zero
    if (isobar_id, aligned_subsystem) in {(3, 1), (1, 2), (2, 3)}:
        remaining_id = next(iter(allowed_ids - {isobar_id, aligned_subsystem}))
        m0 = sp.Symbol("m_0", nonnegative=True)
        mi = sp.Symbol(f"m_{isobar_id}", nonnegative=True)
        mj = sp.Symbol(f"m_{aligned_subsystem}", nonnegative=True)
        si = sp.Symbol(f"m_{__get_id_complement(isobar_id)}", nonnegative=True) ** 2
        sj = (
            sp.Symbol(f"m_{__get_id_complement(aligned_subsystem)}", nonnegative=True)
            ** 2
        )
        sk = sp.Symbol(f"m_{__get_id_complement(remaining_id)}", nonnegative=True) ** 2
        theta = sp.acos(
            (
                (m0**2 + mi**2 - si) * (m0**2 + mj**2 - sj)
                - 2 * m0**2 * (sk - mi**2 - mj**2)
            )
            / (sp.sqrt(Kallen(m0**2, mj**2, sj)) * sp.sqrt(Kallen(m0**2, si, mi**2)))
        )
        return symbol, theta
    _, theta = formulate_theta_hat_angle(
        isobar_id=aligned_subsystem,
        aligned_subsystem=isobar_id,
    )
    return symbol, -theta


def formulate_zeta_angle(  # noqa: C901, PLR0911, PLR0914
    rotated_state: int,
    aligned_subsystem: int,
    reference_subsystem: int,
) -> tuple[sp.Symbol, sp.acos]:
    r"""Formulate expression for the alignment angle :math:`\zeta^i_{j(k)}`."""
    zeta_symbol = sp.Symbol(
        Rf"\zeta^{rotated_state}_{{{aligned_subsystem}({reference_subsystem})}}",
        real=True,
    )
    if rotated_state == 0:
        _, theta = formulate_theta_hat_angle(aligned_subsystem, reference_subsystem)
        return zeta_symbol, theta
    if reference_subsystem == 0:
        _, zeta_expr = formulate_zeta_angle(
            rotated_state, aligned_subsystem, rotated_state
        )
        return zeta_symbol, zeta_expr
    if aligned_subsystem == reference_subsystem:
        return zeta_symbol, sp.S.Zero
    m0, m1, m2, m3 = sp.symbols("m_(:4)", nonnegative=True)
    s1 = sp.Symbol("m_23", nonnegative=True) ** 2
    s2 = sp.Symbol("m_13", nonnegative=True) ** 2
    s3 = sp.Symbol("m_12", nonnegative=True) ** 2
    if (rotated_state, aligned_subsystem, reference_subsystem) == (1, 1, 3):
        cos_zeta_expr = (
            2 * m1**2 * (s2 - m0**2 - m2**2)
            + (m0**2 + m1**2 - s1) * (s3 - m1**2 - m2**2)
        ) / (sp.sqrt(Kallen(m0**2, m1**2, s1)) * sp.sqrt(Kallen(s3, m1**2, m2**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) == (1, 2, 1):
        cos_zeta_expr = (
            2 * m1**2 * (s3 - m0**2 - m3**2)
            + (m0**2 + m1**2 - s1) * (s2 - m1**2 - m3**2)
        ) / (sp.sqrt(Kallen(m0**2, m1**2, s1)) * sp.sqrt(Kallen(s2, m1**2, m3**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) == (2, 2, 1):
        cos_zeta_expr = (
            2 * m2**2 * (s3 - m0**2 - m3**2)
            + (m0**2 + m2**2 - s2) * (s1 - m2**2 - m3**2)
        ) / (sp.sqrt(Kallen(m0**2, m2**2, s2)) * sp.sqrt(Kallen(s1, m2**2, m3**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) == (2, 3, 2):
        cos_zeta_expr = (
            2 * m2**2 * (s1 - m0**2 - m1**2)
            + (m0**2 + m2**2 - s2) * (s3 - m2**2 - m1**2)
        ) / (sp.sqrt(Kallen(m0**2, m2**2, s2)) * sp.sqrt(Kallen(s3, m2**2, m1**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) == (3, 3, 2):
        cos_zeta_expr = (
            2 * m3**2 * (s1 - m0**2 - m1**2)
            + (m0**2 + m3**2 - s3) * (s2 - m3**2 - m1**2)
        ) / (sp.sqrt(Kallen(m0**2, m3**2, s3)) * sp.sqrt(Kallen(s2, m3**2, m1**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) == (3, 1, 3):
        cos_zeta_expr = (
            2 * m3**2 * (s2 - m0**2 - m2**2)
            + (m0**2 + m3**2 - s3) * (s1 - m3**2 - m2**2)
        ) / (sp.sqrt(Kallen(m0**2, m3**2, s3)) * sp.sqrt(Kallen(s1, m3**2, m2**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) in {  # Eq (A10)
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, 2),
    }:
        mi, si = _create_mass_mandelstam_pair(rotated_state)
        mj, sj = _create_mass_mandelstam_pair(aligned_subsystem)
        mk, sk = _create_mass_mandelstam_pair(reference_subsystem)
        cos_zeta_expr = (
            2 * mi**2 * (mj**2 + mk**2 - si)
            + (sj - mi**2 - mk**2) * (sk - mi**2 - mj**2)
        ) / (sp.sqrt(Kallen(sj, mk**2, mi**2)) * sp.sqrt(Kallen(sk, mi**2, mj**2)))
        return zeta_symbol, sp.acos(cos_zeta_expr)
    if (rotated_state, aligned_subsystem, reference_subsystem) in {
        (1, 3, 1),
        (2, 1, 2),
        (3, 2, 3),
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 1),
        (1, 3, 2),
        (2, 1, 3),
        (3, 2, 1),
    }:
        _, zeta_expr = formulate_zeta_angle(
            rotated_state=rotated_state,
            aligned_subsystem=reference_subsystem,
            reference_subsystem=aligned_subsystem,
        )
        return zeta_symbol, -zeta_expr
    msg = (
        "No expression for"
        f" ζ^{rotated_state}_{aligned_subsystem}({reference_subsystem})"
    )
    raise NotImplementedError(msg)


def _create_mass_mandelstam_pair(i: int) -> tuple[sp.Symbol, sp.Pow]:
    m_i, m_jk = sp.symbols(f"m_{i} m_{__get_id_complement(i)}", nonnegative=True)
    return m_i, m_jk**2


def __get_id_complement(state_id: int) -> str:
    complement = tuple(sorted({1, 2, 3} - {state_id}))
    return "".join(map(str, complement))
