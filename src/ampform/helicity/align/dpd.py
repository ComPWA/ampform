"""Spin alignment with Dalitz-Plot Decomposition.

See :cite:`mikhasenkoDalitzplotDecompositionThreebody2020`.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import sympy as sp
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform.kinematics.angles import formulate_zeta_angle

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

if TYPE_CHECKING:
    from sympy.physics.quantum.spin import WignerD


class DPDAlignmentWignerGenerator:
    def __init__(self, reference_subsystem: Literal[1, 2, 3] = 1) -> None:
        self.angle_definitions: dict[sp.Symbol, sp.acos] = {}
        self.reference_subsystem = reference_subsystem

    def __call__(
        self,
        j: sp.Rational,
        m: sp.Rational,
        m_prime: sp.Rational,
        rotated_state: int,
        aligned_subsystem: int,
    ) -> sp.Rational | WignerD:
        # pylint: disable=too-many-arguments
        if j == 0:
            return sp.Rational(1)
        zeta, zeta_expr = formulate_zeta_angle(
            rotated_state, aligned_subsystem, self.reference_subsystem
        )
        self.angle_definitions[zeta] = zeta_expr
        return Wigner.d(j, m, m_prime, zeta)
