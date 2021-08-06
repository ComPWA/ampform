r"""Experimental, symbol :math:`\boldsymbol{K}`-matrix implementations.

See :doc:`/usage/dynamics/k-matrix`.

This module is an implementation of :doc:`compwa-org:report/005`,
:doc:`compwa-org:report/009`, and :doc:`compwa-org:report/010`. It works with
classes to keep the code organized and to enable caching of the matrix
multiplications, but this might change once these dynamics are implemented into
the amplitude builder.
"""

import functools
from abc import ABC, abstractmethod
from typing import Tuple, Union

import sympy as sp

from ampform.sympy import create_symbol_matrix


class KMatrix(ABC):
    @classmethod
    @abstractmethod
    def formulate(
        cls,
        n_channels: int,
        n_resonances: Union[int, sp.Symbol],
        parametrize: bool = True,
    ) -> sp.Matrix:
        """Formulate :math:`K`-matrix with its own parametrization."""


class NonRelativisticKMatrix(KMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_kt_matrices(n_channels: int) -> Tuple[sp.Matrix, sp.Matrix]:
        k_matrix = create_symbol_matrix("K", n_channels, n_channels)
        t_matrix = k_matrix * (sp.eye(n_channels) - sp.I * k_matrix).inv()
        return k_matrix, t_matrix

    @classmethod
    def formulate(
        cls, n_channels: int, n_resonances: int, parametrize: bool = True
    ) -> sp.Matrix:
        k_matrix, t_matrix = cls._create_kt_matrices(n_channels)
        if not parametrize:
            return t_matrix
        return t_matrix.xreplace(
            {
                k_matrix[i, j]: cls.parametrization(
                    i=i,
                    j=j,
                    s=sp.Symbol("s"),
                    resonance_mass=sp.IndexedBase("m"),
                    resonance_width=sp.IndexedBase("Gamma"),
                    residue_constant=sp.IndexedBase("gamma"),
                    n_resonances=n_resonances,
                    resonance_idx=sp.Symbol("R", integer=True, positive=True),
                )
                for i in range(n_channels)
                for j in range(n_channels)
            }
        )

    @staticmethod
    def parametrization(  # pylint: disable=too-many-arguments
        i: int,
        j: int,
        s: sp.Symbol,
        resonance_mass: sp.IndexedBase,
        resonance_width: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        n_resonances: Union[int, sp.Symbol],
        resonance_idx: Union[int, sp.Symbol],
    ) -> sp.Expr:
        def residue_function(resonance_idx: int, i: int) -> sp.Expr:
            return residue_constant[resonance_idx, i] * sp.sqrt(
                resonance_mass[resonance_idx] * resonance_width[resonance_idx]
            )

        g_i = residue_function(resonance_idx, i)
        g_j = residue_function(resonance_idx, j)
        parametrization = (g_i * g_j) / (
            resonance_mass[resonance_idx] ** 2 - s
        )
        return sp.Sum(parametrization, (resonance_idx, 1, n_resonances))
