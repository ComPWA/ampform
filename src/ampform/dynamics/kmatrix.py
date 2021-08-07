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
from typing import Any, Optional, Tuple, Union

import sympy as sp

from ampform.dynamics import (
    CoupledWidth,
    PhaseSpaceFactor,
    PhaseSpaceFactorProtocol,
)
from ampform.sympy import create_symbol_matrix


class TMatrix(ABC):
    @classmethod
    @abstractmethod
    def formulate(
        cls,
        n_channels: int,
        n_resonances: Union[int, sp.Symbol],
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        """Formulate :math:`K`-matrix with its own parametrization."""


class RelativisticKMatrix(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(
        n_channels: int,
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        sqrt_rho = sp.zeros(n_channels, n_channels)
        sqrt_rho_dagger = sp.zeros(n_channels, n_channels)
        for i in range(n_channels):
            rho = sp.Symbol(f"rho{i}")
            sqrt_rho[i, i] = sp.sqrt(rho)
            sqrt_rho_dagger[i, i] = 1 / sp.conjugate(sp.sqrt(rho))
        k_matrix = create_symbol_matrix("K", n_channels, n_channels)
        t_hat = k_matrix * (sp.eye(n_channels) - sp.I * rho * k_matrix).inv()
        t_matrix = sqrt_rho_dagger * t_hat * sqrt_rho
        return t_matrix, k_matrix

    @classmethod
    def formulate(
        cls,
        n_channels: int,
        n_resonances: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        t_matrix, k_matrix = cls._create_matrices(n_channels)
        if not parametrize:
            return t_matrix
        phsp_factor: PhaseSpaceFactorProtocol = kwargs.get(
            "phsp_factor", PhaseSpaceFactor
        )
        s = sp.Symbol("s")
        m_a = sp.IndexedBase("m_a")
        m_b = sp.IndexedBase("m_b")
        return t_matrix.xreplace(
            {
                k_matrix[i, j]: cls.parametrization(
                    i=i,
                    j=j,
                    s=s,
                    resonance_mass=sp.IndexedBase("m"),
                    resonance_width=sp.IndexedBase("Gamma"),
                    m_a=m_a,
                    m_b=m_b,
                    residue_constant=sp.IndexedBase("gamma"),
                    n_resonances=n_resonances,
                    resonance_idx=sp.Symbol("R", integer=True, positive=True),
                    angular_momentum=kwargs.get("angular_momentum", 0),
                    meson_radius=kwargs.get("meson_radius", 1),
                    phsp_factor=phsp_factor,
                )
                for i in range(n_channels)
                for j in range(n_channels)
            }
        ).xreplace(
            {
                sp.Symbol(f"rho{i}"): phsp_factor(s, m_a[i], m_b[i])
                for i in range(n_channels)
            }
        )

    @staticmethod
    def parametrization(  # pylint: disable=too-many-arguments, too-many-locals
        i: int,
        j: int,
        s: sp.Symbol,
        resonance_mass: sp.IndexedBase,
        resonance_width: sp.IndexedBase,
        m_a: sp.IndexedBase,
        m_b: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        n_resonances: Union[int, sp.Symbol],
        resonance_idx: Union[int, sp.Symbol],
        angular_momentum: Union[int, sp.Symbol] = 0,
        meson_radius: Union[int, sp.Symbol] = 1,
        phsp_factor: Optional[PhaseSpaceFactorProtocol] = None,
    ) -> sp.Expr:
        def residue_function(resonance_idx: int, i: int) -> sp.Expr:
            return residue_constant[resonance_idx, i] * sp.sqrt(
                resonance_mass[resonance_idx]
                * CoupledWidth(
                    s=s,
                    mass0=resonance_mass[resonance_idx],
                    gamma0=resonance_width[resonance_idx, i],
                    m_a=m_a[i],
                    m_b=m_b[i],
                    angular_momentum=angular_momentum,
                    meson_radius=meson_radius,
                    phsp_factor=phsp_factor,
                )
            )

        g_i = residue_function(resonance_idx, i)
        g_j = residue_function(resonance_idx, j)
        parametrization = (g_i * g_j) / (
            resonance_mass[resonance_idx] ** 2 - s
        )
        return sp.Sum(parametrization, (resonance_idx, 1, n_resonances))


class NonRelativisticKMatrix(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(n_channels: int) -> Tuple[sp.Matrix, sp.Matrix]:
        k_matrix = create_symbol_matrix("K", n_channels, n_channels)
        t_matrix = k_matrix * (sp.eye(n_channels) - sp.I * k_matrix).inv()
        return t_matrix, k_matrix

    @classmethod
    def formulate(
        cls,
        n_channels: int,
        n_resonances: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        t_matrix, k_matrix = cls._create_matrices(n_channels)
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


class NonRelativisticPVector(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(
        n_channels: int,
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        k_matrix = create_symbol_matrix("K", m=n_channels, n=n_channels)
        p_vector = create_symbol_matrix("P", m=n_channels, n=1)
        t_matrix = (sp.eye(n_channels) - sp.I * k_matrix).inv() * p_vector
        return t_matrix, k_matrix, p_vector

    @classmethod
    def formulate(
        cls,
        n_channels: int,
        n_resonances: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        t_matrix, k_matrix, p_vector = cls._create_matrices(n_channels)
        if not parametrize:
            return t_matrix
        s = sp.Symbol("s")
        resonance_mass = sp.IndexedBase("m")
        resonance_width = sp.IndexedBase("Gamma")
        residue_constant = sp.IndexedBase("gamma")
        resonance_idx = sp.Symbol("R", integer=True, positive=True)
        return t_matrix.xreplace(
            {
                k_matrix[i, j]: NonRelativisticKMatrix.parametrization(
                    i=i,
                    j=j,
                    s=s,
                    resonance_mass=resonance_mass,
                    resonance_width=resonance_width,
                    residue_constant=residue_constant,
                    n_resonances=n_resonances,
                    resonance_idx=resonance_idx,
                )
                for i in range(n_channels)
                for j in range(n_channels)
            }
        ).xreplace(
            {
                p_vector[i]: cls.parametrization(
                    i=i,
                    s=sp.Symbol("s"),
                    resonance_mass=resonance_mass,
                    resonance_width=resonance_width,
                    residue_constant=residue_constant,
                    beta_constant=sp.IndexedBase("beta"),
                    n_resonances=n_resonances,
                    resonance_idx=resonance_idx,
                )
                for i in range(n_channels)
            }
        )

    @staticmethod
    def parametrization(  # pylint: disable=too-many-arguments
        i: int,
        s: sp.Symbol,
        resonance_mass: sp.IndexedBase,
        resonance_width: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        beta_constant: sp.IndexedBase,
        n_resonances: Union[int, sp.Symbol],
        resonance_idx: Union[int, sp.Symbol],
    ) -> sp.Expr:
        beta = beta_constant[resonance_idx]
        gamma = residue_constant[resonance_idx, i]
        mass = resonance_mass[resonance_idx]
        width = resonance_width[resonance_idx]
        parametrization = beta * gamma * mass * width / (mass ** 2 - s)
        return sp.Sum(parametrization, (resonance_idx, 1, n_resonances))
