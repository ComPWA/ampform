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
    BlattWeisskopfSquared,
    BreakupMomentumSquared,
    EnergyDependentWidth,
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
        n_poles: Union[int, sp.Symbol],
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        """Formulate :math:`K`-matrix with its own parametrization."""


class RelativisticKMatrix(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(
        n_channels: int, return_t_hat: bool = False
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        # pylint: disable=no-member
        rho = _create_rho_matrix(n_channels)
        sqrt_rho: sp.Matrix = sp.sqrt(rho).doit()
        sqrt_rho_conj = sp.conjugate(sqrt_rho)
        k_matrix = create_symbol_matrix("K", n_channels, n_channels)
        t_hat = k_matrix * (sp.eye(n_channels) - sp.I * rho * k_matrix).inv()
        if return_t_hat:
            return t_hat, k_matrix
        t_matrix = sqrt_rho_conj * t_hat * sqrt_rho
        return t_matrix, k_matrix

    @classmethod
    def formulate(
        cls,
        n_channels: int,
        n_poles: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        r"""Implementation of :eq:`T-hat in terms of K-hat`.

        Args:
            n_channels: Number of coupled channels.
            n_poles: Number of poles.
            parametrize: Set to `False` if don't want to parametrize and
                only get symbols for the matrix multiplication of
                :math:`\boldsymbol{K}` and :math:`\boldsymbol{\rho}`.

            return_t_hat: Set to `True` if you want to get the
                Lorentz-invariant :math:`\boldsymbol{\hat{T}}`-matrix instead
                of the :math:`\boldsymbol{T}`-matrix from
                Eq. :eq:`K-hat and T-hat`.
        """
        return_t_hat: bool = kwargs.pop("return_t_hat", False)
        t_matrix, k_matrix = cls._create_matrices(n_channels, return_t_hat)
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
                    pole_position=sp.IndexedBase("m"),
                    pole_width=sp.IndexedBase("Gamma"),
                    m_a=m_a,
                    m_b=m_b,
                    residue_constant=sp.IndexedBase("gamma"),
                    n_poles=n_poles,
                    pole_id=sp.Symbol("R", integer=True, positive=True),
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
        pole_position: sp.IndexedBase,
        pole_width: sp.IndexedBase,
        m_a: sp.IndexedBase,
        m_b: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        n_poles: Union[int, sp.Symbol],
        pole_id: Union[int, sp.Symbol],
        angular_momentum: Union[int, sp.Symbol] = 0,
        meson_radius: Union[int, sp.Symbol] = 1,
        phsp_factor: Optional[PhaseSpaceFactorProtocol] = None,
    ) -> sp.Expr:
        def residue_function(pole_id: int, i: int) -> sp.Expr:
            return residue_constant[pole_id, i] * sp.sqrt(
                pole_position[pole_id]
                * EnergyDependentWidth(
                    s=s,
                    mass0=pole_position[pole_id],
                    gamma0=pole_width[pole_id, i],
                    m_a=m_a[i],
                    m_b=m_b[i],
                    angular_momentum=angular_momentum,
                    meson_radius=meson_radius,
                    phsp_factor=phsp_factor,
                )
            )

        g_i = residue_function(pole_id, i)
        g_j = residue_function(pole_id, j)
        parametrization = (g_i * g_j) / (pole_position[pole_id] ** 2 - s)
        return sp.Sum(parametrization, (pole_id, 1, n_poles))


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
        n_poles: int,
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
                    pole_position=sp.IndexedBase("m"),
                    pole_width=sp.IndexedBase("Gamma"),
                    residue_constant=sp.IndexedBase("gamma"),
                    n_poles=n_poles,
                    pole_id=sp.Symbol("R", integer=True, positive=True),
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
        pole_position: sp.IndexedBase,
        pole_width: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        n_poles: Union[int, sp.Symbol],
        pole_id: Union[int, sp.Symbol],
    ) -> sp.Expr:
        def residue_function(pole_id: int, i: int) -> sp.Expr:
            return residue_constant[pole_id, i] * sp.sqrt(
                pole_position[pole_id] * pole_width[pole_id, i]
            )

        g_i = residue_function(pole_id, i)
        g_j = residue_function(pole_id, j)
        parametrization = (g_i * g_j) / (pole_position[pole_id] ** 2 - s)
        return sp.Sum(parametrization, (pole_id, 1, n_poles))


class NonRelativisticPVector(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(
        n_channels: int,
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        k_matrix = create_symbol_matrix("K", m=n_channels, n=n_channels)
        p_vector = create_symbol_matrix("P", m=n_channels, n=1)
        f_vector = (sp.eye(n_channels) - sp.I * k_matrix).inv() * p_vector
        return f_vector, k_matrix, p_vector

    @classmethod
    def formulate(
        cls,
        n_channels: int,
        n_poles: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        f_vector, k_matrix, p_vector = cls._create_matrices(n_channels)
        if not parametrize:
            return f_vector
        s = sp.Symbol("s")
        pole_position = sp.IndexedBase("m")
        pole_width = sp.IndexedBase("Gamma")
        residue_constant = sp.IndexedBase("gamma")
        pole_id = sp.Symbol("R", integer=True, positive=True)
        return f_vector.xreplace(
            {
                k_matrix[i, j]: NonRelativisticKMatrix.parametrization(
                    i=i,
                    j=j,
                    s=s,
                    pole_position=pole_position,
                    pole_width=pole_width,
                    residue_constant=residue_constant,
                    n_poles=n_poles,
                    pole_id=pole_id,
                )
                for i in range(n_channels)
                for j in range(n_channels)
            }
        ).xreplace(
            {
                p_vector[i]: cls.parametrization(
                    i=i,
                    s=sp.Symbol("s"),
                    pole_position=pole_position,
                    pole_width=pole_width,
                    residue_constant=residue_constant,
                    beta_constant=sp.IndexedBase("beta"),
                    n_poles=n_poles,
                    pole_id=pole_id,
                )
                for i in range(n_channels)
            }
        )

    @staticmethod
    def parametrization(  # pylint: disable=too-many-arguments
        i: int,
        s: sp.Symbol,
        pole_position: sp.IndexedBase,
        pole_width: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        beta_constant: sp.IndexedBase,
        n_poles: Union[int, sp.Symbol],
        pole_id: Union[int, sp.Symbol],
    ) -> sp.Expr:
        beta = beta_constant[pole_id]
        gamma = residue_constant[pole_id, i]
        mass = pole_position[pole_id]
        width = pole_width[pole_id, i]
        parametrization = beta * gamma * mass * width / (mass ** 2 - s)
        return sp.Sum(parametrization, (pole_id, 1, n_poles))


class RelativisticPVector(TMatrix):
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _create_matrices(
        n_channels: int, return_f_hat: bool = False
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        # pylint: disable=no-member
        k_matrix = create_symbol_matrix("K", m=n_channels, n=n_channels)
        rho = _create_rho_matrix(n_channels)
        sqrt_rho: sp.Matrix = sp.sqrt(rho).doit()
        sqrt_rho_conj: sp.Matrix = sp.conjugate(sqrt_rho)
        k_matrix = create_symbol_matrix("K", n_channels, n_channels)
        k_hat = sqrt_rho_conj.inv() * k_matrix * sqrt_rho.inv()
        p_vector = create_symbol_matrix("P", m=n_channels, n=1)
        f_hat = (sp.eye(n_channels) - sp.I * k_hat * rho).inv() * p_vector
        if return_f_hat:
            return f_hat, k_matrix, p_vector
        f_vector = sqrt_rho * f_hat
        return f_vector, k_matrix, p_vector

    @classmethod
    def formulate(  # pylint: disable=too-many-locals
        cls,
        n_channels: int,
        n_poles: int,
        parametrize: bool = True,
        **kwargs: Any,
    ) -> sp.Matrix:
        r"""Implementation of :eq:`F in terms of P`.

        Args:
            n_channels: Number of coupled channels.
            n_poles: Number of poles.
            parametrize: Set to `False` if don't want to parametrize and
                only get symbols for the matrix multiplication of
                :math:`\boldsymbol{K}` and :math:`\boldsymbol{\rho}`.

            return_f_hat: Set to `True` if you want to get the
                Lorentz-invariant :math:`\hat{F}`-vector instead of the
                :math:`T`-vector from Eq. :eq:`invariant vectors`.
        """
        return_f_hat: bool = kwargs.pop("return_f_hat", False)
        f_vector, k_matrix, p_vector = cls._create_matrices(
            n_channels, return_f_hat
        )
        if not parametrize:
            return f_vector
        phsp_factor: PhaseSpaceFactorProtocol = kwargs.get(
            "phsp_factor", PhaseSpaceFactor
        )
        s = sp.Symbol("s")
        pole_position = sp.IndexedBase("m")
        pole_width = sp.IndexedBase("Gamma")
        residue_constant = sp.IndexedBase("gamma")
        m_a = sp.IndexedBase("m_a")
        m_b = sp.IndexedBase("m_b")
        pole_id = sp.Symbol("R", integer=True, positive=True)
        angular_momentum = kwargs.get("angular_momentum", 0)
        meson_radius = kwargs.get("meson_radius", 1)
        return (
            f_vector.xreplace(
                {
                    k_matrix[i, j]: RelativisticKMatrix.parametrization(
                        i=i,
                        j=j,
                        s=s,
                        pole_position=pole_position,
                        pole_width=pole_width,
                        m_a=m_a,
                        m_b=m_b,
                        residue_constant=residue_constant,
                        n_poles=n_poles,
                        pole_id=pole_id,
                        angular_momentum=angular_momentum,
                        meson_radius=meson_radius,
                    )
                    for i in range(n_channels)
                    for j in range(n_channels)
                }
            )
            .xreplace(
                {
                    p_vector[i]: cls.parametrization(
                        i=i,
                        s=s,
                        pole_position=pole_position,
                        pole_width=pole_width,
                        m_a=m_a,
                        m_b=m_b,
                        beta_constant=sp.IndexedBase("beta"),
                        residue_constant=residue_constant,
                        n_poles=n_poles,
                        pole_id=pole_id,
                        angular_momentum=angular_momentum,
                        meson_radius=meson_radius,
                    )
                    for i in range(n_channels)
                }
            )
            .xreplace(
                {
                    sp.Symbol(f"rho{i}"): phsp_factor(s, m_a[i], m_b[i])
                    for i in range(n_channels)
                }
            )
        )

    @staticmethod
    def parametrization(  # pylint: disable=too-many-arguments, too-many-locals
        i: int,
        s: sp.Symbol,
        pole_position: sp.IndexedBase,
        pole_width: sp.IndexedBase,
        m_a: sp.IndexedBase,
        m_b: sp.IndexedBase,
        beta_constant: sp.IndexedBase,
        residue_constant: sp.IndexedBase,
        n_poles: Union[int, sp.Symbol],
        pole_id: Union[int, sp.Symbol],
        angular_momentum: Union[int, sp.Symbol] = 0,
        meson_radius: Union[int, sp.Symbol] = 1,
    ) -> sp.Expr:
        beta = beta_constant[pole_id]
        gamma = residue_constant[pole_id, i]
        mass0 = pole_position[pole_id]
        width = pole_width[pole_id, i]
        q_squared = BreakupMomentumSquared(s, m_a[i], m_b[i])
        form_factor_squared = BlattWeisskopfSquared(
            angular_momentum, z=q_squared * meson_radius ** 2
        )
        form_factor = sp.sqrt(form_factor_squared)
        return sp.Sum(
            beta * gamma * mass0 * width * form_factor / (mass0 ** 2 - s),
            (pole_id, 1, n_poles),
        )


def _create_rho_matrix(n_channels: int) -> sp.Matrix:
    """Create a phase space matrix with :code:`n_channels`.

    >>> _create_rho_matrix(n_channels=2)
    Matrix([
    [rho0,    0],
    [   0, rho1]])
    """
    rho_matrix: sp.Matrix = sp.zeros(n_channels, n_channels)
    for i in range(n_channels):
        rho_matrix[i, i] = sp.Symbol(f"rho{i}")
    return rho_matrix
