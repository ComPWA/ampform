# pylint: disable=redefined-outer-name
from __future__ import annotations

from typing import Iterable, Sequence

import pytest
import qrules
import sympy as sp
from qrules import ParticleCollection
from qrules.particle import Particle

from ampform import get_builder


def calculate_sympy_integral(
    intensity,
    integration_variables: Iterable[sp.Symbol],
    jacobi_determinant=None,
) -> sp.Expr:
    if jacobi_determinant is None:
        for int_var in integration_variables:
            if "theta" in int_var.name:
                intensity *= sp.sin(int_var)
    else:
        intensity *= jacobi_determinant
    integral = sp.integrate(
        intensity,
        *(
            (x, -sp.pi, sp.pi) if "phi" in x.name else (x, 0, sp.pi)
            for x in integration_variables
        ),
    )
    return sp.trigsimp(
        sp.nsimplify(
            sp.re(integral).doit(),
            rational=True,
        )
    )


def normalize(sympy_expression: sp.Expr, variable_names: Sequence[str]) -> sp.Expr:
    variables = [sp.Symbol(x, real=True) for x in variable_names]
    normalization = sp.integrate(
        sympy_expression,
        *((x, -sp.pi, sp.pi) if "phi" in x.name else (x, 0, sp.pi) for x in variables),
    )
    return sp.trigsimp((sympy_expression / normalization).expand(trig=True))


class TestEpemToDmD0Pip:
    @pytest.fixture(scope="class")
    def sympy_model(self, particle_database: ParticleCollection) -> sp.Expr:
        epem = Particle(
            name="EpEm",
            pid=12345678,
            mass=4.36,
            spin=1.0,
            parity=-1,
            c_parity=-1,
        )
        particles = ParticleCollection(particle_database)
        particles.add(epem)

        reaction = qrules.generate_transitions(
            initial_state=[("EpEm", [-1])],
            final_state=[("D0", [0]), ("D-", [0]), ("pi+", [0])],
            allowed_intermediate_particles=["D(2)*(2460)+"],
            allowed_interaction_types="em",
            particle_db=particles,
        )

        amplitude_model = get_builder(reaction).formulate()
        full_model = sp.simplify(
            amplitude_model.expression.subs(amplitude_model.parameter_defaults)
            .doit()
            .expand(complex=True)
        )
        assert sp.im(full_model) == 0
        return sp.re(full_model)

    @pytest.mark.parametrize(
        ("angular_variables", "expected_distribution_function"),
        [
            (  # cos(theta) distribution from epem decay
                "theta_02",
                1 + sp.cos(sp.Symbol("theta_02", real=True)) ** 2,
            ),
            (  # phi distribution of the epem decay
                "phi_0",
                1,
            ),
            (  # cos(theta') distribution from D2*
                "theta_0^02",
                1 - (2 * sp.cos(sp.Symbol("theta_0^02", real=True)) ** 2 - 1) ** 2,
            ),
            (  # phi' distribution of the D2* decay
                "phi_0^02",
                3 - 2 * sp.sin(sp.Symbol("phi_0^02", real=True)) ** 2,
            ),
            (  # 2d distribution of the D2* decay
                ["theta_0^02", "phi_0^02"],
                (1 - sp.cos(sp.Symbol("theta_0^02", real=True)) ** 2)
                * (sp.cos(sp.Symbol("theta_0^02", real=True)) ** 2)
                * (2 + sp.cos(2 * sp.Symbol("phi_0^02", real=True))),
            ),
        ],
    )
    def test_angular_distributions(
        self,
        angular_variables: str | Sequence[str],
        expected_distribution_function: sp.Expr,
        sympy_model: sp.Expr,
    ) -> None:
        free_symbols: set[sp.Symbol] = sympy_model.free_symbols  # type: ignore[assignment]
        assert {s.name for s in free_symbols} == {
            "phi_0^02",
            "theta_02",
            "theta_0^02",
        }

        if isinstance(angular_variables, str):
            angular_variables = (angular_variables,)

        # remove angular variable
        integration_variable_set = set(angular_variables)
        integration_variables = [
            x for x in free_symbols if x.name not in integration_variable_set
        ]

        # Note: using nsimplify with rational=True solves assertion failure due
        # to float point imprecision
        assert normalize(
            expected_distribution_function, angular_variables
        ) == normalize(
            calculate_sympy_integral(
                sympy_model,
                integration_variables,
            ),
            angular_variables,
        )


class TestD1ToD0PiPi:
    @pytest.fixture(scope="class")
    def sympy_model(self) -> sp.Expr:
        reaction = qrules.generate_transitions(
            initial_state=[("D(1)(2420)0", [-1])],
            final_state=[("D0", [0]), ("pi-", [0]), ("pi+", [0])],
            allowed_intermediate_particles=["D*"],
            allowed_interaction_types="strong",
            formalism="helicity",
        )
        amplitude_model = get_builder(reaction).formulate()

        coefficient = sp.Symbol(
            R"C_{D_{1}(2420)^{0} \to D^{*}(2010)^{+}_{0} \pi^{-}_{0}; "
            R"D^{*}(2010)^{+} \to D^{0}_{0} \pi^{+}_{0}}"
        )
        if coefficient not in amplitude_model.parameter_defaults:
            raise KeyError(
                f"Coefficient {coefficient} does not exist in parameter"
                " defaults. Choose any of"
                f" {list(amplitude_model.parameter_defaults)}"
            )
        amplitude_model.parameter_defaults[coefficient] = 0.5

        full_model = sp.simplify(
            amplitude_model.expression.subs(amplitude_model.parameter_defaults)
            .doit()
            .expand(complex=True)
        )
        assert sp.im(full_model) == 0
        return sp.re(full_model)

    @pytest.mark.parametrize(
        ("angular_variables", "expected_distribution_function"),
        [
            (  # theta distribution from D1 decay
                "theta_02",
                sp.Rational(5, 4)
                + sp.Rational(3, 4) * sp.cos(sp.Symbol("theta_02", real=True)) ** 2,
            ),
            (  # theta distribution from D*
                "theta_0^02",
                1 - sp.Rational(3, 4) * sp.cos(sp.Symbol("theta_0^02", real=True)) ** 2,
            ),
            (  # phi distribution of the D* decay
                "phi_0^02",
                1 - sp.Rational(4, 9) * sp.cos(2 * sp.Symbol("phi_0^02", real=True)),
            ),
        ],
    )
    def test_angular_distributions(
        self,
        angular_variables: str | Sequence[str],
        expected_distribution_function: sp.Expr,
        sympy_model: sp.Expr,
    ) -> None:
        free_symbols: set[sp.Symbol] = sympy_model.free_symbols  # type: ignore[assignment]
        assert {s.name for s in free_symbols} == {
            "phi_0^02",
            "theta_02",
            "theta_0^02",
        }

        if isinstance(angular_variables, str):
            angular_variables = (angular_variables,)

        # remove angular variable
        integration_variable_set = set(angular_variables)
        integration_variables = [
            x for x in free_symbols if x.name not in integration_variable_set
        ]

        # Note: using nsimplify with rational=True solves assertion failure due
        # to float point imprecision
        assert normalize(
            expected_distribution_function, angular_variables
        ) == normalize(
            calculate_sympy_integral(
                sympy_model,
                integration_variables,
            ),
            angular_variables,
        )
