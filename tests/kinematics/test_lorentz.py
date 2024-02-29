from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp
from numpy.lib.scimath import sqrt as complex_sqrt
from sympy.printing.numpy import NumPyPrinter

from ampform.kinematics.lorentz import (
    ArraySize,
    BoostMatrix,
    BoostZMatrix,
    Energy,
    FourMomenta,
    FourMomentumSymbol,
    FourMomentumX,
    FourMomentumY,
    FourMomentumZ,
    InvariantMass,
    NegativeMomentum,
    RotationYMatrix,
    RotationZMatrix,
    ThreeMomentum,
    _OnesArray,
    _ZerosArray,
    compute_boost_chain,
    compute_invariant_masses,
    three_momentum_norm,
)
from ampform.sympy._array_expressions import (
    ArrayMultiplication,
    ArraySlice,
    ArraySymbol,
)

if TYPE_CHECKING:
    from qrules.topology import Topology

    from ampform.sympy import NumPyPrintable


class TestBoostMatrix:
    def test_boost_in_z_direction_reduces_to_z_boost(self):
        p = FourMomentumSymbol("p", shape=[])
        expr = BoostMatrix(p)
        func = sp.lambdify(p, expr.doit(), cse=True)
        p_array = np.array([[5, 0, 0, 1]])
        matrix = func(p_array)[0]
        assert pytest.approx(matrix) == np.array([
            [1.02062073, 0, 0, -0.20412415],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-0.20412415, 0, 0, 1.02062073],
        ])

        beta = three_momentum_norm(p) / Energy(p)
        z_expr = BoostZMatrix(beta, n_events=ArraySize(p))
        z_func = sp.lambdify(p, z_expr.doit(), cse=True)
        z_matrix = z_func(p_array)[0]
        assert pytest.approx(matrix) == z_matrix

    @pytest.mark.parametrize("state_id", [0, 1, 2, 3])
    def test_boost_into_rest_frame_gives_mass(
        self,
        state_id: int,
        data_sample: dict[int, np.ndarray],
        topology_and_momentum_symbols: tuple[Topology, FourMomenta],
    ):
        pi0_mass = 0.135
        masses = {0: pi0_mass, 1: 0, 2: pi0_mass, 3: pi0_mass}
        _, momenta = topology_and_momentum_symbols
        momentum = momenta[state_id]
        momentum_array = data_sample[state_id]
        boost = BoostMatrix(momentum)
        expr = ArrayMultiplication(boost, momentum)
        func = sp.lambdify(momentum, expr.doit(), cse=True)
        boosted_array: np.ndarray = func(momentum_array)
        assert not np.all(np.isnan(boosted_array))
        boosted_array = np.nan_to_num(boosted_array, nan=masses[state_id])
        mass_array = boosted_array[:, 0]
        assert pytest.approx(mass_array, abs=1e-2) == masses[state_id]
        p_xyz = boosted_array[:, 1:]
        assert pytest.approx(p_xyz) == 0

    @pytest.mark.parametrize("state_id", [0, 2, 3])
    def test_boosting_back_gives_original_momentum(
        self, state_id: int, data_sample: dict[int, np.ndarray]
    ):
        p = FourMomentumSymbol("p", shape=[])
        boost = BoostMatrix(p)
        inverse_boost = BoostMatrix(NegativeMomentum(p))
        expr = ArrayMultiplication(inverse_boost, boost, p)
        func = sp.lambdify(p, expr.doit(), cse=True)
        momentum_array = data_sample[state_id]
        computed_momentum: np.ndarray = func(momentum_array)
        assert not np.any(np.isnan(computed_momentum))
        assert pytest.approx(computed_momentum, abs=1e-2) == momentum_array


class TestBoostZMatrix:
    def test_boost_into_own_rest_frame_gives_mass(self):
        p = FourMomentumSymbol("p", shape=[])
        n_events = ArraySize(p)
        beta = three_momentum_norm(p) / Energy(p)
        expr = BoostZMatrix(beta, n_events)
        func = sp.lambdify(p, expr.doit(), cse=True)
        p_array = np.array([[5, 0, 0, 1]])
        boost_z = func(p_array)[0]
        boosted_array = np.einsum("...ij,...j->...i", boost_z, p_array)
        mass = 4.89897949
        assert pytest.approx(boosted_array[0]) == [mass, 0, 0, 0]

        mass_expr = InvariantMass(p)
        func = sp.lambdify(p, mass_expr.doit(), cse=True)
        mass_array = func(p_array)
        assert pytest.approx(mass_array[0]) == mass

    def test_numpycode_cse_in_expression_tree(self):
        p, beta, phi, theta = sp.symbols("p beta phi theta")
        expr = ArrayMultiplication(
            BoostZMatrix(beta, n_events=ArraySize(p)),
            RotationYMatrix(theta, n_events=ArraySize(p)),
            RotationZMatrix(phi, n_events=ArraySize(p)),
            p,
        )
        func = sp.lambdify([], expr.doit(), cse=True)
        src = inspect.getsource(func)
        expected_src = """
        def _lambdifygenerated():
            x0 = 1/sqrt(1 - beta**2)
            x1 = len(p)
            x2 = ones(x1)
            x3 = zeros(x1)
            return (einsum("...ij,...jk,...kl,...l->...i", array(
                    [
                        [x0, x3, x3, -beta*x0],
                        [x3, x2, x3, x3],
                        [x3, x3, x2, x3],
                        [-beta*x0, x3, x3, x0],
                    ]
                ).transpose((2, 0, 1)), array(
                    [
                        [x2, x3, x3, x3],
                        [x3, cos(theta), x3, sin(theta)],
                        [x3, x3, x2, x3],
                        [x3, -sin(theta), x3, cos(theta)],
                    ]
                ).transpose((2, 0, 1)), array(
                    [
                        [x2, x3, x3, x3],
                        [x3, cos(phi), -sin(phi), x3],
                        [x3, sin(phi), cos(phi), x3],
                        [x3, x3, x3, x2],
                    ]
                ).transpose((2, 0, 1)), p))
        """
        expected_src = textwrap.dedent(expected_src)
        assert src.strip() == expected_src.strip()


class TestFourMomentumXYZ:
    def symbols(
        self,
    ) -> tuple[FourMomentumSymbol, Energy, FourMomentumX, FourMomentumY, FourMomentumZ]:
        p = FourMomentumSymbol("p", shape=[])
        e = Energy(p)
        p_x = FourMomentumX(p)
        p_y = FourMomentumY(p)
        p_z = FourMomentumZ(p)
        return p, e, p_x, p_y, p_z

    def test_elements(self):
        p, e, p_x, p_y, p_z = self.symbols()
        assert e.evaluate() == ArraySlice(p, indices=(slice(None), 0))
        assert p_x.evaluate() == ArraySlice(p, indices=(slice(None), 1))
        assert p_y.evaluate() == ArraySlice(p, indices=(slice(None), 2))
        assert p_z.evaluate() == ArraySlice(p, indices=(slice(None), 3))

    def test_latex(self):
        _, e, p_x, p_y, p_z = self.symbols()
        assert sp.latex(e) == R"E\left(p\right)"
        assert sp.latex(p_x) == "{p}_x"
        assert sp.latex(p_y) == "{p}_y"
        assert sp.latex(p_z) == "{p}_z"
        a = ArraySymbol("A", shape=[])
        b = ArraySymbol("B", shape=[])
        expr = FourMomentumX(a + b)
        assert sp.latex(expr) == R"\left(A + B\right)_x"


class TestInvariantMass:
    def test_latex(self):
        p = FourMomentumSymbol("p1", shape=[])
        mass = InvariantMass(p)
        latex = sp.latex(mass)
        assert latex == "m_{p_{1}}"

    @pytest.mark.parametrize(
        ("state_id", "expected_mass"),
        [
            (0, 0.13498),
            (1, 0.00048 + 0.00032j),
            (2, 0.13498),
            (3, 0.13498),
        ],
    )
    def test_numpy(
        self,
        data_sample: dict[int, np.ndarray],
        state_id: int,
        expected_mass: float,
    ):
        p = FourMomentumSymbol(f"p{state_id}", shape=[])
        mass = InvariantMass(p)
        np_mass = sp.lambdify(p, mass.doit(), cse=True)
        four_momenta = data_sample[state_id]
        computed_values = np_mass(four_momenta)
        average_mass = np.average(computed_values)
        assert pytest.approx(average_mass, abs=1e-5) == expected_mass


class TestThreeMomentum:
    @property
    def p_norm(self) -> ThreeMomentum:
        p = FourMomentumSymbol("p", shape=[])
        return ThreeMomentum(p)

    def test_latex(self):
        latex = sp.latex(self.p_norm)
        assert latex == R"\vec{p}"

    def test_numpy(self):
        numpy_code = _generate_numpy_code(self.p_norm)
        assert numpy_code == "p[:, 1:]"


class TestNegativeMomentum:
    def test_same_as_inverse(self, data_sample: dict[int, np.ndarray]):
        p = FourMomentumSymbol("p", shape=[])
        expr = NegativeMomentum(p)
        func = sp.lambdify(p, expr.doit(), cse=True)
        for p_array in data_sample.values():
            negative_array = func(p_array)
            assert pytest.approx(negative_array[:, 0]) == p_array[:, 0]
            assert pytest.approx(negative_array[:, 1:]) == -p_array[:, 1:]


class TestRotationYMatrix:
    @pytest.fixture(scope="session")
    def rotation_expr(self):
        angle = sp.Symbol("a")
        return RotationYMatrix(angle, n_events=ArraySize(angle))

    @pytest.fixture(scope="session")
    def rotation_func(self, rotation_expr: RotationYMatrix):
        angle = sp.Symbol("a")
        return sp.lambdify(angle, rotation_expr.doit(), cse=True)

    def test_numpycode_cse(self, rotation_expr: RotationYMatrix):
        func = sp.lambdify([], rotation_expr.doit(), cse=True)
        src = inspect.getsource(func)
        expected_src = """
        def _lambdifygenerated():
            x0 = len(a)
            return (array(
                    [
                        [ones(x0), zeros(x0), zeros(x0), zeros(x0)],
                        [zeros(x0), cos(a), zeros(x0), sin(a)],
                        [zeros(x0), zeros(x0), ones(x0), zeros(x0)],
                        [zeros(x0), -sin(a), zeros(x0), cos(a)],
                    ]
                ).transpose((2, 0, 1)))
        """
        expected_src = textwrap.dedent(expected_src)
        assert src.strip() == expected_src.strip()

    def test_rotation_over_pi_flips_xz(self, rotation_func):
        vectors = np.array([[1, 1, 1, 1]])
        angle_array = np.array([np.pi])
        rotated_vectors = np.einsum(
            "...ij,...j->...j", rotation_func(angle_array), vectors
        )
        assert pytest.approx(rotated_vectors) == np.array([[1, -1, 1, -1]])


class TestRotationZMatrix:
    @pytest.fixture(scope="session")
    def rotation_expr(self):
        angle = sp.Symbol("a")
        return RotationZMatrix(angle, n_events=ArraySize(angle))

    @pytest.fixture(scope="session")
    def rotation_func(self, rotation_expr: RotationZMatrix):
        angle = sp.Symbol("a")
        return sp.lambdify(angle, rotation_expr.doit(), cse=True)

    def test_numpycode_cse(self, rotation_expr: RotationZMatrix):
        func = sp.lambdify([], rotation_expr.doit(), cse=True)
        src = inspect.getsource(func)
        expected_src = """
        def _lambdifygenerated():
            x0 = len(a)
            return (array(
                    [
                        [ones(x0), zeros(x0), zeros(x0), zeros(x0)],
                        [zeros(x0), cos(a), -sin(a), zeros(x0)],
                        [zeros(x0), sin(a), cos(a), zeros(x0)],
                        [zeros(x0), zeros(x0), zeros(x0), ones(x0)],
                    ]
                ).transpose((2, 0, 1)))
        """
        expected_src = textwrap.dedent(expected_src)
        assert src.strip() == expected_src.strip()

    def test_rotation_over_pi_flips_xy(self, rotation_func):
        vectors = np.array([[1, 1, 1, 1]])
        angle_array = np.array([np.pi])
        rotated_vectors = np.einsum(
            "...ij,...j->...j", rotation_func(angle_array), vectors
        )
        assert pytest.approx(rotated_vectors) == np.array([[1, -1, -1, 1]])


@pytest.mark.parametrize("rotation", [RotationYMatrix, RotationZMatrix])
def test_rotation_latex_repr_is_identical_with_doit(rotation):
    angle, n_events = sp.symbols("a n")
    expr = rotation(angle, n_events)
    assert sp.latex(expr) == sp.latex(expr.doit())


@pytest.mark.parametrize("rotation", [RotationYMatrix, RotationZMatrix])
def test_rotation_over_multiple_two_pi_is_identity(rotation):
    angle = sp.Symbol("a")
    expr = rotation(angle, n_events=ArraySize(angle))
    func = sp.lambdify(angle, expr.doit(), cse=True)
    angle_array = np.arange(-2, 4, 1) * 2 * np.pi
    rotation_matrices = func(angle_array)
    identity = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    identity = np.tile(identity, reps=(len(angle_array), 1, 1))
    assert pytest.approx(rotation_matrices) == identity


class TestOnesZerosArray:
    @pytest.mark.parametrize("array_type", ["ones", "zeros"])
    @pytest.mark.parametrize("shape", [10, (4, 2), [3, 5, 7]])
    def test_numpycode(self, array_type, shape):
        if array_type == "ones":
            expr_class: type[NumPyPrintable] = _OnesArray
            array_func = np.ones
        elif array_type == "zeros":
            expr_class = _ZerosArray
            array_func = np.zeros
        else:
            raise NotImplementedError
        array_expr = expr_class(shape)
        create_array = sp.lambdify([], array_expr)
        array = create_array()
        np.testing.assert_array_equal(array, array_func(shape))


def test_compute_invariant_masses_names(
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)
    mass_names = set(map(str, invariant_masses))
    assert set(mass_names) == {
        "m_0",
        "m_1",
        "m_2",
        "m_3",
        "m_23",
        "m_123",
        "m_0123",
    }


def test_compute_invariant_masses_single_mass(
    data_sample: dict[int, np.ndarray],
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    momentum_values = data_sample.values()
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)
    for i in topology.outgoing_edge_ids:
        symbol = sp.Symbol(f"m_{i}", nonnegative=True)
        expr = invariant_masses[symbol]
        np_expr = sp.lambdify(momentum_symbols.values(), expr.doit(), cse=True)
        expected = __compute_mass(data_sample[i])
        computed = np_expr(*momentum_values)
        # cspell:ignore atol
        np.testing.assert_allclose(computed, expected, atol=1e-5)


@pytest.mark.parametrize("mass_name", ["m_23", "m_123", "m_0123"])
def test_compute_invariant_masses(
    mass_name: str,
    data_sample: dict[int, np.ndarray],
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    momentum_values = data_sample.values()
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)

    mass_symbol = sp.Symbol(mass_name, nonnegative=True)
    expr = invariant_masses[mass_symbol]
    np_expr = sp.lambdify(momentum_symbols.values(), expr.doit(), cse=True)
    computed = np.average(np_expr(*momentum_values))
    indices = map(int, mass_name[2:])
    masses = __compute_mass(sum(data_sample[i] for i in indices))  # type: ignore[arg-type]
    expected = np.average(masses)
    assert pytest.approx(computed, abs=1e-8) == expected


def __compute_mass(array: np.ndarray) -> np.ndarray:
    energy = array[:, 0]
    three_momentum = array[:, 1:]
    mass_squared = energy**2 - np.sum(three_momentum**2, axis=1)
    return complex_sqrt(mass_squared)


@pytest.mark.parametrize(
    ("state_id", "expected"),
    [
        (
            0,
            ["B(p0)"],
        ),
        (
            1,
            [
                "B(p1+p2+p3)",
                "B(mul(B(p1+p2+p3), p1))",
            ],
        ),
        (
            2,
            [
                "B(p1+p2+p3)",
                "B(mul(B(p1+p2+p3), p2+p3))",
                "B(mul(B(mul(B(p1+p2+p3), p2+p3)), mul(B(p1+p2+p3), p2)))",
            ],
        ),
        (
            3,
            [
                "B(p1+p2+p3)",
                "B(mul(B(p1+p2+p3), p2+p3))",
                "B(mul(B(mul(B(p1+p2+p3), p2+p3)), mul(B(p1+p2+p3), p3)))",
            ],
        ),
    ],
)
def test_compute_boost_chain(
    state_id: int,
    expected: list[str],
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    boost_chain = compute_boost_chain(topology, momentum_symbols, state_id)
    boost_chain_str = [
        str(expr)
        .replace("BoostMatrix", "B")
        .replace("ArrayMultiplication", "mul")
        .replace(" + ", "+")
        for expr in boost_chain
    ]
    assert boost_chain_str == expected


def _generate_numpy_code(expr: sp.Expr) -> str:
    # cspell:ignore doprint
    printer = NumPyPrinter()
    return printer.doprint(expr)
