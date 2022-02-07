# pylint: disable=no-member, no-self-use, redefined-outer-name
# cspell:ignore atol doprint
import inspect
import textwrap
from typing import Dict, Tuple

import numpy as np
import pytest
import sympy as sp
from numpy.lib.scimath import sqrt as complex_sqrt
from qrules.topology import Topology, create_isobar_topologies
from sympy.printing.numpy import NumPyPrinter

from ampform.kinematics import (
    BoostZMatrix,
    Energy,
    FourMomenta,
    FourMomentumSymbol,
    FourMomentumX,
    FourMomentumY,
    FourMomentumZ,
    InvariantMass,
    Phi,
    RotationYMatrix,
    RotationZMatrix,
    Theta,
    ThreeMomentumNorm,
    _ArraySize,
    _OnesArray,
    _ZerosArray,
    compute_helicity_angles,
    compute_invariant_masses,
    create_four_momentum_symbols,
    determine_attached_final_state,
)
from ampform.sympy._array_expressions import (
    ArrayMultiplication,
    ArraySlice,
    ArraySymbol,
)


@pytest.fixture(scope="session")
def topology_and_momentum_symbols(
    data_sample: Dict[int, np.ndarray]
) -> Tuple[Topology, FourMomenta]:
    n = len(data_sample)
    assert n == 4
    topologies = create_isobar_topologies(n)
    topology = topologies[1]
    momentum_symbols = create_four_momentum_symbols(topology)
    return topology, momentum_symbols


@pytest.fixture(scope="session")
def helicity_angles(
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta]
) -> Dict[str, sp.Expr]:
    topology, momentum_symbols = topology_and_momentum_symbols
    return compute_helicity_angles(momentum_symbols, topology)


class TestBoostZMatrix:
    def test_boost_into_own_rest_frame_gives_mass(self):
        p = FourMomentumSymbol("p")
        n_events = _ArraySize(p)
        beta = ThreeMomentumNorm(p) / Energy(p)
        expr = BoostZMatrix(beta, n_events)
        func = sp.lambdify(p, expr.doit())
        p_array = np.array([[5, 0, 0, 1]])
        boost_z = func(p_array)[0]
        boosted_array = np.einsum("...ij,...j->...i", boost_z, p_array)
        mass = 4.89897949
        assert pytest.approx(boosted_array[0]) == [mass, 0, 0, 0]

        expr = InvariantMass(p)
        func = sp.lambdify(p, expr.doit())
        mass_array = func(p_array)
        assert pytest.approx(mass_array[0]) == mass

    def test_numpycode_cse_in_expression_tree(self):
        p, beta, phi, theta = sp.symbols("p beta phi theta")
        expr = ArrayMultiplication(
            BoostZMatrix(beta, n_events=_ArraySize(p)),
            RotationYMatrix(theta, n_events=_ArraySize(p)),
            RotationZMatrix(phi, n_events=_ArraySize(p)),
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
    ) -> Tuple[
        FourMomentumSymbol, Energy, FourMomentumX, FourMomentumY, FourMomentumZ
    ]:
        p = FourMomentumSymbol("p")
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
        a, b = sp.symbols("A B", cls=ArraySymbol)
        expr = FourMomentumX(a + b)
        assert sp.latex(expr) == R"\left(A + B\right)_x"


class TestInvariantMass:
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
        data_sample: Dict[int, np.ndarray],
        state_id: int,
        expected_mass: float,
    ):
        p = FourMomentumSymbol(f"p{state_id}")
        mass = InvariantMass(p)
        np_mass = sp.lambdify(p, mass.doit(), "numpy")
        four_momenta = data_sample[state_id]
        computed_values = np_mass(four_momenta)
        average_mass = np.average(computed_values)
        assert pytest.approx(average_mass, abs=1e-5) == expected_mass


class TestThreeMomentumNorm:
    @property
    def p_norm(self) -> ThreeMomentumNorm:
        p = FourMomentumSymbol("p")
        return ThreeMomentumNorm(p)

    def test_latex(self):
        latex = sp.latex(self.p_norm)
        assert latex == R"\left|\vec{p}\right|"

    def test_numpy(self):
        numpy_code = _generate_numpy_code(self.p_norm)
        assert numpy_code == "numpy.sqrt(sum(p[:, 1:]**2, axis=1))"


class TestPhi:
    @property
    def phi(self) -> Theta:
        p = FourMomentumSymbol("p")
        return Phi(p)

    def test_latex(self):
        latex = sp.latex(self.phi)
        assert latex == R"\phi\left(p\right)"

    def test_numpy(self):
        phi = self.phi.doit()
        numpy_code = _generate_numpy_code(phi)
        assert numpy_code == "numpy.arctan2(p[:, 2], p[:, 1])"


class TestTheta:
    @property
    def theta(self) -> Theta:
        p = FourMomentumSymbol("p")
        return Theta(p)

    def test_latex(self):
        latex = sp.latex(self.theta)
        assert latex == R"\theta\left(p\right)"

    def test_numpy(self):
        theta = self.theta.doit()
        numpy_code = _generate_numpy_code(theta)
        assert (
            numpy_code
            == "numpy.arccos(p[:, 3]/numpy.sqrt(sum(p[:, 1:]**2, axis=1)))"
        )


class TestRotationYMatrix:
    @pytest.fixture(scope="session")
    def rotation_expr(self):
        angle, n_events = sp.symbols("a n")
        return RotationYMatrix(angle, n_events)

    @pytest.fixture(scope="session")
    def rotation_func(self, rotation_expr):
        angle = sp.Symbol("a")
        rotation_expr = rotation_expr.doit()
        rotation_expr = rotation_expr.subs(sp.Symbol("n"), _ArraySize(angle))
        return sp.lambdify(angle, rotation_expr, cse=True)

    def test_numpycode_cse(self, rotation_expr: RotationYMatrix):
        func = sp.lambdify([], rotation_expr.doit(), cse=True)
        src = inspect.getsource(func)
        expected_src = """
        def _lambdifygenerated():
            return (array(
                    [
                        [ones(n), zeros(n), zeros(n), zeros(n)],
                        [zeros(n), cos(a), zeros(n), sin(a)],
                        [zeros(n), zeros(n), ones(n), zeros(n)],
                        [zeros(n), -sin(a), zeros(n), cos(a)],
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
        angle, n_events = sp.symbols("a n")
        return RotationZMatrix(angle, n_events)

    @pytest.fixture(scope="session")
    def rotation_func(self, rotation_expr):
        angle = sp.Symbol("a")
        rotation_expr = rotation_expr.doit()
        rotation_expr = rotation_expr.subs(sp.Symbol("n"), _ArraySize(angle))
        return sp.lambdify(angle, rotation_expr, cse=True)

    def test_numpycode_cse(self, rotation_expr: RotationZMatrix):
        func = sp.lambdify([], rotation_expr.doit(), cse=True)
        src = inspect.getsource(func)
        expected_src = """
        def _lambdifygenerated():
            return (array(
                    [
                        [ones(n), zeros(n), zeros(n), zeros(n)],
                        [zeros(n), cos(a), -sin(a), zeros(n)],
                        [zeros(n), sin(a), cos(a), zeros(n)],
                        [zeros(n), zeros(n), zeros(n), ones(n)],
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
    expr = rotation(angle)
    func = sp.lambdify(angle, expr.doit(), cse=True)
    angle_array = np.arange(-2, 4, 1) * 2 * np.pi
    rotation_matrices = func(angle_array)
    identity = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    identity = np.tile(identity, reps=(len(angle_array), 1, 1))
    assert pytest.approx(rotation_matrices) == identity


class TestOnesZerosArray:
    @pytest.mark.parametrize("array_type", ["ones", "zeros"])
    @pytest.mark.parametrize("shape", [10, (4, 2), [3, 5, 7]])
    def test_numpycode(self, array_type, shape):
        if array_type == "ones":
            expr_class = _OnesArray
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


@pytest.mark.parametrize("use_cse", [False, True])
@pytest.mark.parametrize(
    ("angle_name", "expected_values"),
    [
        (
            "phi_1+2+3",
            np.array(
                [
                    2.79758,
                    2.51292,
                    -1.07396,
                    -1.88051,
                    1.06433,
                    -2.30129,
                    2.36878,
                    -2.46888,
                    0.568649,
                    -2.8792,
                ]
            ),
        ),
        (
            "theta_1+2+3",
            np.arccos(
                [
                    -0.914298,
                    -0.994127,
                    0.769715,
                    -0.918418,
                    0.462214,
                    0.958535,
                    0.496489,
                    -0.674376,
                    0.614968,
                    -0.0330843,
                ]
            ),
        ),
        (
            "phi_2+3,1+2+3",
            np.array(
                [
                    1.04362,
                    1.87349,
                    0.160733,
                    -2.81088,
                    2.84379,
                    2.29128,
                    2.24539,
                    -1.20272,
                    0.615838,
                    2.98067,
                ]
            ),
        ),
        (
            "theta_2+3,1+2+3",
            np.arccos(
                [
                    -0.772533,
                    0.163659,
                    0.556365,
                    0.133251,
                    -0.0264361,
                    0.227188,
                    -0.166924,
                    0.652761,
                    0.443122,
                    0.503577,
                ]
            ),
        ),
        (
            "phi_2,2+3,1+2+3",
            np.array(
                [  # WARNING: subsystem solution (ComPWA) results in pi differences
                    -2.77203 + np.pi,
                    1.45339 - np.pi,
                    -2.51096 + np.pi,
                    2.71085 - np.pi,
                    -1.12706 + np.pi,
                    -3.01323 + np.pi,
                    2.07305 - np.pi,
                    0.502648 - np.pi,
                    -1.23689 + np.pi,
                    1.7605 - np.pi,
                ]
            ),
        ),
        (
            "theta_2,2+3,1+2+3",
            np.arccos(
                [
                    0.460324,
                    -0.410464,
                    0.248566,
                    -0.301959,
                    -0.522502,
                    0.787267,
                    0.488066,
                    0.954167,
                    -0.553114,
                    0.00256349,
                ]
            ),
        ),
    ],
)
def test_compute_helicity_angles(  # pylint: disable=too-many-arguments
    use_cse: bool,
    data_sample: Dict[int, np.ndarray],
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta],
    angle_name: str,
    expected_values: np.ndarray,
    helicity_angles: Dict[str, sp.Expr],
):
    _, momentum_symbols = topology_and_momentum_symbols
    four_momenta = data_sample.values()
    expr = helicity_angles[angle_name]
    np_angle = sp.lambdify(momentum_symbols.values(), expr.doit(), cse=use_cse)
    computed = np_angle(*four_momenta)
    np.testing.assert_allclose(computed, expected_values, atol=1e-5)


def test_compute_invariant_masses_names(
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta]
):
    topology, momentum_symbols = topology_and_momentum_symbols
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)
    assert set(invariant_masses) == {
        "m_0",
        "m_1",
        "m_2",
        "m_3",
        "m_23",
        "m_123",
        "m_0123",
    }


def test_compute_invariant_masses_single_mass(
    data_sample: Dict[int, np.ndarray],
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    momentum_values = data_sample.values()
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)
    for i in topology.outgoing_edge_ids:
        expr = invariant_masses[f"m_{i}"]
        np_expr = sp.lambdify(momentum_symbols.values(), expr.doit(), "numpy")
        expected = __compute_mass(data_sample[i])
        computed = np_expr(*momentum_values)
        np.testing.assert_allclose(computed, expected, atol=1e-5)


@pytest.mark.parametrize("mass_name", ["m_23", "m_123", "m_0123"])
def test_compute_invariant_masses(
    mass_name: str,
    data_sample: Dict[int, np.ndarray],
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta],
):
    topology, momentum_symbols = topology_and_momentum_symbols
    momentum_values = data_sample.values()
    invariant_masses = compute_invariant_masses(momentum_symbols, topology)

    expr = invariant_masses[mass_name]
    np_expr = sp.lambdify(momentum_symbols.values(), expr.doit(), "numpy")
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


def test_determine_attached_final_state():
    topologies = create_isobar_topologies(4)
    # outer states
    for topology in topologies:
        for i in topology.outgoing_edge_ids:
            assert determine_attached_final_state(topology, state_id=i) == [i]
        for i in topology.incoming_edge_ids:
            assert determine_attached_final_state(
                topology, state_id=i
            ) == list(topology.outgoing_edge_ids)
    # intermediate states
    topology = topologies[0]
    assert determine_attached_final_state(topology, state_id=4) == [0, 1]
    assert determine_attached_final_state(topology, state_id=5) == [2, 3]
    topology = topologies[1]
    assert determine_attached_final_state(topology, state_id=4) == [1, 2, 3]
    assert determine_attached_final_state(topology, state_id=5) == [2, 3]


def _generate_numpy_code(expr: sp.Expr) -> str:
    printer = NumPyPrinter()
    return printer.doprint(expr)
