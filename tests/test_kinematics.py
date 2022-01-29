# pylint: disable=no-member, no-self-use, redefined-outer-name
# cspell:ignore atol doprint
from typing import Dict, Tuple

import numpy as np
import pytest
import sympy as sp
from numpy.lib.scimath import sqrt as complex_sqrt
from qrules.topology import Topology, create_isobar_topologies
from sympy.printing.numpy import NumPyPrinter

from ampform.kinematics import (
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
    Phi,
    Theta,
    ThreeMomentum,
    compute_helicity_angles,
    compute_invariant_masses,
    create_four_momentum_symbols,
    three_momentum_norm,
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


class TestBoostMatrix:
    def test_boost_in_z_direction_reduces_to_z_boost(self):
        p = FourMomentumSymbol("p")
        expr = BoostMatrix(p)
        func = sp.lambdify(p, expr)
        p_array = np.array([[5, 0, 0, 1]])
        matrix = func(p_array)[0]
        assert pytest.approx(matrix) == np.array(
            [
                [1.02062073, 0, 0, -0.20412415],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-0.20412415, 0, 0, 1.02062073],
            ]
        )

        beta = three_momentum_norm(p) / Energy(p)
        z_expr = BoostZMatrix(beta)
        z_func = sp.lambdify(p, z_expr.doit())
        z_matrix = z_func(p_array)[0]
        assert pytest.approx(matrix) == z_matrix

    def test_boost_into_rest_frame_gives_mass(
        self,
        data_sample: Dict[int, np.ndarray],
        topology_and_momentum_symbols: Tuple[Topology, FourMomenta],
    ):
        # pylint: disable=too-many-locals
        _, momenta = topology_and_momentum_symbols
        pi0_mass = 0.135
        masses = {0: pi0_mass, 1: 0, 2: pi0_mass, 3: pi0_mass}
        for state_id, momentum_array in data_sample.items():
            momentum = momenta[state_id]
            boost = BoostMatrix(momentum)
            expr = ArrayMultiplication(boost, momentum)
            func = sp.lambdify(momentum, expr)
            boosted_array: np.ndarray = func(momentum_array)
            assert not np.all(np.isnan(boosted_array))
            mass = boosted_array[:, 0]
            mass = mass[~np.isnan(mass)]
            assert pytest.approx(mass, abs=1e-2) == masses[state_id]
            p_xyz = boosted_array[:, 1:]
            p_xyz = np.nan_to_num(p_xyz, nan=0)
            assert pytest.approx(p_xyz) == 0


class TestBoostZMatrix:
    def test_boost_into_own_rest_frame_gives_mass(self):
        p = FourMomentumSymbol("p")
        beta = three_momentum_norm(p) / Energy(p)
        expr = BoostZMatrix(beta)
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


class TestFourMomentumXYZ:
    def symbols(
        self,
    ) -> Tuple[
        ArraySymbol, Energy, FourMomentumX, FourMomentumY, FourMomentumZ
    ]:
        p = ArraySymbol("p")
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
        p = ArraySymbol(f"p{state_id}")
        mass = InvariantMass(p)
        np_mass = sp.lambdify(p, mass.doit(), "numpy")
        four_momenta = data_sample[state_id]
        computed_values = np_mass(four_momenta)
        average_mass = np.average(computed_values)
        assert pytest.approx(average_mass, abs=1e-5) == expected_mass


class TestThreeMomentum:
    @property
    def p_norm(self) -> ThreeMomentum:
        p = ArraySymbol("p")
        return ThreeMomentum(p)

    def test_latex(self):
        latex = sp.latex(self.p_norm)
        assert latex == R"\vec{p}"

    def test_numpy(self):
        numpy_code = _generate_numpy_code(self.p_norm)
        assert numpy_code == "p[:, 1:]"


class TestPhi:
    @property
    def phi(self) -> Theta:
        p = ArraySymbol("p")
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
        p = ArraySymbol("p")
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


class TestNegativeMomentum:
    def test_same_as_inverse(self, data_sample: Dict[int, np.ndarray]):
        p = FourMomentumSymbol("p")
        expr = NegativeMomentum(p)
        func = sp.lambdify(p, expr.doit())
        for p_array in data_sample.values():
            negative_array = func(p_array)
            assert pytest.approx(negative_array[:, 0]) == p_array[:, 0]
            assert pytest.approx(negative_array[:, 1:]) == -p_array[:, 1:]


@pytest.mark.parametrize(
    ("angle_name", "expected_values"),
    [
        (
            "phi_0",
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
            "theta_0",
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
            "phi_1^123",
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
            "theta_1^123",
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
            "phi_2^23,123",
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
            "theta_2^23,123",
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
def test_compute_helicity_angles(
    data_sample: Dict[int, np.ndarray],
    topology_and_momentum_symbols: Tuple[Topology, FourMomenta],
    angle_name: str,
    expected_values: np.ndarray,
    helicity_angles: Dict[str, sp.Expr],
):
    _, momentum_symbols = topology_and_momentum_symbols
    four_momenta = data_sample.values()
    expr = helicity_angles[angle_name]
    np_angle = sp.lambdify(momentum_symbols.values(), expr.doit())
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


def _generate_numpy_code(expr: sp.Expr) -> str:
    printer = NumPyPrinter()
    return printer.doprint(expr)
