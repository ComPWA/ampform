# cspell:ignore atol doprint
# pylint: disable=redefined-outer-name
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from qrules.topology import Topology
from sympy.printing.numpy import NumPyPrinter

from ampform.helicity.decay import get_parent_id
from ampform.kinematics.angles import (
    Phi,
    Theta,
    compute_helicity_angles,
    compute_wigner_rotation_matrix,
)
from ampform.kinematics.lorentz import FourMomenta, FourMomentumSymbol


@pytest.fixture(scope="session")
def helicity_angles(
    topology_and_momentum_symbols: tuple[Topology, FourMomenta]
) -> dict[sp.Symbol, sp.Expr]:
    topology, momentum_symbols = topology_and_momentum_symbols
    return compute_helicity_angles(momentum_symbols, topology)


class TestPhi:
    @property
    def phi(self):
        p = FourMomentumSymbol("p", shape=[])
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
    def theta(self):
        p = FourMomentumSymbol("p", shape=[])
        return Theta(p)

    def test_latex(self):
        latex = sp.latex(self.theta)
        assert latex == R"\theta\left(p\right)"

    def test_numpy(self):
        theta = self.theta.doit()
        numpy_code = _generate_numpy_code(theta)
        assert (
            numpy_code == "numpy.arccos(p[:, 3]/numpy.sqrt(sum(p[:, 1:]**2, axis=1)))"
        )


@pytest.mark.parametrize("use_cse", [False, True])
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
def test_compute_helicity_angles(  # pylint: disable=too-many-arguments
    use_cse: bool,
    data_sample: dict[int, np.ndarray],
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
    angle_name: str,
    expected_values: np.ndarray,
    helicity_angles: dict[sp.Symbol, sp.Expr],
):
    _, momentum_symbols = topology_and_momentum_symbols
    four_momenta = data_sample.values()
    angle_symbol = sp.Symbol(angle_name, real=True)
    expr = helicity_angles[angle_symbol]
    np_angle = sp.lambdify(momentum_symbols.values(), expr.doit(), cse=use_cse)
    computed = np_angle(*four_momenta)
    np.testing.assert_allclose(computed, expected_values, atol=1e-5)


@pytest.mark.parametrize(
    ("state_id", "expected"),
    [
        (
            0,
            "MatrixMultiplication(BoostMatrix(NegativeMomentum(p0)), BoostMatrix(p0))",
        ),
        (
            1,
            "MatrixMultiplication(BoostMatrix(NegativeMomentum(p1)),"
            " BoostMatrix(p1 + p2 + p3),"
            " BoostMatrix(ArrayMultiplication(BoostMatrix(p1 + p2 + p3),"
            " p1)))",
        ),
        (
            2,
            "MatrixMultiplication(BoostMatrix(NegativeMomentum(p2)), BoostMatrix(p1 +"
            " p2 + p3), BoostMatrix(ArrayMultiplication(BoostMatrix(p1 + p2 + p3), p2 +"
            " p3)),"
            " BoostMatrix(ArrayMultiplication(BoostMatrix(ArrayMultiplication(BoostMatrix(p1"
            " + p2 + p3), p2 + p3)), ArrayMultiplication(BoostMatrix(p1 + p2 + p3),"
            " p2))))",
        ),
        (
            3,
            "MatrixMultiplication(BoostMatrix(NegativeMomentum(p3)), BoostMatrix(p1 +"
            " p2 + p3), BoostMatrix(ArrayMultiplication(BoostMatrix(p1 + p2 + p3), p2 +"
            " p3)),"
            " BoostMatrix(ArrayMultiplication(BoostMatrix(ArrayMultiplication(BoostMatrix(p1"
            " + p2 + p3), p2 + p3)), ArrayMultiplication(BoostMatrix(p1 + p2 + p3),"
            " p3))))",
        ),
    ],
)
def test_compute_wigner_rotation_matrix(
    state_id: int,
    expected: str,
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momenta = topology_and_momentum_symbols
    expr = compute_wigner_rotation_matrix(topology, momenta, state_id)
    assert str(expr) == expected


@pytest.mark.parametrize(
    "state_id",
    [
        0,
        pytest.param(2, marks=pytest.mark.slow),
        pytest.param(3, marks=pytest.mark.slow),
    ],
)
def test_compute_wigner_rotation_matrix_numpy(
    state_id: int,
    data_sample: dict[int, np.ndarray],
    topology_and_momentum_symbols: tuple[Topology, FourMomenta],
):
    topology, momenta = topology_and_momentum_symbols
    expr = compute_wigner_rotation_matrix(topology, momenta, state_id)
    func = sp.lambdify(momenta.values(), expr.doit(), cse=True)
    momentum_array = data_sample[state_id]
    wigner_matrix_array = func(*data_sample.values())
    assert wigner_matrix_array.shape == (len(momentum_array), 4, 4)
    if get_parent_id(topology, state_id) == -1:
        product = np.einsum("...ij,...j->...j", wigner_matrix_array, momentum_array)
        assert pytest.approx(product) == momentum_array
    matrix_column_norms = np.linalg.norm(wigner_matrix_array, axis=1)
    assert pytest.approx(matrix_column_norms) == 1


def _generate_numpy_code(expr: sp.Expr) -> str:
    printer = NumPyPrinter()
    return printer.doprint(expr)
