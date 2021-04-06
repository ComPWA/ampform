# cspell:ignore atol
# pylint: disable=no-self-use
import numpy as np
import pytest
from qrules.topology import create_isobar_topologies

from ampform.data import EventCollection
from ampform.kinematics import (
    _compute_helicity_angles,
    _compute_invariant_masses,
)


def test_compute_helicity_angles(data_sample: EventCollection):
    expected_angles = {
        "phi_1+2+3": np.array(
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
        "theta_1+2+3": np.arccos(
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
        "phi_2+3,1+2+3": np.array(
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
        "theta_2+3,1+2+3": np.arccos(
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
        "phi_2,2+3,1+2+3": np.array(
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
        "theta_2,2+3,1+2+3": np.arccos(
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
    }
    topologies = create_isobar_topologies(4)
    topology = topologies[1]
    angles = _compute_helicity_angles(data_sample, topology)
    assert len(angles) == len(expected_angles)
    assert set(angles) == set(expected_angles)
    for angle_name in angles:
        np.testing.assert_allclose(
            angles[angle_name],
            expected_angles[angle_name],
            atol=1e-5,
        )


def test_compute_invariant_masses(data_sample: EventCollection):
    topologies = create_isobar_topologies(4)
    topology = topologies[1]
    invariant_masses = _compute_invariant_masses(data_sample, topology)
    assert set(invariant_masses) == {
        "m_0",
        "m_0123",
        "m_1",
        "m_123",
        "m_2",
        "m_23",
        "m_3",
    }
    for i in topology.outgoing_edge_ids:
        inv_mass = invariant_masses[f"m_{i}"]
        assert pytest.approx(inv_mass) == (data_sample[i].mass())
    jpsi_mass = np.average(invariant_masses["m_0123"])
    assert pytest.approx(jpsi_mass, abs=1e-5) == 3.0969
    assert (
        pytest.approx(invariant_masses["m_123"])
        == data_sample.sum([1, 2, 3]).mass()
    )
    assert (
        pytest.approx(invariant_masses["m_23"])
        == data_sample.sum([2, 3]).mass()
    )
