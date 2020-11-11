import pytest

from expertsystem.particle import Parity
from expertsystem.reaction.conservation_rules import (
    IdenticalParticleSymmetryOutEdgeInput,
    identical_particle_symmetrization,
)
from expertsystem.reaction.quantum_numbers import EdgeQuantumNumbers

# Currently need to cast to the proper Edge/NodeQuantumNumber type, see
# https://github.com/ComPWA/expertsystem/issues/255


@pytest.mark.parametrize(
    "in_edges, out_edges, expected",
    [
        (
            [
                EdgeQuantumNumbers.parity(Parity(parity)),
            ],
            [
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(1.0),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(1.0),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
            ],
            parity == 1,
        )
        for parity in [-1, 1]
    ],
)
def test_identical_boson_symmetrization(in_edges, out_edges, expected):
    assert identical_particle_symmetrization(in_edges, out_edges) is expected


@pytest.mark.parametrize(
    "in_edges, out_edges, expected",
    [
        (
            [
                EdgeQuantumNumbers.parity(Parity(parity)),
            ],
            [
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(0.5),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(0.5),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
            ],
            parity == -1,
        )
        for parity in [-1, 1]
    ],
)
def test_identical_fermion_symmetrization(in_edges, out_edges, expected):
    assert identical_particle_symmetrization(in_edges, out_edges) is expected


@pytest.mark.parametrize(
    "in_edges, out_edges, expected",
    [
        (
            [
                EdgeQuantumNumbers.parity(Parity(parity)),
            ],
            [
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(1.0),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(1.0),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(-10),
                ),
            ],
            True,
        )
        for parity in [-1, 1]
    ],
)
def test_nonidentical_particle_symmetrization(in_edges, out_edges, expected):
    assert identical_particle_symmetrization(in_edges, out_edges) is expected


@pytest.mark.parametrize(
    "in_edges, out_edges, expected",
    [
        (
            [
                EdgeQuantumNumbers.parity(Parity(parity)),
            ],
            [
                IdenticalParticleSymmetryOutEdgeInput(
                    spin_magnitude=EdgeQuantumNumbers.spin_magnitude(1.0),
                    spin_projection=EdgeQuantumNumbers.spin_projection(0),
                    pid=EdgeQuantumNumbers.pid(10),
                ),
            ],
            True,
        )
        for parity in [-1, 1]
    ],
)
def test_single_particle_case(in_edges, out_edges, expected):
    assert identical_particle_symmetrization(in_edges, out_edges) is expected
