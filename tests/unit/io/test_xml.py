# pylint: disable=protected-access

from expertsystem import io
from expertsystem.particle import Particle, Spin


def test_particle():
    state = Particle(
        "MyParticle",
        pid=123,
        mass=2.5,
        width=0.3,
        spin=1.5,
        isospin=Spin(1.0, -1.0),
        charge=-1,
    )
    converted_dict = io._xml.object_to_dict(state)
    quantum_numbers = converted_dict["QuantumNumber"]
    spin_dict = quantum_numbers[0]
    charge_dict = quantum_numbers[1]
    assert spin_dict["Value"] == 1.5
    assert charge_dict["Value"] == -1
