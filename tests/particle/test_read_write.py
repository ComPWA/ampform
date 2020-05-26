from expertsystem.state import particle
from expertsystem.state.particle import particle_list
from expertsystem.ui.system_control import load_default_particle_list


def test_import_xml() -> None:
    load_default_particle_list()
    assert len(particle_list) == 70
    assert "sigma+" in particle_list.keys()
    assert "mu+" in particle_list.keys()

    some_particle = particle_list["gamma"]
    quantum_numbers = some_particle[particle.LABELS.QuantumNumber.name]
    quantum_number = quantum_numbers[0]
    assert (
        quantum_number[particle.LABELS.Class.name]
        == particle.StateQuantumNumberNames.Spin.name
    )
    assert int(quantum_number[particle.LABELS.Value.name]) == 1


def test_xml_io() -> None:
    load_default_particle_list()
    particle.write_particle_list_to_xml("test_particle_list.xml")
    particle_list.clear()
    particle.load_particle_list_from_xml("test_particle_list.xml")
    assert len(particle_list) == 70
