# pylint: disable=redefined-outer-name
import pytest

from expertsystem import io
from expertsystem.particle import ParticleCollection

XML_FILE = "particle_selection.xml"
YAML_FILE = "particle_selection.yml"


@pytest.fixture(scope="session")
def particle_selection(output_dir, particle_database: ParticleCollection):
    selection = ParticleCollection()
    selection += particle_database.filter(lambda p: p.name.startswith("pi"))
    selection += particle_database.filter(lambda p: p.name.startswith("K"))
    selection += particle_database.filter(lambda p: p.name.startswith("D"))
    selection += particle_database.filter(lambda p: p.name.startswith("J/psi"))
    io.write(selection, output_dir + XML_FILE)
    io.write(selection, output_dir + YAML_FILE)
    return selection


def test_not_implemented_errors(
    output_dir, particle_selection: ParticleCollection
):
    with pytest.raises(NotImplementedError):
        io.load_particle_collection(output_dir + __file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, output_dir + __file__)
    with pytest.raises(Exception):
        io.write(particle_selection, output_dir + "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, output_dir + "wont_work_anyway.xml")


@pytest.mark.parametrize("filename", [XML_FILE, YAML_FILE])
def test_write_read_particle_collection(
    output_dir, particle_selection: ParticleCollection, filename: str
):
    assert len(particle_selection) == 181
    imported_collection = io.load_particle_collection(output_dir + filename)
    assert len(particle_selection) == len(imported_collection)
    for particle in particle_selection:
        exported = particle_selection[particle.name]
        imported = imported_collection[particle.name]
        assert imported == exported


def test_equivalence_xml_yaml_particle_list(output_dir):
    xml_particles = io.load_particle_collection(output_dir + XML_FILE)
    yml_particles = io.load_particle_collection(output_dir + YAML_FILE)
    for particle in xml_particles:
        xml_particle = xml_particles[particle.name]
        yml_particle = yml_particles[particle.name]
        assert xml_particle == yml_particle
        assert xml_particle is not yml_particle
