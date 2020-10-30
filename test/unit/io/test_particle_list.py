# pylint: disable=redefined-outer-name
import pytest

from expertsystem import io
from expertsystem.particle import ParticleCollection

XML_FILE = "particle_list.xml"
YAML_FILE = "particle_list.yml"


@pytest.fixture(scope="module")
def particle_selection(particle_database: ParticleCollection):
    selection = ParticleCollection()
    selection += particle_database.filter(lambda p: "pi" in p.name)
    selection += particle_database.filter(lambda p: "K" in p.name)
    selection += particle_database.filter(lambda p: "D" in p.name)
    selection += particle_database.filter(lambda p: "J/psi" in p.name)
    return selection


def write_test_files(particle_selection):
    io.write(particle_selection, XML_FILE)
    io.write(particle_selection, YAML_FILE)


def test_not_implemented_errors(particle_selection: ParticleCollection):
    with pytest.raises(NotImplementedError):
        io.load_particle_collection(__file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, __file__)
    with pytest.raises(Exception):
        io.write(particle_selection, "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, "wont_work_anyway.xml")


@pytest.mark.parametrize("filename", [XML_FILE, YAML_FILE])
def test_write_read_particle_collection(
    particle_selection: ParticleCollection, filename: str
):
    write_test_files(particle_selection)
    assert len(particle_selection) == 182
    imported_collection = io.load_particle_collection(filename)
    assert len(particle_selection) == len(imported_collection)
    for particle in particle_selection:
        exported = particle_selection[particle.name]
        imported = imported_collection[particle.name]
        assert imported == exported


def test_equivalence_xml_yaml_particle_list(
    particle_selection: ParticleCollection,
):
    write_test_files(particle_selection)
    xml_particle_collection = io.load_particle_collection(XML_FILE)
    yml_particle_collection = io.load_particle_collection(YAML_FILE)
    for particle in xml_particle_collection:
        xml_particle = xml_particle_collection[particle.name]
        yml_particle = yml_particle_collection[particle.name]
        assert xml_particle == yml_particle
        assert xml_particle is not yml_particle
