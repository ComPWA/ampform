# pylint: disable=redefined-outer-name
import pytest

from expertsystem import io
from expertsystem.data import ParticleCollection


XML_FILE = "particle_list.xml"
YAML_FILE = "particle_list.yml"


@pytest.fixture(scope="package")
def particle_selection(particle_database):
    selection = ParticleCollection()
    selection += particle_database.find_subset("pi")
    selection += particle_database.find_subset("K")
    selection += particle_database.find_subset("D")
    selection += particle_database.find_subset("J/psi")
    return selection


def write_test_files(particle_selection):
    io.write(particle_selection, XML_FILE)
    io.write(particle_selection, YAML_FILE)


def test_not_implemented_errors(particle_selection):
    with pytest.raises(NotImplementedError):
        io.load_particle_collection(__file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, __file__)
    with pytest.raises(Exception):
        io.write(particle_selection, "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, "wont_work_anyway.xml")


@pytest.mark.parametrize("filename", [XML_FILE, YAML_FILE])
def test_write_read_particle_collection(particle_selection, filename):
    write_test_files(particle_selection)
    assert len(particle_selection) == 202
    imported_collection = io.load_particle_collection(filename)
    assert len(particle_selection) == len(imported_collection)
    for name in particle_selection:
        assert particle_selection[name] == imported_collection[name]


def test_equivalence_xml_yaml_particle_list(particle_selection):
    write_test_files(particle_selection)
    xml_particle_collection = io.load_particle_collection(XML_FILE)
    yml_particle_collection = io.load_particle_collection(YAML_FILE)
    for name in xml_particle_collection:
        assert xml_particle_collection[name] == yml_particle_collection[name]
