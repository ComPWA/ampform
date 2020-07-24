from os.path import dirname, realpath

import pytest

import expertsystem
from expertsystem import io
from expertsystem import ui
from expertsystem.data import (
    MeasuredValue,
    Parity,
    Particle,
    ParticleCollection,
)
from expertsystem.state import particle

EXPERTSYSTEM_PATH = dirname(realpath(expertsystem.__file__))
_XML_FILE = f"{EXPERTSYSTEM_PATH}/particle_list.xml"
_YAML_FILE = f"{EXPERTSYSTEM_PATH}/particle_list.yml"

J_PSI = Particle(
    name="J/psi",
    pid=443,
    mass=MeasuredValue(3.0969),
    width=MeasuredValue(9.29e-05),
    spin=1,
    charge=0,
    parity=Parity(-1),
    c_parity=Parity(-1),
    g_parity=Parity(-1),
)


def test_not_implemented_errors():
    with pytest.raises(NotImplementedError):
        io.load_particle_collection(f"{EXPERTSYSTEM_PATH}/../README.md")
    with pytest.raises(NotImplementedError):
        dummy = ParticleCollection()
        io.write(dummy, f"{EXPERTSYSTEM_PATH}/particle_list.csv")
    with pytest.raises(Exception):
        dummy = ParticleCollection()
        io.write(dummy, "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, "wont_work_anyway.xml")


@pytest.mark.parametrize("input_file", [_XML_FILE, _YAML_FILE])
def test_load_particle_collection(input_file):
    particles = io.load_particle_collection(input_file)
    assert len(particles) == 69
    assert "J/psi" in particles
    j_psi = particles["J/psi"]
    assert j_psi == J_PSI
    particle_names = list(particles.keys())
    for name, particle_name in zip(particle_names, particles):
        assert name == particle_name


@pytest.mark.parametrize("input_file", [_XML_FILE, _YAML_FILE])
def test_write_particle_collection(input_file):
    particles_imported = io.load_particle_collection(input_file)
    file_extension = input_file.split(".")[-1]
    output_file = f"exported_particle_list.{file_extension}"
    io.write(particles_imported, output_file)
    particles_exported = io.load_particle_collection(output_file)
    assert particles_imported == particles_exported


def test_yaml_to_xml():
    yaml_particle_collection = io.load_particle_collection(_YAML_FILE)
    xml_file = "particle_list_test.xml"
    io.write(yaml_particle_collection, xml_file)
    xml_particle_collection = io.load_particle_collection(xml_file)
    assert xml_particle_collection == yaml_particle_collection
    dummy_particle = Particle(name="0", pid=0, charge=0, spin=0, mass=0)
    yaml_particle_collection.add(dummy_particle)
    assert xml_particle_collection != yaml_particle_collection


def test_equivalence_xml_yaml_particle_list():
    xml_particle_collection = io.load_particle_collection(_XML_FILE)
    yml_particle_collection = io.load_particle_collection(_YAML_FILE)
    assert xml_particle_collection == yml_particle_collection


class TestInternalParticleDict:
    ui.load_default_particle_list()

    @staticmethod
    def test_particle_validation():
        for item in particle.DATABASE.values():
            io.xml.validation.particle(item)

    @staticmethod
    def test_build_particle_from_internal_database():
        definition = particle.DATABASE["J/psi"]
        j_psi = io.xml.dict_to_particle(definition)
        assert j_psi == J_PSI

    @staticmethod
    def test_dump_via_particle_collection():
        particles = io.xml.dict_to_particle_collection(particle.DATABASE)
        io.write(particles, "DATABASE_via_ParticleCollection.xml")
