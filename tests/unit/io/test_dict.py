# pylint: disable=redefined-outer-name
import pytest

from expertsystem import io
from expertsystem.particle import ParticleCollection


def test_not_implemented_errors(
    output_dir: str, particle_selection: ParticleCollection
):
    with pytest.raises(NotImplementedError):
        io.load(__file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, output_dir + "test.py")
    with pytest.raises(Exception):
        io.write(particle_selection, output_dir + "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, output_dir + "wont_work_anyway.yml")


def test_serialization(
    output_dir: str, particle_selection: ParticleCollection
):
    io.write(particle_selection, output_dir + "particle_selection.yml")
    assert len(particle_selection) == 181
    asdict = io.asdict(particle_selection)
    imported_collection = io.fromdict(asdict)
    assert isinstance(imported_collection, ParticleCollection)
    assert len(particle_selection) == len(imported_collection)
    for particle in particle_selection:
        exported = particle_selection[particle.name]
        imported = imported_collection[particle.name]
        assert imported == exported
