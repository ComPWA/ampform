import pytest

from expertsystem import io
from expertsystem.particle import ParticleCollection


def test_asdict_fromdict(particle_selection: ParticleCollection):
    asdict = io.asdict(particle_selection)
    fromdict = io.fromdict(asdict)
    assert particle_selection == fromdict
    for particle in particle_selection:
        asdict = io.asdict(particle)
        fromdict = io.fromdict(asdict)
        assert particle == fromdict


def test_fromdict_exceptions():
    with pytest.raises(NotImplementedError):
        io.fromdict({"non-sense": 1})
