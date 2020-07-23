import pytest

from expertsystem.state import particle
from expertsystem.ui import load_default_particle_list


load_default_particle_list()


class TestFind:
    @staticmethod
    def test_exceptions():
        with pytest.raises(LookupError):
            particle.find_particle(666)
        with pytest.raises(NotImplementedError):
            particle.find_particle(float())

    @staticmethod
    def test_pid_search():
        assert len(particle.DATABASE) == 69
        omega = particle.find_particle(223)
        assert omega["Name"] == "omega(782)"

    @staticmethod
    def test_name_search():
        omega = particle.find_particle("omega")
        assert isinstance(omega, dict)
        assert omega["Pid"] == "223"

    @staticmethod
    def test_name_slice_search():
        f_resonances = particle.find_particle("f0")
        assert len(f_resonances) == 2
        no_search_results = particle.find_particle("non-existing")
        assert no_search_results == dict()
