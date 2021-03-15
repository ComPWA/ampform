import pytest

from expertsystem import load_default_particles
from expertsystem.particle import ParticleCollection


@pytest.fixture(scope="session")
def particle_database() -> ParticleCollection:
    return load_default_particles()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"
