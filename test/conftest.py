import pytest

from expertsystem.particle import ParticleCollection
from expertsystem.reaction import load_default_particles


@pytest.fixture(scope="module")
def particle_database() -> ParticleCollection:
    return load_default_particles()


@pytest.fixture(scope="module")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/test/output/"
