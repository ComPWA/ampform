import pytest

from expertsystem.state.particle import (
    DATABASE,
    create_antiparticle,
    create_particle,
)
from expertsystem.ui import load_default_particle_list

load_default_particle_list()


@pytest.mark.parametrize(
    "particle_name", ["p", "phi(1020)", "W-", "gamma"],
)
def test_create_particle(particle_name):
    template_particle = DATABASE[particle_name]
    new_particle = create_particle(
        template_particle,
        name="testparticle",
        pid=89,
        charge=0.12,
        spin=3 / 2,
        muon_lepton_number=4,
        width=0.5,
    )

    assert new_particle.name == "testparticle"
    assert new_particle.pid == 89
    assert new_particle.state.charge == 0.12
    assert new_particle.state.spin == 1.5
    assert new_particle.state.muon_lepton_number == 4
    assert new_particle.width == 0.5
    assert (
        new_particle.state.baryon_number
        == template_particle.state.baryon_number
    )
    assert (
        new_particle.state.strangeness == template_particle.state.strangeness
    )
    assert new_particle.mass == template_particle.mass


@pytest.mark.parametrize(
    "particle_name, anti_particle_name",
    [("D+", "D-"), ("p", "pbar"), ("mu+", "mu-"), ("W+", "W-")],
)
def test_create_antiparticle(particle_name, anti_particle_name):
    template_particle = DATABASE[particle_name]
    anti_particle = create_antiparticle(template_particle)
    comparison_particle = DATABASE[anti_particle_name]

    assert anti_particle.pid == comparison_particle.pid
    assert anti_particle.mass == comparison_particle.mass
    assert anti_particle.width == comparison_particle.width
    assert anti_particle.state == comparison_particle.state
    assert anti_particle.name == "anti-" + particle_name
