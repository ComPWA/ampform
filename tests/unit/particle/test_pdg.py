# pylint: disable=redefined-outer-name
import particle
import pytest

from expertsystem.particle import ParticleCollection, load_pdg


@pytest.fixture(scope="session")
def pdg() -> ParticleCollection:
    return load_pdg()


def test_maybe_qq():
    expected_maybe_qq = {
        "a(0)(980)+",
        "a(0)(980)-",
        "a(0)(980)0",
        "f(0)(1500)",
        "f(0)(500)",
        "f(0)(980)",
        "pi(1)(1400)+",
        "pi(1)(1400)-",
        "pi(1)(1400)0",
        "pi(1)(1600)+",
        "pi(1)(1600)-",
        "pi(1)(1600)0",
    }
    maybe_qq_search_results = particle.Particle.findall(
        lambda p: "qq" in p.quarks.lower()
    )
    assert expected_maybe_qq == {item.name for item in maybe_qq_search_results}


def test_pdg_size(pdg: ParticleCollection):
    assert len(pdg) in [
        512,  # particle==0.13
        519,  # particle==0.14
    ]
    assert len(pdg.filter(lambda p: "~" in p.name)) in [
        165,  # particle==0.13
        172,  # particle==0.14
    ]


def test_missing_in_pdg(
    pdg: ParticleCollection,
    particle_database: ParticleCollection,
):
    particle_list_names = set(particle_database)
    pdg_names = set(pdg)
    in_common = particle_list_names & pdg_names
    missing_in_pdg = particle_list_names ^ in_common
    assert {p.name for p in missing_in_pdg} == {
        "Y(4260)",
    }
