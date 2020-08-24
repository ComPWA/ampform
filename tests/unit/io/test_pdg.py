# pylint: disable=redefined-outer-name
from particle import Particle

import pytest

from expertsystem import io


@pytest.fixture(scope="module")
def pdg():
    return io.load_pdg()


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
    maybe_qq_search_results = Particle.findall(
        lambda p: "qq" in p.quarks.lower()
    )
    assert expected_maybe_qq == {item.name for item in maybe_qq_search_results}


def test_pdg_size(pdg):
    assert len(pdg) == 532
    assert len(pdg.find_subset("~")) == 166


def test_missing_in_pdg(pdg, particle_database):
    particle_list_names = set(particle_database)
    pdg_names = set(pdg)
    in_common = particle_list_names & pdg_names
    missing_in_pdg = particle_list_names ^ in_common
    assert missing_in_pdg == {
        "Y(4260)",
    }


def test_pdg_entries(pdg, particle_database):
    for name in particle_database:
        if name not in pdg:
            continue
        internal_particle = particle_database[name]
        pdg_particle = pdg[name]
        assert pdg_particle.name == internal_particle.name
        assert pdg_particle.pid == internal_particle.pid
        assert pdg_particle.state == internal_particle.state
        assert pdg_particle.mass == pytest.approx(internal_particle.mass)
        assert pdg_particle.width == pytest.approx(internal_particle.width)
