import pytest

from expertsystem.data import ParticleCollection, create_particle


def test_find(particle_database: ParticleCollection):
    f2_1950 = particle_database.find(9050225)
    assert f2_1950.name == "f(2)(1950)"
    assert f2_1950.mass == 1.936
    phi = particle_database.find("phi(1020)")
    assert phi.pid == 333
    assert pytest.approx(phi.width) == 0.004249


@pytest.mark.parametrize("search_term", [666, "non-existing"])
def test_find_fail(particle_database: ParticleCollection, search_term):
    with pytest.raises(LookupError):
        particle_database.find(search_term)


def test_filter(particle_database: ParticleCollection):
    search_result = particle_database.filter(lambda p: "f(0)" in p.name)
    f0_1500_from_subset = search_result["f(0)(1500)"]
    assert len(search_result) == 5
    assert f0_1500_from_subset.mass == 1.506
    assert f0_1500_from_subset is particle_database["f(0)(1500)"]
    assert f0_1500_from_subset is not particle_database["f(0)(980)"]

    # test iadd
    particle_database += search_result

    search_result = particle_database.filter(lambda p: p.pid == 22)
    gamma_from_subset = search_result["gamma"]
    assert len(search_result) == 1
    assert gamma_from_subset.pid == 22
    assert gamma_from_subset is particle_database["gamma"]
    filtered_result = particle_database.filter(
        lambda p: p.mass > 1.8
        and p.mass < 2.0
        and p.spin == 2
        and p.strangeness == 1
    )
    assert set(filtered_result) == {"K(2)(1820)0", "K(2)(1820)+"}


def test_exceptions(particle_database):
    gamma_1 = create_particle(particle_database["gamma"], name="gamma_1")
    gamma_2 = create_particle(particle_database["gamma"], name="gamma_2")
    particle_database += gamma_1
    particle_database += gamma_2
    with pytest.raises(NotImplementedError):
        particle_database.find(3.14)  # type: ignore
    with pytest.raises(NotImplementedError):
        particle_database += 3.14  # type: ignore
    with pytest.raises(AssertionError):
        assert gamma_1 == "gamma"
