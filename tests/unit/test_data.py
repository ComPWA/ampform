# pylint: disable=redefined-outer-name, no-self-use
import typing
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from expertsystem.data import (
    GellmannNishijima,
    Parity,
    Particle,
    ParticleCollection,
    Spin,
    create_antiparticle,
    create_particle,
)


class TestGellmannNishijima:
    @staticmethod
    @pytest.mark.parametrize(
        "state",
        [
            Particle(
                "p1",
                1,
                spin=0.0,
                mass=1,
                charge=1,
                isospin=Spin(1.0, 0.0),
                strangeness=2,
            ),
            Particle(
                "p1",
                1,
                spin=1.0,
                mass=1,
                charge=1,
                isospin=Spin(1.5, 0.5),
                charmness=1,
            ),
            Particle(
                "p1",
                1,
                spin=0.5,
                mass=1,
                charge=1.5,  # type: ignore
                isospin=Spin(1.0, 1.0),
                baryon_number=1,
            ),
        ],
    )
    def test_computations(state):
        assert GellmannNishijima.compute_charge(state) == state.charge
        assert (
            GellmannNishijima.compute_isospin_projection(
                charge=state.charge,
                baryon_number=state.baryon_number,
                strangeness=state.strangeness,
                charmness=state.charmness,
                bottomness=state.bottomness,
                topness=state.topness,
            )
            == state.isospin.projection
        )

    @staticmethod
    def test_isospin_none():
        state = Particle("p1", 1, mass=1, spin=0.0, charge=1, isospin=None)
        assert GellmannNishijima.compute_charge(state) is None


class TestParity:
    @staticmethod
    def test_init_and_eq():
        parity = Parity(+1)
        assert parity == +1
        assert int(parity) == +1

    @typing.no_type_check  # https://github.com/python/mypy/issues/4610
    @staticmethod
    def test_comparison():
        neg = Parity(-1)
        pos = Parity(+1)
        assert pos > 0
        assert neg < 0
        assert neg < pos
        assert neg <= pos
        assert pos > neg
        assert pos >= neg
        assert pos >= 0
        assert neg <= 0
        assert 0 < pos  # pylint: disable=misplaced-comparison-constant

    @staticmethod
    def test_hash():
        neg = Parity(-1)
        pos = Parity(+1)
        assert {pos, neg, deepcopy(pos)} == {neg, pos}

    @staticmethod
    def test_neg():
        parity = Parity(+1)
        flipped_parity = -parity
        assert flipped_parity.value == -parity.value

    @pytest.mark.parametrize("value", [-1, +1])
    def test_repr(self, value):
        parity = Parity(value)
        from_repr = eval(repr(parity))  # pylint: disable=eval-used
        assert from_repr == parity

    @staticmethod
    def test_exceptions():
        with pytest.raises(ValueError):
            Parity(1.2)


class TestParticle:
    @staticmethod
    def test_repr(particle_database: ParticleCollection):
        for particle in particle_database.values():
            from_repr = eval(repr(particle))  # pylint: disable=eval-used
            assert from_repr == particle

    @pytest.mark.parametrize(
        "name, is_lepton",
        [
            ("J/psi(1S)", False),
            ("p", False),
            ("e+", True),
            ("e-", True),
            ("nu(e)", True),
            ("nu(tau)~", True),
            ("tau+", True),
        ],
    )
    def test_is_lepton(
        self, name, is_lepton, particle_database: ParticleCollection
    ):
        assert particle_database[name].is_lepton() == is_lepton

    @staticmethod
    def test_exceptions():
        with pytest.raises(FrozenInstanceError):
            test_state = Particle(
                "MyParticle",
                123,
                mass=1.2,
                width=0.1,
                spin=1,
                charge=0,
                isospin=Spin(1, 0),
            )
            test_state.charge = 1  # type: ignore
        with pytest.raises(ValueError):
            Particle(
                name="Fails Gell-Mann–Nishijima formula",
                pid=666,
                mass=0.0,
                spin=1,
                charge=0,
                parity=Parity(-1),
                c_parity=Parity(-1),
                g_parity=Parity(-1),
                isospin=Spin(0.0, 0.0),
                charmness=1,
            )

    @staticmethod
    def test_eq():
        particle = Particle(
            name="MyParticle",
            pid=123,
            mass=1.2,
            spin=1,
            charge=0,
            isospin=Spin(1, 0),
        )
        assert particle != Particle(
            "MyParticle", pid=123, mass=1.5, width=0.2, spin=1
        )
        same_particle = deepcopy(particle)
        assert particle is not same_particle
        assert particle == same_particle
        assert hash(particle) == hash(same_particle)
        different_labels = Particle(
            name="Different name, same QNs",
            pid=753,
            mass=1.2,
            spin=1,
            charge=0,
            isospin=Spin(1, 0),
        )
        assert particle == different_labels
        assert hash(particle) == hash(different_labels)
        assert particle.name != different_labels.name
        assert particle.pid != different_labels.pid


class TestParticleCollection:
    @staticmethod
    def test_init(particle_database):
        new_pdg = ParticleCollection(list(particle_database.values()))
        assert new_pdg is not particle_database
        assert new_pdg == particle_database
        with pytest.raises(ValueError):
            ParticleCollection(1)  # type: ignore

    @staticmethod
    def test_find(particle_database: ParticleCollection):
        f2_1950 = particle_database.find(9050225)
        assert f2_1950.name == "f(2)(1950)"
        assert f2_1950.mass == 1.936
        phi = particle_database.find("phi(1020)")
        assert phi.pid == 333
        assert pytest.approx(phi.width) == 0.004249

    @pytest.mark.parametrize("search_term", [666, "non-existing"])
    def test_find_fail(
        self, particle_database: ParticleCollection, search_term
    ):
        with pytest.raises(LookupError):
            particle_database.find(search_term)

    @staticmethod
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

    @staticmethod
    def test_repr(particle_database):
        from_repr = eval(repr(particle_database))  # pylint: disable=eval-used
        assert from_repr == particle_database

    @staticmethod
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

    @pytest.mark.parametrize("name", ["gamma", "pi0", "K+"])
    def test_contains(self, name, particle_database):
        assert name in particle_database

    @staticmethod
    def test_key_error(particle_database: ParticleCollection):
        try:
            search_term = "omega"
            assert particle_database[search_term]
        except LookupError as error:
            candidates = {
                name for name in particle_database if search_term in name
            }
            assert error.args[-1] == candidates

    @staticmethod
    def test_keys(particle_database):
        assert set(particle_database.keys()) == set(particle_database)


class TestSpin:
    @staticmethod
    def test_init_and_eq():
        isospin = Spin(1.5, -0.5)
        assert isospin == 1.5
        assert float(isospin) == 1.5
        assert isospin.magnitude == 1.5
        assert isospin.projection == -0.5

    @staticmethod
    def test_hash():
        spin1 = Spin(0.0, 0.0)
        spin2 = Spin(1.5, -0.5)
        assert {spin2, spin1, deepcopy(spin1), deepcopy(spin2)} == {
            spin1,
            spin2,
        }

    @staticmethod
    def test_neg():
        isospin = Spin(1.5, -0.5)
        flipped_spin = -isospin
        assert flipped_spin.magnitude == isospin.magnitude
        assert flipped_spin.projection == -isospin.projection

    @pytest.mark.parametrize("spin", [Spin(2.5, -0.5), Spin(1, 0)])
    def test_repr(self, spin):
        from_repr = eval(repr(spin))  # pylint: disable=eval-used
        assert from_repr == spin

    @pytest.mark.parametrize(
        "magnitude, projection",
        [(0.3, 0.3), (1.0, 0.5), (0.5, 0.0), (-0.5, 0.5)],
    )
    def test_exceptions(self, magnitude, projection):
        with pytest.raises(ValueError):
            print(Spin(magnitude, projection))


@pytest.mark.parametrize(
    "particle_name, anti_particle_name",
    [("D+", "D-"), ("mu+", "mu-"), ("W+", "W-")],
)
def test_create_antiparticle(
    particle_database: ParticleCollection,
    particle_name,
    anti_particle_name,
):
    template_particle = particle_database[particle_name]
    anti_particle = create_antiparticle(
        template_particle, new_name=anti_particle_name
    )
    comparison_particle = particle_database[anti_particle_name]

    assert anti_particle == comparison_particle


def test_create_antiparticle_tilde(particle_database: ParticleCollection):
    anti_particles = particle_database.filter(lambda p: "~" in p.name)
    assert len(anti_particles) == 165
    for anti_particle in anti_particles.values():
        particle_name = anti_particle.name.replace("~", "")
        if "+" in particle_name:
            particle_name = particle_name.replace("+", "-")
        elif "-" in particle_name:
            particle_name = particle_name.replace("-", "+")
        created_particle = create_antiparticle(anti_particle, particle_name)

        assert created_particle == particle_database[particle_name]


@pytest.mark.parametrize(
    "particle_name",
    ["p", "phi(1020)", "W-", "gamma"],
)
def test_create_particle(particle_database, particle_name):
    template_particle = particle_database[particle_name]
    new_particle = create_particle(
        template_particle,
        name="testparticle",
        pid=89,
        mass=1.5,
        width=0.5,
    )
    assert new_particle.name == "testparticle"
    assert new_particle.pid == 89
    assert new_particle.charge == template_particle.charge
    assert new_particle.spin == template_particle.spin
    assert new_particle.mass == 1.5
    assert new_particle.width == 0.5
    assert new_particle.baryon_number == template_particle.baryon_number
    assert new_particle.strangeness == template_particle.strangeness
