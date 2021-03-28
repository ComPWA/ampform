# pylint: disable=eval-used,redefined-outer-name, no-self-use
import logging
import typing
from copy import deepcopy

import pytest
from attr.exceptions import FrozenInstanceError
from IPython.lib.pretty import pretty

from expertsystem.particle import (
    GellmannNishijima,
    Parity,
    Particle,
    ParticleCollection,
    Spin,
    _to_fraction,
    create_antiparticle,
    create_particle,
)


class TestGellmannNishijima:
    @staticmethod
    @pytest.mark.parametrize(
        "state",
        [
            Particle(
                name="p1",
                pid=1,
                spin=0.0,
                mass=1,
                charge=1,
                isospin=Spin(1.0, 0.0),
                strangeness=2,
            ),
            Particle(
                name="p1",
                pid=1,
                spin=1.0,
                mass=1,
                charge=1,
                isospin=Spin(1.5, 0.5),
                charmness=1,
            ),
            Particle(
                name="p1",
                pid=1,
                spin=0.5,
                mass=1,
                charge=1.5,  # type: ignore
                isospin=Spin(1.0, 1.0),
                baryon_number=1,
            ),
        ],
    )
    def test_computations(state: Particle):
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
            == state.isospin.projection  # type: ignore
        )

    @staticmethod
    def test_isospin_none():
        state = Particle(
            name="p1", pid=1, mass=1, spin=0.0, charge=1, isospin=None
        )
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
        from_repr = eval(repr(parity))
        assert from_repr == parity

    @staticmethod
    def test_exceptions():
        with pytest.raises(TypeError):
            Parity(1.2)  # type: ignore
        with pytest.raises(ValueError):
            Parity(0)


class TestParticle:
    @staticmethod
    def test_repr(particle_database: ParticleCollection):
        for instance in particle_database:
            from_repr = eval(repr(instance))
            assert from_repr == instance
            from_repr = eval(pretty(instance))
            assert from_repr == instance

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
                name="MyParticle",
                pid=123,
                mass=1.2,
                width=0.1,
                spin=1,
                charge=0,
                isospin=(1, 0),
            )
            test_state.charge = 1  # type: ignore
        with pytest.raises(ValueError):
            Particle(
                name="Fails Gell-Mann–Nishijima formula",
                pid=666,
                mass=0.0,
                spin=1,
                charge=0,
                parity=-1,
                c_parity=-1,
                g_parity=-1,
                isospin=(0, 0),
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
            isospin=(1, 0),
        )
        assert particle != Particle(
            name="MyParticle", pid=123, mass=1.5, width=0.2, spin=1
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
            isospin=(1, 0),
        )
        assert particle == different_labels
        assert hash(particle) == hash(different_labels)
        assert particle.name != different_labels.name
        assert particle.pid != different_labels.pid

    def test_neg(self, particle_database: ParticleCollection):
        pip = particle_database.find(211)
        pim = particle_database.find(-211)
        assert pip == -pim


class TestParticleCollection:
    @staticmethod
    def test_init(particle_database: ParticleCollection):
        new_pdg = ParticleCollection(particle_database)
        assert new_pdg is not particle_database
        assert new_pdg == particle_database
        with pytest.raises(TypeError):
            ParticleCollection(1)  # type: ignore

    @staticmethod
    def test_equality(particle_database: ParticleCollection):
        assert list(particle_database) == particle_database
        with pytest.raises(NotImplementedError):
            assert particle_database == 0

    @staticmethod
    def test_repr(particle_database: ParticleCollection):
        instance = particle_database
        from_repr = eval(repr(instance))
        assert from_repr == instance
        from_repr = eval(pretty(instance))
        assert from_repr == instance

    def test_add(self, particle_database: ParticleCollection):
        subset_copy = particle_database.filter(
            lambda p: p.name.startswith("omega")
        )
        subset_copy += particle_database.filter(
            lambda p: p.name.startswith("pi")
        )
        n_subset = len(subset_copy)

        new_particle = create_particle(
            particle_database.find(443),
            pid=666,
            name="EpEm",
            mass=1.0,
            width=0.0,
        )
        subset_copy.add(new_particle)
        assert len(subset_copy) == n_subset + 1
        assert subset_copy["EpEm"] is new_particle

    def test_add_warnings(self, particle_database: ParticleCollection, caplog):
        pions = particle_database.filter(lambda p: p.name.startswith("pi"))
        pi_plus = pions["pi+"]
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            pions.add(create_particle(pi_plus, name="new pi+", mass=0.0))
        assert f"{pi_plus.pid}" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            pions.add(create_particle(pi_plus, width=1.0))
        assert "pi+" in caplog.text

    @pytest.mark.parametrize("name", ["gamma", "pi0", "K+"])
    def test_contains(self, name: str, particle_database: ParticleCollection):
        assert name in particle_database
        particle = particle_database[name]
        assert particle in particle_database
        assert particle.pid in particle_database

    def test_discard(self, particle_database: ParticleCollection):
        pions = particle_database.filter(lambda p: p.name.startswith("pi"))
        n_pions = len(pions)
        pim = pions["pi-"]
        pip = pions["pi+"]

        pions.discard(pions["pi+"])
        assert len(pions) == n_pions - 1
        assert "pi+" not in pions
        assert pip.name == "pi+"  # still exists

        pions.remove("pi-")
        assert len(pions) == n_pions - 2
        assert pim not in pions
        assert pim.name == "pi-"  # still exists

        with pytest.raises(NotImplementedError):
            pions.discard(111)  # type: ignore

    @staticmethod
    def test_filter(particle_database: ParticleCollection):
        search_result = particle_database.filter(lambda p: "f(0)" in p.name)
        f0_1500_from_subset = search_result["f(0)(1500)"]
        assert len(search_result) == 5
        assert f0_1500_from_subset.mass == 1.506
        assert f0_1500_from_subset is particle_database["f(0)(1500)"]
        assert f0_1500_from_subset is not particle_database["f(0)(980)"]

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
        assert filtered_result.names == {
            "K(2)(1820)0",
            "K(2)(1820)+",
        }

    @staticmethod
    def test_find(particle_database: ParticleCollection):
        f2_1950 = particle_database.find(9050225)
        assert f2_1950.name == "f(2)(1950)"
        assert f2_1950.mass == 1.936
        phi = particle_database.find("phi(1020)")
        assert phi.pid == 333
        assert pytest.approx(phi.width) == 0.004249

    @pytest.mark.parametrize(
        "search_term, expected",
        [
            (666, None),
            ("non-existing", None),
            # cspell:disable
            ("gamm", "'gamma'"),
            ("gama", "'gamma', 'Sigma0', 'Sigma-', 'Sigma+', 'Lambda'"),
            (
                "omega",
                "'omega(782)', 'omega(1420)', 'omega(3)(1670)', 'omega(1650)'",
            ),
            ("p~~", "'p~'"),
            ("~", "'p~', 'n~'"),
            ("lambda", "'Lambda', 'Lambda~', 'Lambda(c)+', 'Lambda(b)0'"),
            # cspell:enable
        ],
    )
    def test_find_fail(
        self, particle_database: ParticleCollection, search_term, expected
    ):
        with pytest.raises(LookupError) as exception:
            particle_database.find(search_term)
        if expected is not None:
            message = str(exception.value.args[0])
            message = message.strip("?")
            message = message.strip("]")
            assert message.endswith(expected)

    @staticmethod
    def test_exceptions(particle_database: ParticleCollection):
        gamma = particle_database["gamma"]
        with pytest.raises(ValueError):
            particle_database += create_particle(gamma, name="gamma_new")
        with pytest.raises(NotImplementedError):
            particle_database.find(3.14)  # type: ignore
        with pytest.raises(NotImplementedError):
            particle_database += 3.14  # type: ignore
        with pytest.raises(NotImplementedError):
            assert 3.14 in particle_database
        with pytest.raises(AssertionError):
            assert gamma == "gamma"


class TestSpin:
    @staticmethod
    def test_init_and_eq():
        isospin = Spin(1.5, -0.5)
        assert isospin == 1.5
        assert float(isospin) == 1.5
        assert isospin.magnitude == 1.5
        assert isospin.projection == -0.5
        isospin = Spin(1, -0.0)
        assert isinstance(isospin.magnitude, float)
        assert isinstance(isospin.projection, float)
        assert isospin.magnitude == 1.0
        assert isospin.projection == 0.0

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

    @pytest.mark.parametrize(
        "instance", [Spin(2.5, -0.5), Spin(1, 0), Spin(3, -1), Spin(0, 0)]
    )
    def test_repr(self, instance):
        from_repr = eval(repr(instance))
        assert from_repr == instance
        from_repr = eval(pretty(instance))
        assert from_repr == instance

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
    assert len(anti_particles) in [
        165,  # particle==0.13
        172,  # particle==0.14
    ]
    for anti_particle in anti_particles:
        particle_name = anti_particle.name.replace("~", "")
        if "+" in particle_name:
            particle_name = particle_name.replace("+", "-")
        elif "-" in particle_name:
            particle_name = particle_name.replace("-", "+")
        created_particle = create_antiparticle(anti_particle, particle_name)

        assert created_particle == particle_database[particle_name]


def test_create_antiparticle_by_pid(particle_database: ParticleCollection):
    n_particles_with_neg_pid = 0
    for particle in particle_database:
        anti_particles_by_pid = particle_database.filter(
            lambda p: p.pid
            == -particle.pid  # pylint: disable=cell-var-from-loop
        )
        if len(anti_particles_by_pid) != 1:
            continue
        n_particles_with_neg_pid += 1
        anti_particle = next(iter(anti_particles_by_pid))
        particle_from_anti = -anti_particle
        assert particle == particle_from_anti
    assert n_particles_with_neg_pid in [
        428,  # particle==0.13
        442,  # particle==0.14
    ]


@pytest.mark.parametrize(
    "particle_name",
    ["p", "phi(1020)", "W-", "gamma"],
)
def test_create_particle(
    particle_database: ParticleCollection, particle_name: str
):
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


@pytest.mark.parametrize(
    "value, render_plus, expected",
    [
        (0, False, "0"),
        (0, True, "0"),
        (-1, False, "-1"),
        (-1, True, "-1"),
        (1, False, "1"),
        (1, True, "+1"),
        (1.0, True, "+1"),
        (0.5, True, "+1/2"),
        (-0.5, True, "-1/2"),
        (+1.5, False, "3/2"),
        (+1.5, True, "+3/2"),
        (-1.5, True, "-3/2"),
    ],
)
def test_to_fraction(value, render_plus: bool, expected: str):
    assert _to_fraction(value, render_plus) == expected
