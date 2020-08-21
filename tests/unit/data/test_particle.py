# pylint: disable=redefined-outer-name
import pytest

from expertsystem.data import (
    GellmannNishijima,
    Parity,
    Particle,
    ParticleCollection,
    QuantumState,
    Spin,
    create_antiparticle,
    create_particle,
)


J_PSI = Particle(
    name="J/psi(1S)",
    pid=443,
    mass=3.0969,
    width=9.29e-05,
    state=QuantumState[float](
        spin=1,
        charge=0,
        parity=Parity(-1),
        c_parity=Parity(-1),
        g_parity=Parity(-1),
        isospin=Spin(0.0, 0.0),
    ),
)


@pytest.mark.parametrize(
    "instance", [ParticleCollection(), Spin(2.5, -0.5), Parity(1), J_PSI],
)
def test_repr(instance):
    copy_from_repr = eval(repr(instance))  # pylint: disable=eval-used
    assert copy_from_repr == instance


def test_parity():
    with pytest.raises(ValueError):
        Parity(1.2)
    parity = Parity(+1)
    assert parity == +1
    assert int(parity) == +1
    flipped_parity = -parity
    assert flipped_parity.value == -parity.value


def test_spin():
    with pytest.raises(ValueError):
        Spin(1, -2)
    isospin = Spin(1.5, -0.5)
    assert isospin == 1.5
    assert float(isospin) == 1.5
    assert isospin.magnitude == 1.5
    assert isospin.projection == -0.5

    flipped_spin = -isospin
    assert flipped_spin.magnitude == isospin.magnitude
    assert flipped_spin.projection == -isospin.projection


def test_particle():
    assert J_PSI.mass == 3.0969
    assert J_PSI.width == 9.29e-05
    assert J_PSI.state.bottomness == 0


@pytest.mark.parametrize(
    "particle_name", ["p", "phi(1020)", "W-", "gamma"],
)
def test_create_particle(
    particle_database, particle_name  # pylint: disable=W0621
):
    template_particle = particle_database[particle_name]
    new_particle = create_particle(
        template_particle, name="testparticle", pid=89, mass=1.5, width=0.5,
    )

    assert new_particle.name == "testparticle"
    assert new_particle.pid == 89
    assert new_particle.state.charge == template_particle.state.charge
    assert new_particle.state.spin == template_particle.state.spin
    assert new_particle.mass == 1.5
    assert new_particle.width == 0.5
    assert (
        new_particle.state.baryon_number
        == template_particle.state.baryon_number
    )
    assert (
        new_particle.state.strangeness == template_particle.state.strangeness
    )


@pytest.mark.parametrize(
    "particle_name, anti_particle_name",
    [("D+", "D-"), ("mu+", "mu-"), ("W+", "W-")],
)
def test_create_antiparticle(
    particle_database,  # pylint: disable=W0621
    particle_name,
    anti_particle_name,
):
    template_particle = particle_database[particle_name]
    anti_particle = create_antiparticle(template_particle)
    comparison_particle = particle_database[anti_particle_name]

    assert anti_particle.pid == comparison_particle.pid
    assert anti_particle.mass == comparison_particle.mass
    assert anti_particle.width == comparison_particle.width
    assert anti_particle.state == comparison_particle.state
    assert anti_particle.name == "anti-" + particle_name


def test_create_antiparticle_tilde(particle_database):
    anti_particles = particle_database.find_subset("~")
    assert len(anti_particles) == 166
    for anti_particle in anti_particles.values():
        particle_name = anti_particle.name.replace("~", "")
        if "+" in particle_name:
            particle_name = particle_name.replace("+", "-")
        elif "-" in particle_name:
            particle_name = particle_name.replace("-", "+")
        created_particle = create_antiparticle(anti_particle, particle_name)

        assert created_particle.state == particle_database[particle_name].state
        assert created_particle.complex_energy == pytest.approx(
            particle_database[particle_name].complex_energy
        )


class TestGellmannNishijima:
    @staticmethod
    @pytest.mark.parametrize(
        "state",
        [
            QuantumState(
                spin=0.0, charge=1, isospin=Spin(1.0, 0.0), strangeness=2,
            ),
            QuantumState(
                spin=1.0, charge=1, isospin=Spin(1.5, 0.5), charmness=1,
            ),
            QuantumState(
                spin=0.5,
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
        state = QuantumState(spin=0.0, charge=1, isospin=None)
        assert GellmannNishijima.compute_charge(state) is None

    @staticmethod
    def test_exceptions():
        with pytest.raises(ValueError):
            print(
                Particle(
                    name="Fails Gell-Mann–Nishijima formula",
                    pid=666,
                    mass=0.0,
                    state=QuantumState[float](
                        spin=1,
                        charge=0,
                        parity=Parity(-1),
                        c_parity=Parity(-1),
                        g_parity=Parity(-1),
                        isospin=Spin(0.0, 0.0),
                        charmness=1,
                    ),
                )
            )
