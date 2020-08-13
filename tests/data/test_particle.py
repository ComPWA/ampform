import pytest

from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    QuantumState,
    Spin,
    compute_gellmann_nishijima,
)

J_PSI = Particle(
    name="J/psi",
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
    "state, expected",
    [
        (QuantumState(spin=0.0, charge=1, isospin=None), None),
        (
            QuantumState(
                spin=0.0, charge=1, isospin=Spin(1.0, 0.0), strangeness=2,
            ),
            1,
        ),
        (
            QuantumState(
                spin=1.0, charge=1, isospin=Spin(1.5, 0.5), charmness=1,
            ),
            1,
        ),
        (
            QuantumState(
                spin=0.5, charge=1, isospin=Spin(1.0, 1.0), baryon_number=1,
            ),
            1.5,
        ),
    ],
)
def test_gellmann_nishijima(state, expected):
    assert compute_gellmann_nishijima(state) == expected


def test_gellmann_nishijima_exception():
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
