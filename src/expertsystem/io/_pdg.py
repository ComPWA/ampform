"""Create a `.ParticleCollection` instance from PDG info."""

import re
from math import copysign
from typing import Optional, Tuple, Union

from particle import Particle as PdgDatabase
from particle.particle import enums

from expertsystem.particle import (
    GellmannNishijima,
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

__skip_particles = {
    "K(L)0",  # no isospin projection
    "K(S)0",  # no isospin projection
    "B(s2)*(5840)0",  # isospin(0.5, 0.0) ?
    "B(s2)*(5840)~0",  # isospin(0.5, 0.0) ?
}


def __sign(value: Union[float, int]) -> int:
    return int(copysign(1, value))


def load_pdg() -> ParticleCollection:
    all_pdg_particles = PdgDatabase.findall(
        lambda item: item.charge is not None
        and item.charge.is_integer()  # remove quarks
        and item.J is not None  # remove new physics and nuclei
        and abs(item.pdgid) < 1e9  # p and n as nucleus
        and item.name not in __skip_particles
        and not (item.mass is None and not item.name.startswith("nu"))
    )
    particle_collection = ParticleCollection()
    for pdg_particle in all_pdg_particles:
        new_particle = __convert_pdg_instance(pdg_particle)
        particle_collection.add(new_particle)
    return particle_collection


# cspell:ignore pdgid
def __convert_pdg_instance(pdg_particle: PdgDatabase) -> Particle:
    def convert_mass_width(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return (  # https://github.com/ComPWA/expertsystem/issues/178
            float(value) / 1e3
        )

    if pdg_particle.charge is None:
        raise ValueError(f"PDG instance has no charge:\n{pdg_particle}")
    quark_numbers = __compute_quark_numbers(pdg_particle)
    lepton_numbers = __compute_lepton_numbers(pdg_particle)
    if pdg_particle.pdgid.is_lepton:  # convention: C(fermion)=+1
        parity: Optional[Parity] = Parity(__sign(pdg_particle.pdgid))  # type: ignore
    else:
        parity = __create_parity(pdg_particle.P)
    return Particle(
        name=str(pdg_particle.name),
        pid=int(pdg_particle.pdgid),
        mass=convert_mass_width(pdg_particle.mass),
        width=convert_mass_width(pdg_particle.width),
        charge=int(pdg_particle.charge),
        spin=float(pdg_particle.J),
        strangeness=quark_numbers[0],
        charmness=quark_numbers[1],
        bottomness=quark_numbers[2],
        topness=quark_numbers[3],
        baryon_number=__compute_baryonnumber(pdg_particle),
        electron_lepton_number=lepton_numbers[0],
        muon_lepton_number=lepton_numbers[1],
        tau_lepton_number=lepton_numbers[2],
        isospin=__create_isospin(pdg_particle),
        parity=parity,
        c_parity=__create_parity(pdg_particle.C),
        g_parity=__create_parity(pdg_particle.G),
    )


def __compute_quark_numbers(
    pdg_particle: PdgDatabase,
) -> Tuple[int, int, int, int]:
    strangeness = 0
    charmness = 0
    bottomness = 0
    topness = 0
    if pdg_particle.pdgid.is_hadron:
        quark_content = __filter_quark_content(pdg_particle)
        strangeness = quark_content.count("S") - quark_content.count("s")
        charmness = quark_content.count("c") - quark_content.count("C")
        bottomness = quark_content.count("B") - quark_content.count("b")
        topness = quark_content.count("t") - quark_content.count("T")
    return (
        strangeness,
        charmness,
        bottomness,
        topness,
    )


def __compute_lepton_numbers(
    pdg_particle: PdgDatabase,
) -> Tuple[int, int, int]:
    electron_lepton_number = 0
    muon_lepton_number = 0
    tau_lepton_number = 0
    if pdg_particle.pdgid.is_lepton:
        lepton_number = int(__sign(pdg_particle.pdgid))
        if "e" in pdg_particle.name:
            electron_lepton_number = lepton_number
        elif "mu" in pdg_particle.name:
            muon_lepton_number = lepton_number
        elif "tau" in pdg_particle.name:
            tau_lepton_number = lepton_number
    return electron_lepton_number, muon_lepton_number, tau_lepton_number


def __compute_baryonnumber(pdg_particle: PdgDatabase) -> int:
    return int(__sign(pdg_particle.pdgid) * pdg_particle.pdgid.is_baryon)


def __create_isospin(pdg_particle: PdgDatabase) -> Optional[Spin]:
    if pdg_particle.I is None:
        return None
    magnitude = pdg_particle.I
    projection = __compute_isospin_projection(pdg_particle)
    return Spin(magnitude, projection)


def __compute_isospin_projection(pdg_particle: PdgDatabase) -> float:
    if pdg_particle.charge is None:
        raise ValueError(f"PDG instance has no charge:\n{pdg_particle}")
    if "qq" in pdg_particle.quarks.lower():
        strangeness, charmness, bottomness, topness = __compute_quark_numbers(
            pdg_particle
        )
        baryon_number = __compute_baryonnumber(pdg_particle)
        projection = GellmannNishijima.compute_isospin_projection(
            charge=pdg_particle.charge,
            baryon_number=baryon_number,
            strangeness=strangeness,
            charmness=charmness,
            bottomness=bottomness,
            topness=topness,
        )
    else:
        projection = 0.0
        if pdg_particle.pdgid.is_hadron:
            quark_content = __filter_quark_content(pdg_particle)
            projection += quark_content.count("u") + quark_content.count("D")
            projection -= quark_content.count("U") + quark_content.count("d")
            projection *= 0.5
    if (
        pdg_particle.I is not None
        and not (pdg_particle.I - projection).is_integer()
    ):
        raise ValueError(f"Cannot have isospin {(pdg_particle.I, projection)}")
    return projection


def __filter_quark_content(pdg_particle: PdgDatabase) -> str:
    matches = re.search(r"([dDuUsScCbBtT+-]{2,})", pdg_particle.quarks)
    if matches is None:
        return ""
    return matches[1]


def __create_parity(parity_enum: enums.Parity) -> Optional[Parity]:
    if parity_enum is None or parity_enum == enums.Parity.u:
        return None
    if parity_enum == getattr(parity_enum, "o", None):  # particle < 0.14
        return None
    return Parity(parity_enum)
