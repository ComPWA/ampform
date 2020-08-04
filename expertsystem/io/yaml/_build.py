"""Read recipe objects from a YAML file."""

from typing import (
    Optional,
    Union,
)

from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    ParticleQuantumState,
    Spin,
)

from . import validation


def build_particle_collection(definition: dict) -> ParticleCollection:
    validation.particle_list(definition)
    definition = definition["ParticleList"]
    particles = ParticleCollection()
    for name, particle_def in definition.items():
        particles.add(build_particle(name, particle_def))
    return particles


def build_particle(name: str, definition: dict) -> Particle:
    qn_def = definition["QuantumNumbers"]
    return Particle(
        name=name,
        pid=int(definition["PID"]),
        mass=float(definition["Mass"]),
        width=float(definition.get("Width", 0.0)),
        state=ParticleQuantumState(
            charge=int(qn_def["Charge"]),
            spin=float(qn_def["Spin"]),
            strangeness=int(qn_def.get("Strangeness", 0)),
            charmness=int(qn_def.get("Charmness", 0)),
            bottomness=int(qn_def.get("Bottomness", 0)),
            topness=int(qn_def.get("Topness", 0)),
            baryon_number=int(qn_def.get("BaryonNumber", 0)),
            electron_lepton_number=int(qn_def.get("ElectronLN", 0)),
            muon_lepton_number=int(qn_def.get("MuonLN", 0)),
            tau_lepton_number=int(qn_def.get("TauLN", 0)),
            isospin=_yaml_to_isospin(qn_def.get("IsoSpin", None)),
            parity=_yaml_to_parity(qn_def.get("Parity", None)),
            c_parity=_yaml_to_parity(qn_def.get("CParity", None)),
            g_parity=_yaml_to_parity(qn_def.get("GParity", None)),
        ),
    )


def build_spin(definition: Union[dict, float, int, str]) -> Spin:
    def check_missing_projection(magnitude: float) -> None:
        if magnitude != 0.0:
            raise ValueError(
                "Can only have a spin without projection if magnitude = 0"
            )

    if isinstance(definition, (float, int)):
        magnitude = float(definition)
        check_missing_projection(magnitude)
        projection = 0.0
    elif not isinstance(definition, dict):
        raise ValueError(f"Cannot create Spin from definition {definition}")
    else:
        magnitude = float(definition["Value"])
        if "Projection" not in definition:
            check_missing_projection(magnitude)
        projection = definition.get("Projection", 0.0)
    return Spin(magnitude, projection)


def _yaml_to_parity(
    definition: Optional[Union[float, int, str]]
) -> Optional[Parity]:
    if definition is None:
        return None
    return Parity(definition)


def _yaml_to_isospin(
    definition: Optional[Union[dict, float, int, str]]
) -> Optional[Spin]:
    if definition is None:
        return None
    spin = build_spin(definition)
    if spin.magnitude == 0:
        return None
    return spin
