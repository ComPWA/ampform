"""Read recipeobjects from a YAML file."""

from typing import (
    Optional,
    Union,
)

from expertsystem.data import (
    MeasuredValue,
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

from .validation import validate_particle_list


def _build_particle_collection(definition: dict) -> ParticleCollection:
    validate_particle_list(definition)
    definition = definition["ParticleList"]
    particles = ParticleCollection()
    for name, particle_def in definition.items():
        particles.add(_build_particle(name, particle_def))
    return particles


def _build_particle(name: str, definition: dict) -> Particle:
    qn_def = definition["QuantumNumbers"]
    return Particle(
        name=name,
        pid=int(definition["PID"]),
        mass=_build_measured_value(definition["Mass"]),
        width=_build_measured_value_optional(definition.get("Width", None)),
        charge=float(qn_def["Charge"]),
        spin=float(qn_def["Spin"]),
        strangeness=int(qn_def.get("Strangeness", 0)),
        charmness=int(qn_def.get("Charmness", 0)),
        bottomness=int(qn_def.get("Bottomness", 0)),
        topness=int(qn_def.get("Topness", 0)),
        baryon_number=int(qn_def.get("BaryonNumber", 0)),
        electron_number=int(qn_def.get("ElectronLN", 0)),
        muon_number=int(qn_def.get("MuonLN", 0)),
        tau_number=int(qn_def.get("TauLN", 0)),
        isospin=_build_spin(qn_def.get("IsoSpin", None)),
        parity=_build_parity(qn_def.get("Parity", None)),
        c_parity=_build_parity(qn_def.get("CParity", None)),
        g_parity=_build_parity(qn_def.get("GParity", None)),
    )


def _build_measured_value(
    definition: Union[dict, float, int, str]
) -> MeasuredValue:
    if isinstance(definition, (float, int, str)):
        return MeasuredValue(float(definition))
    if "Error" not in definition:
        return MeasuredValue(float(definition["Value"]))
    return MeasuredValue(
        float(definition["Value"]), float(definition["Error"])
    )


def _build_measured_value_optional(
    definition: Optional[Union[dict, float, int, str]]
) -> Optional[MeasuredValue]:
    if definition is None:
        return None
    return _build_measured_value(definition)


def _build_parity(
    definition: Optional[Union[float, int, str]]
) -> Optional[Parity]:
    if definition is None:
        return None
    return Parity(definition)


def _build_spin(
    definition: Optional[Union[dict, float, int, str]]
) -> Optional[Spin]:
    if definition is None:
        return None

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
    if magnitude == 0:
        return None
    return Spin(magnitude, projection)
