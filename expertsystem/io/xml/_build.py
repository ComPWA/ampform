"""Read recipe objects from an XML file."""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    ValuesView,
)

from expertsystem.data import (
    MeasuredValue,
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

from . import validation


def build_particle_collection(definition: dict) -> ParticleCollection:
    if isinstance(definition, dict):
        definition = definition.get("root", definition)
    if isinstance(definition, dict):
        definition = definition.get("ParticleList", definition)
    if isinstance(definition, dict):
        definition = definition.get("Particle", definition)
    if isinstance(definition, list):
        particle_list: Union[List[dict], ValuesView] = definition
    elif isinstance(definition, dict):
        particle_list = definition.values()
    else:
        raise ValueError(
            "The following definition cannot be converted to a ParticleCollection\n"
            f"{definition}"
        )
    collection = ParticleCollection()
    for particle_def in particle_list:
        collection.add(build_particle(particle_def))
    return collection


def build_particle(definition: dict) -> Particle:
    validation.particle(definition)
    qn_defs = _xml_qn_list_to_qn_object(definition["QuantumNumber"])
    return Particle(
        name=str(definition["Name"]),
        pid=int(definition["Pid"]),
        mass=_xml_to_measured_value(definition["Parameter"]),
        width=_xml_to_width(definition),
        charge=float(qn_defs["Charge"]),
        spin=float(qn_defs["Spin"]),
        strangeness=int(qn_defs.get("Strangeness", 0)),
        charmness=int(qn_defs.get("Charm", 0)),
        bottomness=int(qn_defs.get("Bottomness", 0)),
        topness=int(qn_defs.get("Topness", 0)),
        baryon_number=int(qn_defs.get("BaryonNumber", 0)),
        electron_number=int(qn_defs.get("ElectronLN", 0)),
        muon_number=int(qn_defs.get("MuonLN", 0)),
        tau_number=int(qn_defs.get("TauLN", 0)),
        isospin=qn_defs.get("IsoSpin", None),
        parity=qn_defs.get("Parity", None),
        c_parity=qn_defs.get("CParity", None),
        g_parity=qn_defs.get("GParity", None),
    )


def _xml_to_measured_value(definition: dict) -> MeasuredValue:
    if "Error" not in definition:
        return MeasuredValue(float(definition["Value"]))
    return MeasuredValue(
        float(definition["Value"]), float(definition["Error"])
    )


def _xml_to_width(definition: dict) -> Optional[MeasuredValue]:
    definition = definition.get("DecayInfo", {})
    definition = definition.get("Parameter", None)
    if isinstance(definition, list):
        for item in definition:  # type: ignore
            if item["Type"] == "Width":
                definition = item
                break
    if definition is None or not isinstance(definition, dict):
        return None
    return _xml_to_measured_value(definition)


def _xml_qn_list_to_qn_object(definitions: List[dict],) -> Dict[str, Any]:
    output = dict()
    for definition in definitions:
        type_name, quantum_number = _xml_to_quantum_number(definition)
        output[type_name] = quantum_number
    return output


def _xml_to_quantum_number(definition: Dict[str, str]) -> Tuple[str, Any]:
    conversion_map: Dict[str, Callable] = {
        "Spin": _xml_to_float,
        "Charge": _xml_to_float,
        "Strangeness": _xml_to_int,
        "Charm": _xml_to_int,
        "BaryonNumber": _xml_to_int,
        "ElectronLN": _xml_to_int,
        "MuonLN": _xml_to_int,
        "TauLN": _xml_to_int,
        "Parity": _xml_to_parity,
        "CParity": _xml_to_parity,
        "GParity": _xml_to_parity,
        "IsoSpin": _xml_to_spin,
    }
    type_name = definition["Type"]
    for key, converter in conversion_map.items():
        if type_name == key:
            return key, converter(definition)
    raise NotImplementedError(
        f"No conversion defined for type {type_name}\n"
        "Trying to convert definition:\n"
        f"{definition}"
    )


def _xml_to_float(definition: dict) -> float:
    return float(definition["Value"])


def _xml_to_int(definition: dict) -> int:
    return int(definition["Value"])


def _xml_to_parity(definition: dict) -> Parity:
    return Parity(_xml_to_int(definition))


def _xml_to_spin(definition: dict) -> Optional[Spin]:
    magnitude = float(definition["Value"])
    if "Projection" not in definition and magnitude != 0.0:
        raise ValueError(
            "Can only have a spin without projection if magnitude = 0"
        )
    if magnitude == 0.0:
        return None
    projection = float(definition["Projection"])
    return Spin(magnitude, projection)
