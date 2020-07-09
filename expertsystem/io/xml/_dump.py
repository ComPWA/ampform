"""Dump recipe objects to `dict` instances for an XML file.

At the time of writing (`a104dd5
<https://github.com/ComPWA/expertsystem/tree/a104dd5148b599f41dfdddf4935e2c5adc02baa6>`_),
the `expertsystem` assumes nested `dict` structures of this 'XML format'. This
module therefore serves as a bridge between `.ParticleCollection` and the
`.state.particle` module.
"""

from typing import (
    Any,
    Dict,
    List,
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

from .validation import validate_particle


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = dict()
    for name, particle in particles.items():
        output[name] = from_particle(particle)
    return output


def from_particle(particle: Particle) -> dict:
    output_dict = {
        "Name": particle.name,
        "Pid": particle.pid,
        "Parameter": _from_measured_value(
            particle.mass, name=f"Mass_{particle.name}"
        ),
    }
    output_dict["QuantumNumber"] = _to_quantum_number_list(particle)
    if particle.width is not None:
        decay_info = {
            "Parameter": [
                _from_measured_value(
                    particle.width, name=f"Width_{particle.name}"
                )
            ]
        }
        output_dict["DecayInfo"] = decay_info
    validate_particle(output_dict)
    return output_dict


def _from_measured_value(instance: MeasuredValue, name: str) -> dict:
    type_name = name.split("_")[0]
    output = {
        "Name": name,
        "Type": type_name,
        "Value": instance.value,
    }
    if instance.uncertainty is not None:
        output["Error"] = instance.uncertainty
    return output


def _to_quantum_number_list(particle: Particle) -> List[Dict[str, Any]]:
    conversion_map: Dict[
        str, Union[Optional[Parity], Optional[Spin], float, int]
    ] = {
        "Spin": particle.spin,
        "Charge": particle.charge,
        "Parity": particle.parity,
        "CParity": particle.c_parity,
        "GParity": particle.g_parity,
        "Strangeness": particle.strangeness,
        "Charm": particle.charmness,
        "Bottom": particle.bottomness,
        "Top": particle.topness,
        "BaryonNumber": particle.baryon_number,
        "ElectronLN": particle.electron_number,
        "MuonLN": particle.muon_number,
        "TauLN": particle.tau_number,
        "IsoSpin": particle.isospin,
    }
    output: List[Dict[str, Any]] = list()
    for type_name, instance in conversion_map.items():
        if instance is None:
            continue
        if type_name not in ["Charge", "Spin"] and instance == 0:
            continue
        definition = _qn_to_dict(instance, type_name)
        output.append(definition)
    return output


def _qn_to_dict(
    instance: Union[Parity, Spin, float, int], type_name: str
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"Type": type_name}
    output["Class"] = "Int"
    if type_name == "Spin":
        output["Class"] = "Spin"
    if isinstance(instance, (float, int)):
        output["Value"] = instance
    elif isinstance(instance, Parity):
        output["Value"] = int(instance)
    elif isinstance(instance, Spin):
        output["Class"] = "Spin"
        output["Value"] = instance.magnitude
        if instance.magnitude != 0:
            output["Projection"] = instance.projection
    return output
