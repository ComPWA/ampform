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
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

from . import validation


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = dict()
    for name, particle in particles.items():
        output[name] = from_particle(particle)
    return output


def from_particle(particle: Particle) -> dict:
    output_dict = {
        "Name": particle.name,
        "Pid": particle.pid,
        "Parameter": {
            "Type": "Mass",
            "Name": f"Mass_{particle.name}",
            "Value": particle.mass,
        },
    }
    output_dict["QuantumNumber"] = _to_quantum_number_list(particle)
    if particle.width != 0.0:
        decay_info = {
            "Parameter": [
                {
                    "Type": "Width",
                    "Name": f"Width_{particle.name}",
                    "Value": particle.width,
                },
            ]
        }
        output_dict["DecayInfo"] = decay_info
    validation.particle(output_dict)
    return output_dict


def _to_quantum_number_list(particle: Particle) -> List[Dict[str, Any]]:
    conversion_map: Dict[
        str, Union[Optional[Parity], Optional[Spin], float, int]
    ] = {
        "Spin": particle.state.spin,
        "Charge": particle.state.charge,
        "Parity": particle.state.parity,
        "CParity": particle.state.c_parity,
        "GParity": particle.state.g_parity,
        "Strangeness": particle.state.strangeness,
        "Charm": particle.state.charmness,
        "Bottom": particle.state.bottomness,
        "Top": particle.state.topness,
        "BaryonNumber": particle.state.baryon_number,
        "ElectronLN": particle.state.electron_lepton_number,
        "MuonLN": particle.state.muon_lepton_number,
        "TauLN": particle.state.tau_lepton_number,
        "IsoSpin": particle.state.isospin,
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
    output: Dict[str, Any] = {
        "Class": "Int",
        "Type": type_name,
    }

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
