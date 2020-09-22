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


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = dict()
    for name, particle in particles.items():
        output[name] = from_particle_state(particle)
    return output


def from_particle_state(instance: Particle) -> dict:
    def create_parameter_dict(
        value: float, type_name: str, state: Particle
    ) -> dict:
        value_dict = {
            "Type": type_name,
            "Value": value,
        }
        if isinstance(state, Particle):
            value_dict["Name"] = f"Mass_{state.name}"
        return {"Parameter": value_dict}

    output_dict: Dict[str, Any] = dict()
    if isinstance(instance, Particle):
        output_dict["Name"] = instance.name
        output_dict["Pid"] = instance.pid
    output_dict.update(create_parameter_dict(instance.mass, "Mass", instance))
    output_dict["QuantumNumber"] = _to_quantum_number_list(instance)
    if instance.width != 0.0:
        output_dict["DecayInfo"] = create_parameter_dict(
            instance.width, "Width", instance
        )

    return output_dict


def _to_quantum_number_list(state: Particle) -> List[Dict[str, Any]]:
    conversion_map: Dict[
        str, Union[Optional[Parity], Optional[Spin], float, int]
    ] = {
        "Spin": state.spin,
        "Charge": state.charge,
        "Parity": state.parity,
        "CParity": state.c_parity,
        "GParity": state.g_parity,
        "Strangeness": state.strangeness,
        "Charmness": state.charmness,
        "Bottomness": state.bottomness,
        "Topness": state.topness,
        "BaryonNumber": state.baryon_number,
        "ElectronLN": state.electron_lepton_number,
        "MuonLN": state.muon_lepton_number,
        "TauLN": state.tau_lepton_number,
        "IsoSpin": state.isospin,
    }
    output: List[Dict[str, Any]] = list()
    for type_name, instance in conversion_map.items():
        if instance is None:
            continue
        if type_name not in ["Charge", "Spin", "IsoSpin"] and instance == 0:
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
