"""Dump recipe objects to `dict` instances for an XML file.

At the time of writing (`a104dd5
<https://github.com/ComPWA/expertsystem/tree/a104dd5148b599f41dfdddf4935e2c5adc02baa6>`_),
the `expertsystem` assumes nested `dict` structures of this 'XML format'. This
module therefore serves as a bridge between `.ParticleCollection` and the
`.state.particle` module.

See also `expertsystem.nested_dict`.
"""

from typing import Any, Dict, List, Optional, Union

from expertsystem.amplitude.model import (
    AmplitudeModel,
    CanonicalDecay,
    CoefficientAmplitude,
    CoherentIntensity,
    Dynamics,
    FitParameter,
    HelicityDecay,
    IncoherentIntensity,
    Kinematics,
    KinematicsType,
    Node,
    NonDynamic,
    NormalizedIntensity,
    RelativisticBreitWigner,
    SequentialAmplitude,
    StrengthIntensity,
)
from expertsystem.particle import Parity, Particle, ParticleCollection, Spin


def from_amplitude_model(model: AmplitudeModel) -> dict:
    particle_list: List[dict] = list()
    for particle in model.particles:
        particle_dict = from_particle(particle)
        dynamics = model.dynamics.get(particle.name, None)
        if dynamics is not None:
            dynamics_dict = __dynamics_to_dict(dynamics)
            new_decay_info = {
                "DecayInfo": {
                    **dynamics_dict,
                    **particle_dict.get("DecayInfo", {}),
                }
            }
            particle_dict.update(new_decay_info)
        particle_list.append(particle_dict)
    return {
        "ParticleList": {"Particle": particle_list},
        **__kinematics_to_dict(model.kinematics),
        "Intensity": __intensity_to_dict(model.intensity),
    }


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {p.name: from_particle(p) for p in particles}


def from_particle(instance: Particle) -> dict:
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
    output_dict["QuantumNumber"] = __to_quantum_number_list(instance)
    if instance.width != 0.0:
        output_dict["DecayInfo"] = create_parameter_dict(
            instance.width, "Width", instance
        )

    return output_dict


def __to_quantum_number_list(state: Particle) -> List[Dict[str, Any]]:
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
        definition = __qn_to_dict(instance, type_name)
        output.append(definition)
    return output


def __qn_to_dict(
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


def __parameter_to_dict(parameter: FitParameter) -> dict:
    name_prefix = parameter.name.split("_")[0]
    name_prefix = name_prefix.lower()
    parameter_type = ""
    if name_prefix == "magnitude":
        parameter_type = "Magnitude"
    elif name_prefix == "phase":
        parameter_type = "Phase"
    elif name_prefix == "strength":
        parameter_type = "Strength"
    else:
        NotImplementedError(f"Cannot determine Type of {parameter}")
    return {
        "Class": "Double",
        "Type": parameter_type,
        "Name": parameter.name,
        "Value": parameter.value,
        "Fix": parameter.is_fixed,
    }


def __kinematics_to_dict(kin: Kinematics) -> dict:
    if kin.kinematics_type == KinematicsType.Helicity:
        kinematics_type = "HelicityKinematics"
    else:
        raise NotImplementedError("No conversion for", kin.kinematics_type)
    return {
        kinematics_type: {
            "InitialState": {
                "Particle": [
                    {"Name": p.name, "Id": i, "PositionIndex": n}
                    for n, (i, p) in enumerate(kin.initial_state.items())
                ]
            },
            "FinalState": {
                "Particle": [
                    {"Name": p.name, "Id": i, "PositionIndex": n}
                    for n, (i, p) in enumerate(kin.final_state.items())
                ],
            },
        }
    }


def __dynamics_to_dict(dynamics: Dynamics) -> dict:
    if isinstance(dynamics, NonDynamic):
        return {"Type": "nonResonant"}
    if isinstance(dynamics, RelativisticBreitWigner):
        return {"Type": "relativisticBreitWigner"}
    raise NotImplementedError("No conversion for", dynamics)


def __intensity_to_dict(  # pylint: disable=too-many-return-statements
    node: Node,
) -> dict:
    if isinstance(node, StrengthIntensity):
        return {
            "Class": "StrengthIntensity",
            "Component": node.component,
            "Parameter": __parameter_to_dict(node.strength),
            "Intensity": __intensity_to_dict(node.intensity),
        }
    if isinstance(node, NormalizedIntensity):
        return {
            "Class": "NormalizedIntensity",
            "Intensity": __intensity_to_dict(node.intensity),
        }
    if isinstance(node, IncoherentIntensity):
        return {
            "Class": "IncoherentIntensity",
            "Intensity": [
                __intensity_to_dict(intensity)
                for intensity in node.intensities
            ],
        }
    if isinstance(node, CoherentIntensity):
        return {
            "Class": "CoherentIntensity",
            "Component": node.component,
            "Amplitude": [
                __intensity_to_dict(intensity) for intensity in node.amplitudes
            ],
        }
    if isinstance(node, CoefficientAmplitude):
        parameters = [
            __parameter_to_dict(node.magnitude),
            __parameter_to_dict(node.phase),
        ]
        output_dict = {
            "Class": "CoefficientAmplitude",
            "Component": node.component,
            "Parameter": parameters,
            "Amplitude": __intensity_to_dict(node.amplitude),
        }
        if node.prefactor is not None:
            output_dict["PreFactor"] = {"Real": node.prefactor}
        return output_dict
    if isinstance(node, SequentialAmplitude):
        return {
            "Class": "SequentialAmplitude",
            "Amplitude": [
                __intensity_to_dict(intensity) for intensity in node.amplitudes
            ],
        }
    if isinstance(node, (HelicityDecay, CanonicalDecay)):
        output_dict = {
            "Class": "HelicityDecay",
            "DecayParticle": {
                "Name": node.decaying_particle.particle.name,
                "Helicity": node.decaying_particle.helicity,
            },
            "DecayProducts": {
                "Particle": [
                    {
                        "Name": decay_product.particle.name,
                        "FinalState": " ".join(
                            [str(i) for i in decay_product.final_state_ids]
                        ),
                        "Helicity": decay_product.helicity,
                    }
                    for decay_product in node.decay_products
                ]
            },
        }
        if node.recoil_system is not None:
            recoil_system = {
                "RecoilFinalState": node.recoil_system.recoil_final_state
            }
            if node.recoil_system.parent_recoil_final_state is not None:
                recoil_system[
                    "ParentRecoilFinalState"
                ] = node.recoil_system.parent_recoil_final_state
            output_dict["RecoilSystem"] = recoil_system
        if isinstance(node, CanonicalDecay):
            output_dict["CanonicalSum"] = {
                "L": int(node.l_s.j_1),
                "S": int(node.l_s.j_2),
                "ClebschGordan": [
                    {
                        "Type": "LS",
                        "@j1": node.l_s.j_1,
                        "@m1": node.l_s.m_1,
                        "@j2": node.l_s.j_2,
                        "@m2": node.l_s.m_2,
                        "J": node.l_s.J,
                        "M": node.l_s.M,
                    },
                    {
                        "Type": "s2s3",
                        "@j1": node.s2s3.j_1,
                        "@m1": node.s2s3.m_1,
                        "@j2": node.s2s3.j_2,
                        "@m2": node.s2s3.m_2,
                        "J": node.s2s3.J,
                        "M": node.s2s3.M,
                    },
                ],
            }
        return output_dict
    raise NotImplementedError("No conversion defined for", node)
