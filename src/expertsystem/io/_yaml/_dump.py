"""Dump recipe objects to `dict` instances for a YAML file."""
from typing import Callable, Dict, List, Optional, Tuple, Union

from expertsystem.amplitude.model import (
    AmplitudeModel,
    BlattWeisskopf,
    CanonicalDecay,
    CoefficientAmplitude,
    CoherentIntensity,
    Dynamics,
    FitParameter,
    FitParameters,
    FormFactor,
    HelicityDecay,
    IncoherentIntensity,
    Kinematics,
    KinematicsType,
    Node,
    NonDynamic,
    NormalizedIntensity,
    ParticleDynamics,
    RelativisticBreitWigner,
    SequentialAmplitude,
    StrengthIntensity,
)
from expertsystem.particle import Parity, Particle, ParticleCollection, Spin

from . import validation


def from_amplitude_model(model: AmplitudeModel) -> dict:
    output_dict = {
        "Kinematics": __kinematics_to_dict(model.kinematics),
        "Parameters": __parameters_to_dict(model.parameters),
        "Intensity": __intensity_to_dict(model.intensity),
        **from_particle_collection(model.particles),
        "Dynamics": __dynamics_section_to_dict(model.dynamics),
    }
    validation.amplitude_model(output_dict)
    return output_dict


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = {p.name: from_particle(p) for p in particles}
    output = {"ParticleList": output}
    validation.particle_list(output)
    return output


def from_particle(particle: Particle) -> dict:
    output_dict: Dict[str, Union[float, int, dict]] = {
        "PID": particle.pid,
        "Mass": particle.mass,
    }
    if particle.width != 0.0:
        output_dict["Width"] = particle.width
    output_dict["QuantumNumbers"] = __to_quantum_number_dict(particle)
    return output_dict


def __to_quantum_number_dict(
    particle: Particle,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    output_dict: Dict[str, Union[float, int, Dict[str, float]]] = {
        "Spin": __attempt_to_int(particle.spin),
        "Charge": int(particle.charge),
    }
    optional_qn: List[
        Tuple[str, Union[Optional[Parity], Spin, int], Union[Callable, int]]
    ] = [
        ("Parity", particle.parity, int),
        ("CParity", particle.c_parity, int),
        ("GParity", particle.g_parity, int),
        ("Strangeness", particle.strangeness, int),
        ("Charmness", particle.charmness, int),
        ("Bottomness", particle.bottomness, int),
        ("Topness", particle.topness, int),
        ("BaryonNumber", particle.baryon_number, int),
        ("ElectronLN", particle.electron_lepton_number, int),
        ("MuonLN", particle.muon_lepton_number, int),
        ("TauLN", particle.tau_lepton_number, int),
    ]
    for key, value, converter in optional_qn:
        if value in [0, None]:
            continue
        output_dict[key] = converter(  # type: ignore
            value
        )  # pylint: disable=not-callable
    if particle.isospin is not None:
        output_dict["IsoSpin"] = __from_spin(particle.isospin)
    return output_dict


def __from_spin(instance: Spin) -> Union[Dict[str, Union[float, int]], int]:
    if instance.magnitude == 0:
        return 0
    return {
        "Value": __attempt_to_int(instance.magnitude),
        "Projection": __attempt_to_int(instance.projection),
    }


def __attempt_to_int(value: Union[Spin, float, int]) -> Union[float, int]:
    if isinstance(value, Spin):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def __parameters_to_dict(parameters: FitParameters) -> List[dict]:
    return [__parameter_to_dict(par) for par in parameters.values()]


def __parameter_to_dict(parameter: FitParameter) -> dict:
    parameter_type = ""
    if "_" in parameter.name:
        name_prefix = parameter.name.split("_")[0]
        name_prefix = name_prefix.lower()
        if name_prefix == "magnitude":
            parameter_type = "Magnitude"
        elif name_prefix == "phase":
            parameter_type = "Phase"
        elif name_prefix == "strength":
            parameter_type = "Strength"
        elif name_prefix == "mesonradius":
            parameter_type = "MesonRadius"
    output_dict = {
        "Name": parameter.name,
        "Value": parameter.value,
    }
    if parameter.is_fixed:
        output_dict["Fix"] = True
    if parameter_type:
        output_dict["Type"] = parameter_type
    return output_dict


def __kinematics_to_dict(kin: Kinematics) -> dict:
    if kin.kinematics_type == KinematicsType.Helicity:
        kinematics_type = "Helicity"
    else:
        raise NotImplementedError("No conversion for", kin.kinematics_type)
    return {
        "Type": kinematics_type,
        "InitialState": [
            {"Particle": p.name, "ID": i} for i, p in kin.initial_state.items()
        ],
        "FinalState": [
            {"Particle": p.name, "ID": i} for i, p in kin.final_state.items()
        ],
    }


def __dynamics_section_to_dict(particle_dynamics: ParticleDynamics) -> dict:
    output_dict = dict()
    for particle_name, dynamics in particle_dynamics.items():
        output_dict[particle_name] = __dynamics_to_dict(dynamics)
    return output_dict


def __dynamics_to_dict(dynamics: Dynamics) -> dict:
    output: dict = {"Type": dynamics.__class__.__name__}
    if isinstance(dynamics, NonDynamic):
        output.update(__form_factor_to_dict(dynamics.form_factor))
        return output
    if isinstance(dynamics, RelativisticBreitWigner):
        output["PoleParameters"] = {
            "Real": dynamics.pole_position.name,
            "Imaginary": dynamics.pole_width.name,
        }
        output.update(__form_factor_to_dict(dynamics.form_factor))
        return output
    raise NotImplementedError("No conversion for", dynamics)


def __form_factor_to_dict(form_factor: Optional[FormFactor]) -> dict:
    if form_factor is None:
        return dict()
    if isinstance(form_factor, BlattWeisskopf):
        return {
            "FormFactor": {
                "Type": "BlattWeisskopf",
                "MesonRadius": form_factor.meson_radius.name,
            }
        }
    raise NotImplementedError("No conversion for", form_factor)


def __intensity_to_dict(  # pylint: disable=too-many-return-statements
    node: Node,
) -> dict:
    if isinstance(node, StrengthIntensity):
        return {
            "Class": "StrengthIntensity",
            "Component": node.component,
            "Strength": node.strength.name,
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
            "Intensities": [
                __intensity_to_dict(intensity)
                for intensity in node.intensities
            ],
        }
    if isinstance(node, CoherentIntensity):
        return {
            "Class": "CoherentIntensity",
            "Component": node.component,
            "Amplitudes": [
                __intensity_to_dict(intensity) for intensity in node.amplitudes
            ],
        }
    if isinstance(node, CoefficientAmplitude):
        output_dict: dict = {
            "Class": "CoefficientAmplitude",
            "Component": node.component,
        }
        if node.prefactor is not None:
            output_dict["PreFactor"] = node.prefactor
        output_dict["Magnitude"] = node.magnitude.name
        output_dict["Phase"] = node.phase.name
        output_dict["Amplitude"] = __intensity_to_dict(node.amplitude)
        return output_dict
    if isinstance(node, SequentialAmplitude):
        return {
            "Class": "SequentialAmplitude",
            "Amplitudes": [
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
            "DecayProducts": [
                {
                    "Name": decay_product.particle.name,
                    "FinalState": decay_product.final_state_ids,
                    "Helicity": decay_product.helicity,
                }
                for decay_product in node.decay_products
            ],
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
            output_dict["Canonical"] = {
                "LS": {
                    "ClebschGordan": {
                        "J": node.l_s.J,
                        "M": node.l_s.M,
                        "j1": node.l_s.j_1,
                        "m1": node.l_s.m_1,
                        "j2": node.l_s.j_2,
                        "m2": node.l_s.m_2,
                    },
                },
                "s2s3": {
                    "ClebschGordan": {
                        "J": node.s2s3.J,
                        "M": node.s2s3.M,
                        "j1": node.s2s3.j_1,
                        "m1": node.s2s3.m_1,
                        "j2": node.s2s3.j_2,
                        "m2": node.s2s3.m_2,
                    }
                },
            }
        return output_dict
    raise NotImplementedError("No conversion defined for", node)
