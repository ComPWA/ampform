"""Dump recipe objects to `dict` instances for a YAML file."""
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from expertsystem.amplitude.model import (
    AmplitudeModel,
    BlattWeisskopf,
    CanonicalDecay,
    CoefficientAmplitude,
    CoherentIntensity,
    Dynamics,
    FitParameter,
    FitParameters,
    HelicityDecay,
    IncoherentIntensity,
    Kinematics,
    KinematicsType,
    Node,
    NonDynamic,
    NormalizedIntensity,
    ParticleDynamics,
    SequentialAmplitude,
    StrengthIntensity,
)
from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

from . import validation


def from_amplitude_model(model: AmplitudeModel) -> dict:
    output_dict = {
        "Kinematics": _kinematics_to_dict(model.kinematics),
        "Parameters": _parameters_to_dict(model.parameters),
        "Intensity": _intensity_to_dict(model.intensity),
        **from_particle_collection(model.particles),
        "Dynamics": _dynamics_section_to_dict(model.dynamics),
    }
    validation.amplitude_model(output_dict)
    return output_dict


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = {
        name: from_particle(particle) for name, particle in particles.items()
    }
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
    output_dict["QuantumNumbers"] = _to_quantum_number_dict(particle)
    return output_dict


def _to_quantum_number_dict(
    particle: Particle,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    output_dict: Dict[str, Union[float, int, Dict[str, float]]] = {
        "Spin": _attempt_to_int(particle.spin),
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
        output_dict["IsoSpin"] = _from_spin(particle.isospin)
    return output_dict


def _from_spin(instance: Spin) -> Union[Dict[str, Union[float, int]], int]:
    if instance.magnitude == 0:
        return 0
    return {
        "Value": _attempt_to_int(instance.magnitude),
        "Projection": _attempt_to_int(instance.projection),
    }


def _attempt_to_int(value: Union[Spin, float, int]) -> Union[float, int]:
    if isinstance(value, Spin):
        value = float(value)
    if value.is_integer():
        return int(value)
    return value


def _parameters_to_dict(parameters: FitParameters) -> List[dict]:
    return [_parameter_to_dict(par) for par in parameters.values()]


def _parameter_to_dict(parameter: FitParameter) -> dict:
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


def _kinematics_to_dict(kin: Kinematics) -> dict:
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


def _dynamics_section_to_dict(particle_dynamics: ParticleDynamics) -> dict:
    output_dict = dict()
    for particle_name, dynamics in particle_dynamics.items():
        output_dict[particle_name] = _dynamics_to_dict(dynamics)
    return output_dict


def _dynamics_to_dict(dynamics: Dynamics) -> dict:
    if isinstance(dynamics, NonDynamic):
        if isinstance(dynamics.form_factor, BlattWeisskopf):
            form_factor = {
                "Type": "BlattWeisskopf",
                "MesonRadius": dynamics.form_factor.meson_radius.value,
            }
        else:
            raise NotImplementedError(
                "No conversion for", dynamics.form_factor
            )
        return {
            "Type": "NonDynamic",
            "FormFactor": form_factor,
        }
    raise NotImplementedError("No conversion for", dynamics)


def _intensity_to_dict(  # pylint: disable=too-many-return-statements
    node: Node,
) -> dict:
    if isinstance(node, StrengthIntensity):
        return {
            "Class": "StrengthIntensity",
            "Component": node.component,
            "Strength": node.strength.name,
            "Intensity": _intensity_to_dict(node.intensity),
        }
    if isinstance(node, NormalizedIntensity):
        return {
            "Class": "NormalizedIntensity",
            "Intensity": _intensity_to_dict(node.intensity),
        }
    if isinstance(node, IncoherentIntensity):
        return {
            "Class": "IncoherentIntensity",
            "Intensities": [
                _intensity_to_dict(intensity) for intensity in node.intensities
            ],
        }
    if isinstance(node, CoherentIntensity):
        return {
            "Class": "CoherentIntensity",
            "Component": node.component,
            "Amplitudes": [
                _intensity_to_dict(intensity) for intensity in node.amplitudes
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
        output_dict["Amplitude"] = _intensity_to_dict(node.amplitude)
        return output_dict
    if isinstance(node, SequentialAmplitude):
        return {
            "Class": "SequentialAmplitude",
            "Amplitudes": [
                _intensity_to_dict(intensity) for intensity in node.amplitudes
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
