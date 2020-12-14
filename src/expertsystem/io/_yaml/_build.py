"""Read recipe objects from a YAML file."""

from typing import Callable, Dict, List, Optional, Union

from expertsystem.amplitude.model import (
    AmplitudeModel,
    AmplitudeNode,
    CanonicalDecay,
    ClebschGordan,
    CoefficientAmplitude,
    CoherentIntensity,
    DecayProduct,
    Dynamics,
    FitParameter,
    FitParameters,
    HelicityDecay,
    HelicityParticle,
    IncoherentIntensity,
    IntensityNode,
    Kinematics,
    KinematicsType,
    NormalizedIntensity,
    ParticleDynamics,
    RecoilSystem,
    SequentialAmplitude,
    StrengthIntensity,
)
from expertsystem.particle import Parity, Particle, ParticleCollection, Spin

from . import validation


def build_amplitude_model(definition: dict) -> AmplitudeModel:
    validation.amplitude_model(definition)
    particles = build_particle_collection(definition, validate=False)
    parameters = __build_fit_parameters(definition["Parameters"])
    kinematics = __build_kinematics(definition["Kinematics"], particles)
    dynamics = __build_particle_dynamics(
        definition["Dynamics"], particles, parameters
    )
    intensity = __build_intensity(
        definition["Intensity"], particles, parameters
    )
    return AmplitudeModel(
        particles=particles,
        kinematics=kinematics,
        parameters=parameters,
        intensity=intensity,
        dynamics=dynamics,
    )


def build_particle_collection(
    definition: dict, validate: bool = True
) -> ParticleCollection:
    if validate:
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
        isospin=__yaml_to_isospin(qn_def.get("IsoSpin", None)),
        parity=__yaml_to_parity(qn_def.get("Parity", None)),
        c_parity=__yaml_to_parity(qn_def.get("CParity", None)),
        g_parity=__yaml_to_parity(qn_def.get("GParity", None)),
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


def __build_fit_parameters(definition: List[dict]) -> FitParameters:
    parameters = FitParameters()
    for parameter_def in definition:
        parameter = __build_fit_parameter(parameter_def)
        parameters.add(parameter)
    return parameters


def __build_fit_parameter(definition: dict) -> FitParameter:
    return FitParameter(
        name=str(definition["Name"]),
        value=float(definition.get("Value", 0.0)),
        is_fixed=bool(definition.get("Fix", False)),
    )


def __build_kinematics(
    definition: dict, particles: ParticleCollection
) -> Kinematics:
    str_to_kinematics_type = {"Helicity": KinematicsType.Helicity}
    kinematics_type = str_to_kinematics_type[definition["Type"]]
    kinematics = Kinematics(
        kinematics_type=kinematics_type,
        particles=particles,
    )
    for item in definition["InitialState"]:
        state_id = int(item["ID"])
        particle_name = str(item["Particle"])
        kinematics.add_initial_state(state_id, particle_name)
    for item in definition["FinalState"]:
        state_id = int(item["ID"])
        particle_name = str(item["Particle"])
        kinematics.add_final_state(state_id, particle_name)
    return kinematics


def __build_particle_dynamics(
    definition: dict,
    particles: ParticleCollection,
    parameters: FitParameters,
) -> ParticleDynamics:
    dynamics = ParticleDynamics(particles=particles, parameters=parameters)
    type_mapping: Dict[str, Callable[[str], Dynamics]] = {
        "NonDynamic": dynamics.set_non_dynamic,
        "RelativisticBreitWigner": dynamics.set_breit_wigner,
    }
    for particle_name, dynamics_def in definition.items():
        dynamics_type = dynamics_def["Type"]
        dynamics_setter = type_mapping.get(dynamics_type, None)
        if dynamics_setter is None:
            raise SyntaxError(
                f"No conversion defined for dynamics type {dynamics_type}"
            )
        dynamics_setter(particle_name)
    return dynamics


def __build_intensity(
    definition: dict, particles: ParticleCollection, parameters: FitParameters
) -> IntensityNode:
    intensity_type = definition["Class"]
    if intensity_type == "StrengthIntensity":
        strength = parameters[definition["Strength"]]
        component = str(definition["Component"])
        return StrengthIntensity(
            component=component,
            strength=strength,
            intensity=__build_intensity(
                definition["Intensity"], particles, parameters
            ),
        )
    if intensity_type == "NormalizedIntensity":
        return NormalizedIntensity(
            intensity=__build_intensity(
                definition["Intensity"], particles, parameters
            )
        )
    if intensity_type == "IncoherentIntensity":
        return IncoherentIntensity(
            intensities=[
                __build_intensity(item, particles, parameters)
                for item in definition["Intensities"]
            ]
        )
    if intensity_type == "CoherentIntensity":
        component = str(definition["Component"])
        amplitudes = [
            __build_amplitude(item, particles, parameters)
            for item in definition["Amplitudes"]
        ]
        return CoherentIntensity(
            component=component,
            amplitudes=amplitudes,
        )
    raise SyntaxError(
        f"No conversion defined for intensity type {intensity_type}"
    )


def __build_amplitude(  # pylint: disable=too-many-locals
    definition: dict, particles: ParticleCollection, parameters: FitParameters
) -> AmplitudeNode:
    amplitude_type = definition["Class"]
    if amplitude_type == "CoefficientAmplitude":
        component = definition["Component"]
        magnitude = parameters[definition["Magnitude"]]
        phase = parameters[definition["Phase"]]
        amplitude = __build_amplitude(
            definition["Amplitude"], particles, parameters
        )
        prefactor = definition.get("PreFactor")
        return CoefficientAmplitude(
            component=component,
            magnitude=magnitude,
            phase=phase,
            amplitude=amplitude,
            prefactor=prefactor,
        )
    if amplitude_type == "SequentialAmplitude":
        amplitudes = [
            __build_amplitude(item, particles, parameters)
            for item in definition["Amplitudes"]
        ]
        return SequentialAmplitude(amplitudes)
    if amplitude_type == "HelicityDecay":
        decay_particle_def = definition["DecayParticle"]
        decaying_particle = HelicityParticle(
            particle=particles[decay_particle_def["Name"]],
            helicity=float(decay_particle_def["Helicity"]),
        )
        decay_products = [
            DecayProduct(
                particles[item["Name"]],
                float(item["Helicity"]),
                list(item["FinalState"]),
            )
            for item in definition["DecayProducts"]
        ]
        recoil_system: Optional[RecoilSystem] = None
        recoil_def = definition.get("RecoilSystem", None)
        if recoil_def is not None:
            recoil_system = RecoilSystem(
                recoil_final_state=recoil_def["RecoilFinalState"]
            )
        canonical_def = definition.get("Canonical", None)
        if canonical_def is None:
            return HelicityDecay(
                decaying_particle, decay_products, recoil_system
            )
        ls_def = canonical_def["LS"]["ClebschGordan"]
        s2s3_def = canonical_def["s2s3"]["ClebschGordan"]
        return CanonicalDecay(
            decaying_particle=decaying_particle,
            decay_products=decay_products,
            recoil_system=recoil_system,
            l_s=ClebschGordan(
                J=float(ls_def["J"]),
                M=float(ls_def["M"]),
                j_1=float(ls_def["j1"]),
                m_1=float(ls_def["m1"]),
                j_2=float(ls_def["j2"]),
                m_2=float(ls_def["m2"]),
            ),
            s2s3=ClebschGordan(
                J=float(s2s3_def["J"]),
                M=float(s2s3_def["M"]),
                j_1=float(s2s3_def["j1"]),
                m_1=float(s2s3_def["m1"]),
                j_2=float(s2s3_def["j2"]),
                m_2=float(s2s3_def["m2"]),
            ),
        )
    raise SyntaxError(
        f"No conversion defined for amplitude type {amplitude_type}"
    )


def __yaml_to_parity(
    definition: Optional[Union[float, int, str]]
) -> Optional[Parity]:
    if definition is None:
        return None
    return Parity(definition)


def __yaml_to_isospin(
    definition: Optional[Union[dict, float, int, str]]
) -> Optional[Spin]:
    if definition is None:
        return None
    return build_spin(definition)
