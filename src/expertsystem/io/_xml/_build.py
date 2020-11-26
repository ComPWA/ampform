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
    particles = build_particle_collection(definition)
    kinematics = __build_kinematics(definition, particles)
    parameters = FitParameters()
    dynamics = __build_particle_dynamics(
        definition["ParticleList"]["Particle"], particles, parameters
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
    qn_defs = __xml_qn_list_to_qn_object(definition["QuantumNumber"])
    return Particle(
        name=str(definition["Name"]),
        pid=int(definition["Pid"]),
        mass=float(definition["Parameter"]["Value"]),
        width=__xml_to_width(definition),
        charge=int(qn_defs["Charge"]),
        spin=float(qn_defs["Spin"]),
        isospin=qn_defs.get("IsoSpin", None),
        strangeness=int(qn_defs.get("Strangeness", 0)),
        charmness=int(qn_defs.get("Charmness", 0)),
        bottomness=int(qn_defs.get("Bottomness", 0)),
        topness=int(qn_defs.get("Topness", 0)),
        baryon_number=int(qn_defs.get("BaryonNumber", 0)),
        electron_lepton_number=int(qn_defs.get("ElectronLN", 0)),
        muon_lepton_number=int(qn_defs.get("MuonLN", 0)),
        tau_lepton_number=int(qn_defs.get("TauLN", 0)),
        parity=qn_defs.get("Parity", None),
        c_parity=qn_defs.get("CParity", None),
        g_parity=qn_defs.get("GParity", None),
    )


def build_spin(definition: dict) -> Spin:
    magnitude = definition["Value"]
    projection = definition.get("Projection", 0.0)
    return Spin(magnitude, projection)


def __build_fit_parameter(
    definition: dict, parameters: FitParameters
) -> FitParameter:
    parameter_name = str(definition["Name"])
    if parameter_name in parameters:
        return parameters[parameter_name]
    parameter = FitParameter(
        name=parameter_name,
        value=float(definition["Value"]),
        is_fixed=(definition["Fix"] == "true"),  # XML problem
    )
    parameters.add(parameter)
    return parameter


def __build_kinematics(
    definition: dict, particles: ParticleCollection
) -> Kinematics:
    str_to_kinematics_type = {"HelicityKinematics": KinematicsType.Helicity}
    kinematics_type = None
    for section in definition:
        kinematics_type = str_to_kinematics_type.get(section)
        if kinematics_type is not None:
            definition = definition[section]
            break
    if kinematics_type is None:
        raise SyntaxError(
            "XML file does not contain a kinematics section of any of the types:",
            set(str_to_kinematics_type),
        )
    kinematics = Kinematics(
        kinematics_type=kinematics_type,
        particles=particles,
    )
    for item in __safe_wrap_in_list(definition["InitialState"]):
        particle_def = item["Particle"]
        state_id = int(particle_def["Id"])
        particle_name = str(particle_def["Name"])
        kinematics.add_initial_state(state_id, particle_name)
    for item in __safe_wrap_in_list(definition["FinalState"]["Particle"]):
        state_id = int(item["Id"])
        particle_name = str(item["Name"])
        kinematics.add_final_state(state_id, particle_name)
    return kinematics


def __build_particle_dynamics(
    definition: List[dict],
    particles: ParticleCollection,
    parameters: FitParameters,
) -> ParticleDynamics:
    dynamics = ParticleDynamics(particles=particles, parameters=parameters)
    type_mapping: Dict[str, Callable[[str], Dynamics]] = {
        "nonResonant": dynamics.set_non_dynamic,
        "relativisticBreitWigner": dynamics.set_breit_wigner,
    }
    for particle_def in definition:
        decay_info: Optional[dict] = particle_def.get("DecayInfo")
        if decay_info is None:
            continue
        dynamics_type = decay_info.get("Type")
        if dynamics_type is None:
            continue
        dynamics_setter = type_mapping.get(dynamics_type, None)
        if dynamics_setter is None:
            raise SyntaxError(
                f"No conversion defined for dynamics type {dynamics_type}"
            )
        particle_name = particle_def["Name"]
        dynamics_setter(particle_name)
    return dynamics


def __build_intensity(
    definition: dict, particles: ParticleCollection, parameters: FitParameters
) -> IntensityNode:
    intensity_type = definition["Class"]
    if intensity_type == "StrengthIntensity":
        strength = __build_fit_parameter(definition["Parameter"], parameters)
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
                for item in definition["Intensity"]
            ]
        )
    if intensity_type == "CoherentIntensity":
        component = str(definition["Component"])
        amplitudes = [
            __build_amplitude(item, particles, parameters)
            for item in definition["Amplitude"]
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
        parameter_defs = definition["Parameter"]
        magnitude = __build_fit_parameter(
            next(filter(lambda p: p["Type"] == "Magnitude", parameter_defs)),
            parameters,
        )
        phase = __build_fit_parameter(
            next(filter(lambda p: p["Type"] == "Phase", parameter_defs)),
            parameters,
        )
        amplitude = __build_amplitude(
            definition["Amplitude"], particles, parameters
        )
        prefactor = definition.get("Prefactor", {}).get("Real")
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
            for item in __safe_wrap_in_list(definition["Amplitude"])
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
                particle=particles[item["Name"]],
                helicity=float(item["Helicity"]),
                final_state_ids=[
                    int(state_id) for state_id in item["FinalState"].split(" ")
                ],
            )
            for item in __safe_wrap_in_list(
                definition["DecayProducts"]["Particle"]
            )
        ]
        recoil_system: Optional[RecoilSystem] = None
        recoil_def = definition.get("RecoilSystem", None)
        if recoil_def is not None:
            recoil_system = RecoilSystem(
                recoil_final_state=[
                    int(state_id)
                    for state_id in __safe_wrap_in_list(
                        recoil_def["RecoilFinalState"]
                    )
                ]
            )
        canonical_def = definition.get("CanonicalSum", None)
        if canonical_def is None:
            return HelicityDecay(
                decaying_particle, decay_products, recoil_system
            )
        clebsch_gordan = __safe_wrap_in_list(canonical_def["ClebschGordan"])
        ls_def = next(filter(lambda p: p["Type"] == "LS", clebsch_gordan))
        s2s3_def = next(filter(lambda p: p["Type"] == "s2s3", clebsch_gordan))
        return CanonicalDecay(
            decaying_particle=decaying_particle,
            decay_products=decay_products,
            recoil_system=recoil_system,
            l_s=ClebschGordan(
                J=float(ls_def["J"]),
                M=float(ls_def["M"]),
                j_1=float(ls_def["@j1"]),
                m_1=float(ls_def["@m1"]),
                j_2=float(ls_def["@j2"]),
                m_2=float(ls_def["@m2"]),
            ),
            s2s3=ClebschGordan(
                J=float(s2s3_def["J"]),
                M=float(s2s3_def["M"]),
                j_1=float(s2s3_def["@j1"]),
                m_1=float(s2s3_def["@m1"]),
                j_2=float(s2s3_def["@j2"]),
                m_2=float(s2s3_def["@m2"]),
            ),
        )
    raise SyntaxError(
        f"No conversion defined for amplitude type {amplitude_type}"
    )


def __xml_to_width(definition: dict) -> float:
    definition = definition.get("DecayInfo", {})
    definition = definition.get("Parameter", None)
    if isinstance(definition, list):
        for item in definition:  # type: ignore
            if item["Type"] == "Width":
                definition = item
                break
    if definition is None or not isinstance(definition, dict):
        return 0.0
    return float(definition["Value"])


def __xml_qn_list_to_qn_object(definitions: List[dict]) -> Dict[str, Any]:
    output = dict()
    for definition in definitions:
        type_name, quantum_number = __xml_to_quantum_number(definition)
        output[type_name] = quantum_number
    return output


def __xml_to_quantum_number(definition: Dict[str, str]) -> Tuple[str, Any]:
    conversion_map: Dict[str, Callable] = {
        "Spin": __xml_to_float,
        "Charge": __xml_to_int,
        "Strangeness": __xml_to_int,
        "Charmness": __xml_to_int,
        "BaryonNumber": __xml_to_int,
        "ElectronLN": __xml_to_int,
        "MuonLN": __xml_to_int,
        "TauLN": __xml_to_int,
        "Parity": __xml_to_parity,
        "CParity": __xml_to_parity,
        "GParity": __xml_to_parity,
        "IsoSpin": build_spin,
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


def __xml_to_float(definition: dict) -> float:
    return float(definition["Value"])


def __xml_to_int(definition: dict) -> int:
    return int(definition["Value"])


def __xml_to_parity(definition: dict) -> Parity:
    return Parity(__xml_to_int(definition))


def __safe_wrap_in_list(instance: object) -> list:
    if isinstance(instance, list):
        return instance
    return [instance]
