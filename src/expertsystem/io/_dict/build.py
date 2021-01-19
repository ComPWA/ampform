"""Read recipe objects from a YAML file."""


from expertsystem.amplitude.model import (
    AmplitudeModel,
    AmplitudeNode,
    BlattWeisskopf,
    CanonicalDecay,
    ClebschGordan,
    CoefficientAmplitude,
    CoherentIntensity,
    DecayProduct,
    Dynamics,
    FitParameter,
    FitParameters,
    FormFactor,
    HelicityDecay,
    HelicityParticle,
    IncoherentIntensity,
    IntensityNode,
    Kinematics,
    KinematicsType,
    NonDynamic,
    NormalizedIntensity,
    ParticleDynamics,
    RecoilSystem,
    RelativisticBreitWigner,
    SequentialAmplitude,
    StrengthIntensity,
)
from expertsystem.particle import Parity, Particle, ParticleCollection, Spin

from . import validate


def build_amplitude_model(definition: dict) -> AmplitudeModel:
    validate.amplitude_model(definition)
    particles = build_particle_collection(definition, do_validate=False)
    parameters = build_fit_parameters(definition)
    kinematics = __build_kinematics(definition["kinematics"], particles)
    dynamics = __build_particle_dynamics(
        definition["dynamics"], particles, parameters
    )
    intensity = __build_intensity(
        definition["intensity"], particles, parameters
    )
    return AmplitudeModel(
        particles=particles,
        kinematics=kinematics,
        parameters=parameters,
        intensity=intensity,
        dynamics=dynamics,
    )


def build_particle_collection(
    definition: dict, do_validate: bool = True
) -> ParticleCollection:
    if do_validate:
        validate.particle_collection(definition)
    return ParticleCollection(
        __build_particle(p) for p in definition["particles"]
    )


def __build_particle(definition: dict) -> Particle:
    isospin_def = definition.get("isospin", None)
    if isospin_def is not None:
        definition["isospin"] = Spin(**isospin_def)
    for parity in ["parity", "c_parity", "g_parity"]:
        parity_def = definition.get(parity, None)
        if parity_def is not None:
            definition[parity] = Parity(**parity_def)
    return Particle(**definition)


def build_fit_parameters(definition: dict) -> FitParameters:
    return FitParameters(FitParameter(**p) for p in definition["parameters"])


def __build_kinematics(
    definition: dict, particles: ParticleCollection
) -> Kinematics:
    kinematics_type = eval(  # pylint: disable=eval-used
        f'{KinematicsType.__name__}.{definition["type"]}'
    )
    return Kinematics(
        type=kinematics_type,
        initial_state={
            i: particles[n] for i, n in definition["initial_state"].items()
        },
        final_state={
            i: particles[n] for i, n in definition["final_state"].items()
        },
    )


def __build_particle_dynamics(
    definition: dict,
    particles: ParticleCollection,
    parameters: FitParameters,
) -> ParticleDynamics:
    particle_dynamics = ParticleDynamics(
        particles=particles, parameters=parameters
    )
    for particle_name, dynamics_def in definition.items():
        particle_dynamics[particle_name] = __build_dynamics(
            dynamics_def, parameters
        )
    return particle_dynamics


def __build_dynamics(definition: dict, parameters: FitParameters) -> Dynamics:
    dynamics_type = definition["type"]
    form_factor = definition.get("form_factor")
    if form_factor is not None:
        form_factor = __build_form_factor(form_factor, parameters)
    if dynamics_type == "NonDynamic":
        return NonDynamic(form_factor)
    if dynamics_type == "RelativisticBreitWigner":
        return RelativisticBreitWigner(
            form_factor=form_factor,
            pole_real=__safely_get_parameter(
                definition["pole_real"], parameters
            ),
            pole_imag=__safely_get_parameter(
                definition["pole_imag"], parameters
            ),
        )
    raise ValueError(f'Dynamics type "{dynamics_type}" not defined')


def __build_form_factor(
    definition: dict, parameters: FitParameters
) -> FormFactor:
    form_factor_type = definition["type"]
    if form_factor_type == "BlattWeisskopf":
        par_name = definition["meson_radius"]
        meson_radius = __safely_get_parameter(par_name, parameters)
        return BlattWeisskopf(meson_radius)
    raise NotImplementedError(
        f'Form factor "{form_factor_type}" does not exist'
    )


def __safely_get_parameter(
    name: str, parameters: FitParameters
) -> FitParameter:
    if name not in parameters:
        raise SyntaxError(
            "Meson radius has not been defined in the Parameters section"
        )
    return parameters[name]


def __build_intensity(
    definition: dict, particles: ParticleCollection, parameters: FitParameters
) -> IntensityNode:
    intensity_type = eval(definition["type"])  # pylint: disable=eval-used
    if intensity_type in {NormalizedIntensity, StrengthIntensity}:
        intensity = __build_intensity(
            definition["intensity"], particles, parameters
        )
        if intensity_type is NormalizedIntensity:
            return NormalizedIntensity(intensity=intensity)
        return StrengthIntensity(
            component=str(definition["component"]),
            strength=parameters[definition["Strength"]],
            intensity=intensity,
        )
    if intensity_type is IncoherentIntensity:
        return IncoherentIntensity(
            intensities=[
                __build_intensity(item, particles, parameters)
                for item in definition["intensities"]
            ]
        )
    if intensity_type is CoherentIntensity:
        return CoherentIntensity(
            component=str(definition["component"]),
            amplitudes=[
                __build_amplitude(item, particles, parameters)
                for item in definition["amplitudes"]
            ],
        )
    raise SyntaxError(
        f"No conversion defined for intensity type {intensity_type}"
    )


def __build_amplitude(  # pylint: disable=too-many-locals
    definition: dict, particles: ParticleCollection, parameters: FitParameters
) -> AmplitudeNode:
    amplitude_type = eval(definition["type"])  # pylint: disable=eval-used
    if amplitude_type is CoefficientAmplitude:
        return CoefficientAmplitude(
            component=definition["component"],
            magnitude=parameters[definition["magnitude"]],
            phase=parameters[definition["phase"]],
            amplitude=__build_amplitude(
                definition["amplitude"], particles, parameters
            ),
            prefactor=definition.get("prefactor"),
        )
    if amplitude_type is SequentialAmplitude:
        return SequentialAmplitude(
            amplitudes=[
                __build_amplitude(item, particles, parameters)
                for item in definition["amplitudes"]
            ]
        )
    if amplitude_type in {CanonicalDecay, HelicityDecay}:
        decaying_particle_def = definition["decaying_particle"]
        decaying_particle = HelicityParticle(
            particle=particles[decaying_particle_def["particle"]],
            helicity=decaying_particle_def["helicity"],
        )
        decay_products = [
            DecayProduct(
                particle=particles[item["particle"]],
                helicity=item["helicity"],
                final_state_ids=item["final_state_ids"],
            )
            for item in definition["decay_products"]
        ]
        recoil_system = definition.get("recoil_system", None)
        if recoil_system is not None:
            recoil_system = RecoilSystem(**recoil_system)
        if amplitude_type is HelicityDecay:
            return HelicityDecay(
                decaying_particle=decaying_particle,
                decay_products=decay_products,
                recoil_system=recoil_system,
            )
        return CanonicalDecay(
            decaying_particle=decaying_particle,
            decay_products=decay_products,
            recoil_system=recoil_system,
            l_s=ClebschGordan(**definition["l_s"]),
            s2s3=ClebschGordan(**definition["s2s3"]),
        )
    raise SyntaxError(
        f"No conversion defined for amplitude type {amplitude_type}"
    )
