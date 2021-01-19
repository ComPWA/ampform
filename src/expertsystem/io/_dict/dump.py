"""Dump recipe objects to `dict` instances for a YAML file."""
from collections import abc
from enum import Enum
from typing import Any

import attr

from expertsystem.amplitude.model import (
    AmplitudeModel,
    FitParameter,
    FitParameters,
    FormFactor,
    Kinematics,
    Node,
    ParticleDynamics,
)
from expertsystem.particle import Parity, Particle, ParticleCollection, Spin


def from_amplitude_model(model: AmplitudeModel) -> dict:
    return {
        "kinematics": __kinematics_to_dict(model.kinematics),
        **from_fit_parameters(model.parameters),
        "intensity": __asdict_with_type(model.intensity),
        **from_particle_collection(model.particles),
        "dynamics": __dynamics_section_to_dict(model.dynamics),
    }


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {"particles": [from_particle(p) for p in particles]}


def from_particle(particle: Particle) -> dict:
    return attr.asdict(
        particle,
        recurse=True,
        value_serializer=__value_serializer,
        filter=lambda attr, value: attr.default != value,
    )


def from_fit_parameters(parameters: FitParameters) -> dict:
    return {"parameters": [from_fit_parameter(p) for p in parameters.values()]}


def from_fit_parameter(parameter: FitParameter) -> dict:
    return attr.asdict(parameter, recurse=True)


def __kinematics_to_dict(kinematics: Kinematics) -> dict:
    return attr.asdict(
        kinematics,
        recurse=True,
        value_serializer=__value_serializer,
    )


def __dynamics_section_to_dict(particle_dynamics: ParticleDynamics) -> dict:
    return {
        particle_name: __asdict_with_type(dynamics)
        for particle_name, dynamics in particle_dynamics.items()
    }


def __value_serializer(  # pylint: disable=too-many-return-statements,unused-argument
    inst: type, field: attr.Attribute, value: Any
) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, abc.Mapping):
        if all(map(lambda p: isinstance(p, Particle), value.values())):
            return {k: v.name for k, v in value.items()}
    if isinstance(value, abc.Iterable):
        if all(map(lambda p: isinstance(p, Node), value)):
            return [__asdict_with_type(item) for item in value]
    if isinstance(value, (FormFactor, Node)):
        return __asdict_with_type(value)
    if isinstance(value, (FitParameter, Particle)):
        return value.name
    if isinstance(value, Parity):
        return {"value": value.value}
    if isinstance(value, Spin):
        return {
            "magnitude": value.magnitude,
            "projection": value.projection,
        }
    return value


def __asdict_with_type(instance: object) -> dict:
    return {
        "type": instance.__class__.__name__,
        **attr.asdict(
            instance,
            filter=lambda attr, value: attr.default != value,
            recurse=True,
            value_serializer=__value_serializer,
        ),
    }
