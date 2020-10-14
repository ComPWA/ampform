"""JSON validation schema for a YAML recipe file."""

import json
from os.path import dirname, realpath

import jsonschema
from jsonschema import RefResolver

_EXPERTSYSTEM_PATH = f"{dirname(realpath(__file__))}/../.."

with open(f"{_EXPERTSYSTEM_PATH}/schemas/yaml/particle-list.json") as stream:
    _SCHEMA_PARTICLES = json.load(stream)
with open(f"{_EXPERTSYSTEM_PATH}/schemas/yaml/amplitude-model.json") as stream:
    _SCHEMA_AMPLITUDE = json.load(stream)


def particle_list(instance: dict) -> None:
    jsonschema.validate(instance=instance, schema=_SCHEMA_PARTICLES)


def amplitude_model(instance: dict) -> None:
    resolver = RefResolver(
        # The key part is here where we build a custom RefResolver
        # and tell it where *this* schema lives in the filesystem
        # Note that `file:` is for unix systems
        f"file://{_EXPERTSYSTEM_PATH}/schemas/yaml/",
        "amplitude-model.json",
    )

    jsonschema.validate(
        instance=instance,
        schema=_SCHEMA_AMPLITUDE,
        resolver=resolver,
    )
