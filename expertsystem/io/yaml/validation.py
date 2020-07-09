"""JSON validation schema for a YAML recipe file."""

import json
from os.path import dirname, isfile, realpath

from jsonschema import validate

import expertsystem

_PACKAGE_PATH = dirname(realpath(expertsystem.__file__))
_SCHEMA_PATH_PARTICLES = f"{_PACKAGE_PATH}/schemas/yaml-particle-list.json"

if not isfile(_SCHEMA_PATH_PARTICLES):
    raise FileNotFoundError(
        f"Could not find particle validation schema {_SCHEMA_PATH_PARTICLES}"
    )

with open(_SCHEMA_PATH_PARTICLES) as json_file:
    _SCHEMA_PARTICLES = json.load(json_file)


def validate_particle_list(instance: dict) -> None:
    validate(instance=instance, schema=_SCHEMA_PARTICLES)
