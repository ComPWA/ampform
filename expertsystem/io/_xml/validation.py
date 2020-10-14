"""JSON validation schema for `dict` instances that come from old XML files.

While support for XML will be slowly phased out (it was previously required by
`pycompwa`), the `expertsystem` internally still works with a specific nested
`dict` structure. This is most problematic in the `.particle` module — the
starting point for the expertsystem. This module helps validating the structure
of such nested `dict` instances.
"""

import json
from os.path import dirname, realpath

import jsonschema

_EXPERTSYSTEM_PATH = f"{dirname(realpath(__file__))}/../.."

with open(f"{_EXPERTSYSTEM_PATH}/schemas/xml/particle.json") as stream:
    _SCHEMA_PARTICLE = json.load(stream)
with open(f"{_EXPERTSYSTEM_PATH}/schemas/xml/particle-list.json") as stream:
    _SCHEMA_PARTICLES = json.load(stream)

_RESOLVER_PARTICLE = jsonschema.RefResolver.from_schema(_SCHEMA_PARTICLE)


def particle(instance: dict) -> None:
    jsonschema.validate(instance=instance, schema=_SCHEMA_PARTICLE)


def particle_list(instance: dict) -> None:
    jsonschema.validate(
        instance=instance,
        schema=_SCHEMA_PARTICLES,
        resolver=_RESOLVER_PARTICLE,
    )
