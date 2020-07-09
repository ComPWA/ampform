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

import expertsystem

_PACKAGE_PATH = dirname(realpath(expertsystem.__file__))
_SCHEMA_PATH_PARTICLE = f"{_PACKAGE_PATH}/schemas/xml/particle.json"


with open(_SCHEMA_PATH_PARTICLE) as json_file:
    _SCHEMA_PARTICLE = json.load(json_file)


def validate_particle(definition: dict) -> None:
    jsonschema.validate(instance=definition, schema=_SCHEMA_PARTICLE)
