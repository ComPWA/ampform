"""Serialization from and to a YAML recipe file."""

import yaml

from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import Particle, ParticleCollection

from . import _build, _dump


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)

    def write_line_break(self, data=None):  # type: ignore
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def load_amplitude_model(filename: str) -> AmplitudeModel:
    with open(filename) as yaml_file:
        definition = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return _build.build_amplitude_model(definition)


def load_particle_collection(filename: str) -> ParticleCollection:
    with open(filename) as yaml_file:
        definition = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return _build.build_particle_collection(definition)


def write(instance: object, filename: str) -> None:
    if isinstance(instance, ParticleCollection):
        output_dict = _dump.from_particle_collection(instance)
    elif isinstance(instance, AmplitudeModel):
        output_dict = _dump.from_amplitude_model(instance)
    else:
        raise NotImplementedError(
            f"No YAML writer for class {instance.__class__.__name__}"
        )
    with open(filename, "w") as yaml_file:
        yaml.dump(
            output_dict,
            yaml_file,
            sort_keys=False,
            Dumper=_IncreasedIndent,
            default_flow_style=False,
        )


def object_to_dict(instance: object) -> dict:
    if isinstance(instance, ParticleCollection):
        return _dump.from_particle_collection(instance)
    if isinstance(instance, Particle):
        return _dump.from_particle(instance)
    raise NotImplementedError
