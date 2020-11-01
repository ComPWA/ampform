"""Serialization module for the `expertsystem`.

The `.io` module provides tools to export or import objects from the
:mod:`.particle`, :mod:`.reaction` and :mod:`.amplitude` modules to and from
disk, so that they can be used by external packages, or just to store (cache)
the state of the system.
"""

from pathlib import Path

from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection
from expertsystem.reaction.topology import StateTransitionGraph, Topology

from . import _dot, _pdg, _xml, _yaml


def load_amplitude_model(filename: str) -> AmplitudeModel:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return _yaml.load_amplitude_model(filename)
    if file_extension == "xml":
        return _xml.load_amplitude_model(filename)
    raise NotImplementedError(
        f'No parser parser defined for file type "{file_extension}"'
    )


def load_particle_collection(filename: str) -> ParticleCollection:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return _yaml.load_particle_collection(filename)
    if file_extension == "xml":
        return _xml.load_particle_collection(filename)
    raise NotImplementedError(
        f'No parser parser defined for file type "{file_extension}"'
    )


def load_pdg() -> ParticleCollection:
    """Create a `.ParticleCollection` with all entries from the PDG.

    PDG info is imported from the `scikit-hep/particle
    <https://github.com/scikit-hep/particle>`_ package.
    """
    return _pdg.load_pdg()


def write(instance: object, filename: str) -> None:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return _yaml.write(instance, filename)
    if file_extension == "xml":
        return _xml.write(instance, filename)
    if file_extension == "gv":
        output_str = convert_to_dot(instance)
        with open(filename, "w") as stream:
            stream.write(output_str)
        return None
    raise NotImplementedError(
        f'No writer defined for file type "{file_extension}"'
    )


def convert_to_dot(instance: object) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph` or a `list` of `.StateTransitionGraph` instances.

    .. seealso:: :doc:`/usage/visualization`
    """
    if isinstance(instance, (StateTransitionGraph, Topology)):
        return _dot.graph_to_dot(instance)
    if isinstance(instance, list):
        return _dot.graph_list_to_dot(instance)
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise Exception(f"No file extension in file {filename}")
    extension = extension[1:]
    return extension
