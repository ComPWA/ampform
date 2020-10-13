"""Serialization module for the `expertsystem`.

The `.io` module provides tools to export or import objects from the
:mod:`.particle`, :mod:`.reaction` and :mod:`.amplitude` modules to and from
disk, so that they can be used by external packages, or just to store (cache)
the state of the system.
"""

from pathlib import Path

from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection

from . import _pdg, dot, xml, yaml


def load_amplitude_model(filename: str) -> AmplitudeModel:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return yaml.load_amplitude_model(filename)
    if file_extension == "xml":
        return xml.load_amplitude_model(filename)
    raise NotImplementedError(
        f'No parser parser defined for file type "{file_extension}"'
    )


def load_particle_collection(filename: str) -> ParticleCollection:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return yaml.load_particle_collection(filename)
    if file_extension == "xml":
        return xml.load_particle_collection(filename)
    raise NotImplementedError(
        f'No parser parser defined for file type "{file_extension}"'
    )


def load_pdg() -> ParticleCollection:
    """Create a `.ParticleCollection` with all entries from the PDG.

    PDG info is imported from the `scikit-hep/particle
    <https://github.com/scikit-hep/particle/blob/master/README.rst>`_ package.
    """
    return _pdg.load_pdg()


def write(instance: object, filename: str) -> None:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return yaml.write(instance, filename)
    if file_extension == "xml":
        return xml.write(instance, filename)
    if file_extension == "gv":
        return dot.write(instance, filename)
    raise NotImplementedError(
        f'No writer defined for file type "{file_extension}"'
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise Exception(f"No file extension in file {filename}")
    extension = extension[1:]
    return extension
