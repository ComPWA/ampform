"""Serialization module for containers of `expertsystem.data`."""

__all__ = [
    "load_particle_collection",
    "write",
]

from pathlib import Path

from expertsystem.data import ParticleCollection

from . import _yaml


def load_particle_collection(filename: str) -> ParticleCollection:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return _yaml.load_particle_collection(filename)
    raise NotImplementedError(
        f'No parser parser defined for file type "{file_extension}"'
    )


def write(instance: object, filename: str) -> None:
    file_extension = _get_file_extension(filename)
    if file_extension in ["yaml", "yml"]:
        return _yaml.write(instance, filename)
    raise NotImplementedError(
        f'No parser writer defined for file type "{file_extension}"'
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise Exception(f"No file extension in file {filename}")
    extension = extension[1:]
    return extension
