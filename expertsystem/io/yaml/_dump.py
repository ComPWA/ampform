"""Dump recipe objects to `dict` instances for a YAML file."""

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)

from . import validation


def from_particle_collection(particles: ParticleCollection) -> dict:
    output = dict()
    for name, particle in particles.items():
        output[name] = from_particle(particle)
    output = {"ParticleList": output}
    validation.particle_list(output)
    return output


def from_particle(particle: Particle) -> dict:
    output_dict: Dict[str, Union[float, int, dict]] = {
        "PID": particle.pid,
        "Mass": particle.mass,
    }
    if particle.width != 0.0:
        output_dict["Width"] = particle.width
    output_dict["QuantumNumbers"] = _to_quantum_number_dict(particle)
    return output_dict


def _to_quantum_number_dict(
    particle: Particle,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    output_dict: Dict[str, Union[float, int, Dict[str, float]]] = {
        "Spin": _attempt_to_int(particle.spin),
        "Charge": int(particle.charge),
    }
    optional_qn: List[
        Tuple[str, Union[Optional[Parity], Spin, int], Union[Callable, int]]
    ] = [
        ("Parity", particle.parity, int),
        ("CParity", particle.c_parity, int),
        ("GParity", particle.g_parity, int),
        ("Strangeness", particle.strangeness, int),
        ("Charmness", particle.charmness, int),
        ("Bottomness", particle.bottomness, int),
        ("Topness", particle.topness, int),
        ("BaryonNumber", particle.baryon_number, int),
        ("ElectronLN", particle.electron_lepton_number, int),
        ("MuonLN", particle.muon_lepton_number, int),
        ("TauLN", particle.tau_lepton_number, int),
    ]
    for key, value, converter in optional_qn:
        if value in [0, None]:
            continue
        output_dict[key] = converter(  # type: ignore
            value
        )  # pylint: disable=not-callable
    if particle.isospin is not None:
        output_dict["IsoSpin"] = _from_spin(particle.isospin)
    return output_dict


def _from_spin(instance: Spin) -> Union[Dict[str, Union[float, int]], int]:
    if instance.magnitude == 0:
        return 0
    return {
        "Value": _attempt_to_int(instance.magnitude),
        "Projection": _attempt_to_int(instance.projection),
    }


def _attempt_to_int(value: Union[Spin, float, int]) -> Union[float, int]:
    if isinstance(value, Spin):
        value = float(value)
    if value.is_integer():
        return int(value)
    return value
