"""Collection of `~enum.Enum` and `dict` mappings.

This module will be phased out through `#254
<https://github.com/ComPWA/expertsystem/issues/254>`_.
"""

from enum import Enum, auto
from typing import Any, Dict

from expertsystem.data import (
    EdgeQuantumNumbers,
    NodeQuantumNumbers,
    Spin,
)


class Labels(Enum):
    """Labels that are useful in the particle module."""

    Class = auto()
    Component = auto()
    DecayInfo = auto()
    Name = auto()
    Parameter = auto()
    Pid = auto()
    PreFactor = auto()
    Projection = auto()
    QuantumNumber = auto()
    Type = auto()
    Value = auto()


class QuantumNumberClasses(Enum):
    """Types of quantum number classes in the form of an enumerate."""

    Int = auto()
    Float = auto()
    Spin = auto()


class StateQuantumNumberNames(Enum):
    """Definition of quantum number names for states."""

    BaryonNumber = auto()
    Bottomness = auto()
    Charge = auto()
    Charmness = auto()
    CParity = auto()
    ElectronLN = auto()
    GParity = auto()
    IsoSpin = auto()
    MuonLN = auto()
    Parity = auto()
    Spin = auto()
    Strangeness = auto()
    TauLN = auto()
    Topness = auto()


class ParticlePropertyNames(Enum):
    """Definition of properties names of particles."""

    Pid = auto()
    Mass = auto()


class ParticleDecayPropertyNames(Enum):
    """Definition of decay properties names of particles."""

    Width = auto()


class InteractionQuantumNumberNames(Enum):
    """Definition of quantum number names for interaction nodes."""

    L = auto()
    S = auto()
    ParityPrefactor = auto()


QNDefaultValues: Dict[StateQuantumNumberNames, Any] = {
    StateQuantumNumberNames.Charge: 0,
    StateQuantumNumberNames.IsoSpin: Spin(0.0, 0.0),
    StateQuantumNumberNames.Strangeness: 0,
    StateQuantumNumberNames.Charmness: 0,
    StateQuantumNumberNames.Bottomness: 0,
    StateQuantumNumberNames.Topness: 0,
    StateQuantumNumberNames.BaryonNumber: 0,
    StateQuantumNumberNames.ElectronLN: 0,
    StateQuantumNumberNames.MuonLN: 0,
    StateQuantumNumberNames.TauLN: 0,
}

QNNameClassMapping = {
    StateQuantumNumberNames.Charge: QuantumNumberClasses.Int,
    StateQuantumNumberNames.ElectronLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.MuonLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.TauLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.BaryonNumber: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Spin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Parity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.CParity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.GParity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.IsoSpin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Strangeness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Charmness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Bottomness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Topness: QuantumNumberClasses.Int,
    InteractionQuantumNumberNames.L: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.S: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.ParityPrefactor: QuantumNumberClasses.Int,
    ParticlePropertyNames.Pid: QuantumNumberClasses.Int,
    ParticlePropertyNames.Mass: QuantumNumberClasses.Float,
    ParticleDecayPropertyNames.Width: QuantumNumberClasses.Float,
}

edge_qn_to_enum = {
    EdgeQuantumNumbers.pid: ParticlePropertyNames.Pid,
    EdgeQuantumNumbers.mass: ParticlePropertyNames.Mass,
    EdgeQuantumNumbers.width: ParticleDecayPropertyNames.Width,
    EdgeQuantumNumbers.spin_magnitude: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.spin_projection: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.charge: StateQuantumNumberNames.Charge,
    EdgeQuantumNumbers.isospin_magnitude: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.isospin_projection: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.strangeness: StateQuantumNumberNames.Strangeness,
    EdgeQuantumNumbers.charmness: StateQuantumNumberNames.Charmness,
    EdgeQuantumNumbers.bottomness: StateQuantumNumberNames.Bottomness,
    EdgeQuantumNumbers.topness: StateQuantumNumberNames.Topness,
    EdgeQuantumNumbers.baryon_number: StateQuantumNumberNames.BaryonNumber,
    EdgeQuantumNumbers.electron_lepton_number: StateQuantumNumberNames.ElectronLN,
    EdgeQuantumNumbers.muon_lepton_number: StateQuantumNumberNames.MuonLN,
    EdgeQuantumNumbers.tau_lepton_number: StateQuantumNumberNames.TauLN,
    EdgeQuantumNumbers.parity: StateQuantumNumberNames.Parity,
    EdgeQuantumNumbers.c_parity: StateQuantumNumberNames.CParity,
    EdgeQuantumNumbers.g_parity: StateQuantumNumberNames.GParity,
    NodeQuantumNumbers.l_magnitude: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.l_projection: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.s_magnitude: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.s_projection: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.parity_prefactor: InteractionQuantumNumberNames.ParityPrefactor,
}
