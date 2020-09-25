"""Collection of `~enum.Enum` and `dict` mappings.

This module will be phased out through `#254
<https://github.com/ComPWA/expertsystem/issues/254>`_.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Union,
)

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


class AbstractQNConverter(ABC):
    """Abstract interface for a quantum number converter."""

    @abstractmethod
    def parse_from_dict(self, data_dict: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: Any,
    ) -> Dict[str, Any]:
        pass


class _IntQNConverter(AbstractQNConverter):
    """Interface for converting `int` quantum numbers."""

    value_label = Labels.Value.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> int:
        return int(data_dict[self.value_label])

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: int,
    ) -> Dict[str, Any]:
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Int.name,
            self.value_label: qn_value,
        }


class _FloatQNConverter(AbstractQNConverter):
    """Interface for converting `float` quantum numbers."""

    value_label = Labels.Value.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> float:
        return float(data_dict[self.value_label])

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: float,
    ) -> Dict[str, Any]:
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Float.name,
            self.value_label: qn_value,
        }


class _SpinQNConverter(AbstractQNConverter):
    """Interface for converting `.Spin` quantum numbers."""

    type_label = Labels.Type.name
    class_label = Labels.Class.name
    value_label = Labels.Value.name
    proj_label = Labels.Projection.name

    def __init__(self, parse_projection: bool = True) -> None:
        self.parse_projection = bool(parse_projection)

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> Spin:
        mag = data_dict[self.value_label]
        proj = 0.0
        if self.parse_projection:
            if self.proj_label not in data_dict:
                if float(mag) != 0.0:
                    raise ValueError(
                        "No projection set for spin-like quantum number!"
                    )
            else:
                proj = data_dict[self.proj_label]
        return Spin(mag, proj)

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: Spin,
    ) -> Dict[str, Any]:
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Spin.name,
            self.value_label: qn_value.magnitude,
            self.proj_label: qn_value.projection,
        }


QNClassConverterMapping = {
    QuantumNumberClasses.Int: _IntQNConverter(),
    QuantumNumberClasses.Float: _FloatQNConverter(),
    QuantumNumberClasses.Spin: _SpinQNConverter(),
}


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
    EdgeQuantumNumbers.pid.__name__: ParticlePropertyNames.Pid,
    EdgeQuantumNumbers.mass.__name__: ParticlePropertyNames.Mass,
    EdgeQuantumNumbers.width.__name__: ParticleDecayPropertyNames.Width,
    EdgeQuantumNumbers.spin_magnitude.__name__: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.spin_projection.__name__: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.charge.__name__: StateQuantumNumberNames.Charge,
    EdgeQuantumNumbers.isospin_magnitude.__name__: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.isospin_projection.__name__: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.strangeness.__name__: StateQuantumNumberNames.Strangeness,
    EdgeQuantumNumbers.charmness.__name__: StateQuantumNumberNames.Charmness,
    EdgeQuantumNumbers.bottomness.__name__: StateQuantumNumberNames.Bottomness,
    EdgeQuantumNumbers.topness.__name__: StateQuantumNumberNames.Topness,
    EdgeQuantumNumbers.baryon_number.__name__: StateQuantumNumberNames.BaryonNumber,
    EdgeQuantumNumbers.electron_lepton_number.__name__: StateQuantumNumberNames.ElectronLN,
    EdgeQuantumNumbers.muon_lepton_number.__name__: StateQuantumNumberNames.MuonLN,
    EdgeQuantumNumbers.tau_lepton_number.__name__: StateQuantumNumberNames.TauLN,
    EdgeQuantumNumbers.parity.__name__: StateQuantumNumberNames.Parity,
    EdgeQuantumNumbers.c_parity.__name__: StateQuantumNumberNames.CParity,
    EdgeQuantumNumbers.g_parity.__name__: StateQuantumNumberNames.GParity,
    NodeQuantumNumbers.l_magnitude.__name__: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.l_projection.__name__: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.s_magnitude.__name__: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.s_projection.__name__: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.parity_prefactor.__name__: InteractionQuantumNumberNames.ParityPrefactor,
}
