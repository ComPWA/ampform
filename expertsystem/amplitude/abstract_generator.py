"""Abstract interfaces for amplitude model generation."""

from abc import ABC, abstractmethod
from typing import List, Optional

from expertsystem.topology import StateTransitionGraph


class AbstractAmplitudeNameGenerator(ABC):
    """Abstract interface for a parameter name generator."""

    @abstractmethod
    def generate_unique_amplitude_name(
        self, graph: StateTransitionGraph, node_id: Optional[int] = None
    ) -> str:
        pass

    @abstractmethod
    def generate_amplitude_coefficient_infos(
        self, graph: StateTransitionGraph
    ) -> dict:
        pass

    @abstractmethod
    def register_amplitude_coefficient_name(
        self, graph: StateTransitionGraph
    ) -> None:
        pass

    @abstractmethod
    def _generate_amplitude_coefficient_name(
        self, graph: StateTransitionGraph, node_id: int
    ) -> str:
        pass


class AbstractAmplitudeGenerator(ABC):
    """Abstract interface for an amplitude model generator."""

    @abstractmethod
    def generate(self, graphs: List[StateTransitionGraph]) -> None:
        pass

    @abstractmethod
    def write_to_file(self, filename: str) -> None:
        pass
