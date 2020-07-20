"""Abstract interfaces for amplitude model generation."""

from abc import ABC, abstractmethod


class AbstractAmplitudeNameGenerator(ABC):
    """Abstract interface for a parameter name generator."""

    @abstractmethod
    def generate_unique_amplitude_name(self, graph, node_id):
        pass

    @abstractmethod
    def generate_amplitude_coefficient_infos(self, graph):
        pass

    @abstractmethod
    def _generate_amplitude_coefficient_name(self, graph, node_id) -> str:
        pass


class AbstractAmplitudeGenerator(ABC):
    """Abstract interface for an amplitude model generator."""

    @abstractmethod
    def generate(self, graphs):
        pass

    @abstractmethod
    def write_to_file(self, filename):
        pass
