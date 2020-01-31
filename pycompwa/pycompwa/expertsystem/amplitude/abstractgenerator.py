from abc import ABC, abstractmethod


class AbstractAmplitudeNameGenerator(ABC):
    @abstractmethod
    def generate_unique_amplitude_name(self, graph, node_id):
        pass

    @abstractmethod
    def generate_amplitude_coefficient_infos(self, graph):
        pass

    @abstractmethod
    def _generate_amplitude_coefficient_names(self, graph, node_id):
        pass


class AbstractAmplitudeGenerator(ABC):
    @abstractmethod
    def generate(self, graphs):
        pass

    @abstractmethod
    def write_to_file(self, filename):
        pass
