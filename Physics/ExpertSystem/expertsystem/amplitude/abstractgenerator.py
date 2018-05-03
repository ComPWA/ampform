from abc import ABC, abstractmethod


class AbstractAmplitudeNameGenerator(ABC):
    @abstractmethod
    def generate(self, graph, node_id):
        pass


class AbstractAmplitudeGenerator(ABC):
    @abstractmethod
    def generate(self, graphs):
        pass

    @abstractmethod
    def write_to_file(self, filename):
        pass

