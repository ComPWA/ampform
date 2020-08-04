"""The expert system facilitates building an amplitude model.

An amplitude model describes the reaction process you want to study with
partial wave analysis techniques. The responsibility of the expert system is to
give advice on the form of an amplitude model based on the problem set one
defines for a reaction process (initial state, final state, allowed
interactions, intermediate states, etc.). Internally, the system propagates the
quantum numbers through the reaction graph, while satisfying the specified
conservation rules.

Afterwards, the amplitude model of the expert system can be exported. This
amplitude model can then for instance be used to generate a data set (toy Monte
Carlo) for this specific reaction process, or to optimize ('fit') its
parameters so that they resemble the data set as good as possible.
"""


__all__ = [
    "amplitude",
    "data",
    "io",
    "state",
    "topology",
    "ui",
]


import sys

from . import amplitude
from . import data
from . import io
from . import state
from . import topology
from . import ui


def __check_python_version() -> None:
    def print_message_and_exit() -> None:
        major = sys.version_info[0]
        minor = sys.version_info[1]
        patch = sys.version_info[2]
        print(f"You are running python {major}.{minor}.{patch}")
        print("The expertsystem module requires Python 3.7 or higher!")
        sys.exit()

    if sys.version_info.major < 3:
        print_message_and_exit()
    elif sys.version_info.major == 3 and sys.version_info.minor < 7:
        print_message_and_exit()


if __name__ == "__main__":
    __check_python_version()
