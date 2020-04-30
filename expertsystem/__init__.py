__all__ = ["amplitude", "state", "topology", "ui"]
from . import amplitude
from . import state
from . import topology
from . import ui


if __name__ == "__main__":
    import sys

    def print_message_and_exit():
        print(
            "You are running python "
            + str(sys.version_info[0])
            + "."
            + str(sys.version_info[1])
            + "."
            + str(sys.version_info[2])
        )
        print("The ComPWA expertsystem required python 3.3 or higher!")
        sys.exit()

    if sys.version_info.major < 3:
        print_message_and_exit()
    elif sys.version_info.major == 3 and sys.version_info.minor < 3:
        print_message_and_exit()
