"""Extract :code:`install_requires` section to :file:`requirements.in`."""

import configparser
from os.path import dirname, realpath


def main() -> None:
    script_dir = dirname(realpath(__file__))
    cfg = configparser.ConfigParser()
    cfg.read(f"{script_dir}/../setup.cfg")
    install_requires = cfg.get("options", "install_requires", raw=False)
    install_requires = install_requires[1:]  # remove first line (empty)
    with open(f"{script_dir}/requirements.in", "w") as stream:
        stream.write(install_requires)


if __name__ == "__main__":
    main()
