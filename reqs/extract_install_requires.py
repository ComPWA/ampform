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

    if "options.extras_require" in cfg:
        extras_require = "\n".join(
            cfg.get("options.extras_require", str(section), raw=False)
            for section in cfg["options.extras_require"]
        )
        extras_require = f"-r requirements.in\n{extras_require}"
        extras_require = extras_require.replace("\n\n", "\n")
        with open(f"{script_dir}/requirements-extras.in", "w") as stream:
            stream.write(extras_require)


if __name__ == "__main__":
    main()
