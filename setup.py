"""
A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

DATA_FILES = [
    "particle_list.xml",
]

INSTALL_REQUIRES = [
    "numpy",
    "progress",
    "xmltodict",
]


def long_description():
    """Parse long description from readme."""
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setuptools.setup(
    name="expertsystem",
    version="0.1-alpha1",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ComPWA/expertsystem",
    packages=setuptools.find_packages(),
    license="GPLv3 or later",
    python_requires=">=3.3",
    install_requires=INSTALL_REQUIRES,
    package_data={"expertsystem": DATA_FILES},
    include_package_data=True,
)
