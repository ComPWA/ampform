"""
A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

DATA_FILES = [
    ("share", ["particle_list.xml", "particle_list.yml"]),
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
    version="0.0-alpha1",
    author="The ComPWA team",
    maintainer_email="compwa-admin@ep1.rub.de",
    url="https://github.com/ComPWA/expertsystem",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="GPLv3 or later",
    python_requires=">=3.6",
    tests_require=["pytest"],
    install_requires=INSTALL_REQUIRES,
    data_files=DATA_FILES,
    package_data=dict(DATA_FILES),
    include_package_data=True,
)
