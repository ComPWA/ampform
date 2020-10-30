import pytest

from expertsystem import io
from expertsystem.amplitude.model import AmplitudeModel


@pytest.mark.parametrize("file_extension", ["xml", "yml"])
@pytest.mark.parametrize("formalism", ["helicity", "canonical"])
def test_write_load(
    file_extension: str,
    formalism: str,
    jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
    jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    output_dir,
):
    filename = output_dir + f"test_write_read_{formalism}.{file_extension}"
    exported = None
    if formalism == "helicity":
        exported = jpsi_to_gamma_pi_pi_helicity_amplitude_model
    elif formalism == "canonical":
        exported = jpsi_to_gamma_pi_pi_canonical_amplitude_model
    else:
        raise NotImplementedError(formalism)
    io.write(exported, filename)
    imported = io.load_amplitude_model(filename)
    assert exported.particles == imported.particles
    assert exported.parameters == imported.parameters
    assert exported.kinematics == imported.kinematics
    assert exported.dynamics == imported.dynamics
    assert exported.intensity == imported.intensity
    assert exported == imported
    assert exported is not imported


@pytest.mark.parametrize("formalism", ["helicity", "canonical"])
def test_equivalence(
    formalism: str,
    jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
    jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    output_dir,
):
    exported = None
    if formalism == "helicity":
        exported = jpsi_to_gamma_pi_pi_helicity_amplitude_model
    elif formalism == "canonical":
        exported = jpsi_to_gamma_pi_pi_canonical_amplitude_model
    else:
        raise NotImplementedError(formalism)
    filename = output_dir + f"test_io_cross_check_{formalism}"
    filename_xml = f"{filename}.xml"
    filename_yml = f"{filename}.yml"
    io.write(exported, filename_xml)
    io.write(exported, filename_yml)
    imported_xml = io.load_amplitude_model(filename_xml)
    imported_yml = io.load_amplitude_model(filename_yml)
    assert imported_xml.particles == imported_yml.particles
    assert imported_xml.parameters == imported_yml.parameters
    assert imported_xml.kinematics == imported_yml.kinematics
    assert imported_xml.dynamics == imported_yml.dynamics
    assert imported_xml.intensity == imported_yml.intensity
    assert imported_xml == imported_yml
    assert imported_xml is not imported_yml
