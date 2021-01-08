import pytest

from expertsystem import io
from expertsystem.amplitude.model import AmplitudeModel


@pytest.mark.parametrize("file_extension", ["json", "yml"])
@pytest.mark.parametrize("formalism", ["helicity", "canonical"])
def test_serialization(
    file_extension: str,
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
    asdict = io.asdict(exported)
    imported = io.fromdict(asdict)
    assert isinstance(imported, AmplitudeModel)
    assert exported.particles == imported.particles
    assert exported.parameters == imported.parameters
    assert exported.kinematics == imported.kinematics
    assert exported.dynamics == imported.dynamics
    assert exported.intensity == imported.intensity
    assert exported == imported
    assert exported is not imported
    filename = output_dir + f"test_write_read_{formalism}.{file_extension}"
    io.write(exported, filename)
