"""All modules related to the creation of amplitude models.

This module formulates theoretical formalisms from :doc:`Partial Wave Analysis
<pwa:index>`: it provides tools to convert the `.StateTransitionGraph`
solutions that the `.reaction` module found into an `.AmplitudeModel`. The
output `.AmplitudeModel` can then be used by external fitter packages to
generate a data set (toy Monte Carlo) for this specific reaction process, or to
optimize ('fit') its parameters so that they resemble the data set as good as
possible.
"""


from expertsystem.reaction import Result

from .canonical_decay import CanonicalAmplitudeGenerator
from .helicity_decay import HelicityAmplitudeGenerator
from .model import AmplitudeModel


def generate(result: Result) -> AmplitudeModel:
    """Generate an amplitude model from a generated `.Result`.

    The type of amplitude model (`.HelicityAmplitudeGenerator` or
    `.CanonicalAmplitudeGenerator`) is determined from the
    `.Result.formalism_type`.
    """
    formalism_type = result.formalism_type
    if formalism_type is None:
        raise ValueError(f"Result does not have a formalism type:\n{result}")
    if formalism_type == "helicity":
        amplitude_generator = HelicityAmplitudeGenerator()
    elif formalism_type in ["canonical-helicity", "canonical"]:
        amplitude_generator = CanonicalAmplitudeGenerator()
    else:
        raise NotImplementedError(
            f'No amplitude generator for formalism type "{formalism_type}"'
        )
    return amplitude_generator.generate(result)
