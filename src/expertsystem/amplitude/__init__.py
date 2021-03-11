"""All modules related to the creation of amplitude models.

This module formulates theoretical formalisms from :doc:`Partial Wave Analysis
<pwa:index>`. It provides tools to convert the `.StateTransitionGraph`
solutions that the `.reaction` module found into an `.HelicityModel`. The
output `.HelicityModel` can then be used by external fitter packages to
generate a data set (toy Monte Carlo) for this specific reaction process, or to
optimize ('fit') its parameters so that they resemble the data set as good as
possible.
"""

from expertsystem.reaction import Result

from .helicity import CanonicalAmplitudeBuilder, HelicityAmplitudeBuilder


def get_builder(result: Result) -> HelicityAmplitudeBuilder:
    """Get the correct `.HelicityAmplitudeBuilder` for a `.Result`.

    For instance, get `.CanonicalAmplitudeBuilder` if the
    `~.Result.formalism_type` is :code:`"canonical-helicity"`.
    """
    formalism_type = result.formalism_type
    if formalism_type is None:
        raise ValueError(f"Result does not have a formalism type:\n{result}")
    if formalism_type == "helicity":
        amplitude_builder = HelicityAmplitudeBuilder(result)
    elif formalism_type in ["canonical-helicity", "canonical"]:
        amplitude_builder = CanonicalAmplitudeBuilder(result)
    else:
        raise NotImplementedError(
            f'No amplitude generator for formalism type "{formalism_type}"'
        )
    return amplitude_builder
