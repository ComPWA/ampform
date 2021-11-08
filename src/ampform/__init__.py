"""Build amplitude models with different PWA formalisms.

AmpForm formalizes formalisms from :doc:`Partial Wave Analysis <pwa:index>`. It
provides tools to convert `~qrules.transition.StateTransition` solutions that
the `.qrules` package found into an `.HelicityModel`. The output
`.HelicityModel` can then be used by external fitter packages to generate a
data set (toy Monte Carlo) for this specific reaction process, or to optimize
('fit') its parameters so that they resemble the data set as good as possible.
"""

from qrules import ReactionInfo

from .helicity import CanonicalAmplitudeBuilder, HelicityAmplitudeBuilder


def get_builder(reaction: ReactionInfo) -> HelicityAmplitudeBuilder:
    """Get the correct `.HelicityAmplitudeBuilder`.

    For instance, get `.CanonicalAmplitudeBuilder` if the
    `~qrules.transition.ReactionInfo.formalism` is
    :code:`"canonical-helicity"`.
    """
    formalism = reaction.formalism
    if formalism is None:
        raise ValueError(
            f"{ReactionInfo.__name__} does not have a formalism"
            f" type:\n{reaction}"
        )
    if formalism == "helicity":
        amplitude_builder = HelicityAmplitudeBuilder(reaction)
    elif formalism in ["canonical-helicity", "canonical"]:
        amplitude_builder = CanonicalAmplitudeBuilder(reaction)
    else:
        raise NotImplementedError(
            f'No amplitude generator for formalism type "{formalism}"'
        )
    return amplitude_builder
