"""Build amplitude models with different PWA formalisms.

AmpForm formalizes formalisms from :doc:`Partial Wave Analysis <pwa:index>`. It provides
tools to convert `~qrules.topology.Transition` solutions that the `.qrules`
package found into an `.HelicityModel`. The output `.HelicityModel` can then be used by
external fitter packages to generate a data set (toy Monte Carlo) for this specific
reaction process, or to optimize ('fit') its parameters so that they resemble the data
set as good as possible.
"""

from qrules import ReactionInfo

from ampform.helicity import CanonicalAmplitudeBuilder, HelicityAmplitudeBuilder


def get_builder(reaction: ReactionInfo) -> HelicityAmplitudeBuilder:
    """Get the correct `.HelicityAmplitudeBuilder`.

    For instance, get `.CanonicalAmplitudeBuilder` if the
    `~qrules.transition.ReactionInfo.formalism` is :code:`"canonical-helicity"`.
    """
    formalism = reaction.formalism
    if formalism is None:
        msg = f"{ReactionInfo.__name__} does not have a formalism type:\n{reaction}"
        raise ValueError(msg)
    if formalism == "helicity":
        amplitude_builder = HelicityAmplitudeBuilder(reaction)
    elif formalism in {"canonical-helicity", "canonical"}:
        amplitude_builder = CanonicalAmplitudeBuilder(reaction)
    else:
        msg = f'No amplitude generator for formalism type "{formalism}"'
        raise NotImplementedError(msg)
    return amplitude_builder
