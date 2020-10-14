"""All modules related to the creation of amplitude models.

This module formulates theoretical formalisms from :doc:`Partial Wave Analysis
<pwa:index>`: it provides tools to convert the `.StateTransitionGraph`
solutions that the `.reaction` module found into an `.AmplitudeModel`. The
output `.AmplitudeModel` can then be used by external fitter packages to
generate a data set (toy Monte Carlo) for this specific reaction process, or to
optimize ('fit') its parameters so that they resemble the data set as good as
possible.
"""


__all__ = [
    "canonical_decay",
    "helicity_decay",
    "model",
]


from . import canonical_decay, helicity_decay, model
