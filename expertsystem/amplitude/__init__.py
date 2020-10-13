"""All modules related to the creation of amplitude models.

This module concerts the `.StateTransitionGraph` solutions that the `.reaction`
module found for a specific reaction process into an `.AmplitudeModel`. As
such, this module is the place that formulates theoretical formalisms from
Partial Wave Analysis.

The output `.AmplitudeModel` can then be used by external fitter packages to
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
