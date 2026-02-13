from __future__ import annotations

import io
import pickle  # noqa: S403
from typing import TYPE_CHECKING

import ampform

if TYPE_CHECKING:
    from qrules import ReactionInfo

    from ampform.helicity import HelicityModel


def test_model_pickling(reaction: ReactionInfo):
    """See https://github.com/ComPWA/ampform/issues/471."""
    builder = ampform.get_builder(reaction)
    model = builder.formulate()
    stream = io.BytesIO()
    pickle.dump(model, stream)
    stream.seek(0)
    loaded_model: HelicityModel = pickle.load(stream)  # noqa: S301
    assert loaded_model.kinematic_variables == model.kinematic_variables
