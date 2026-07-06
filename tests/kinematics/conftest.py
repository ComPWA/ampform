from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qrules.topology import Topology, create_isobar_topologies

from ampform._qrules import get_qrules_version
from ampform.decay import DecayChain, IsobarNode, Particle, State
from ampform.kinematics.lorentz import FourMomenta, create_four_momentum_symbols

if TYPE_CHECKING:
    import numpy as np


@pytest.fixture(scope="session")
def topology_and_momentum_symbols(
    data_sample: dict[int, np.ndarray],
) -> tuple[Topology, FourMomenta]:
    n = len(data_sample)
    assert n == 4
    topologies = create_isobar_topologies(n)
    topology = topologies[1 if get_qrules_version() < (0, 10) else 0]
    momentum_symbols = create_four_momentum_symbols(topology)
    return topology, momentum_symbols


@pytest.fixture(scope="session")
def double_cascade_chain() -> DecayChain:
    """Dummy four-body decay chain X → (R₁₂ → 1 2) (R₃₄ → 3 4)."""
    dummy_args = dict(spin=0, parity=None, mass=0.0, width=0.0)
    resonance = Particle("R", latex="R", **dummy_args)

    def create_state(index: int) -> State:
        return State(f"f{index}", latex=f"f_{index}", index=index, **dummy_args)

    return DecayChain(
        decay=IsobarNode(
            parent=create_state(0),
            child1=IsobarNode(resonance, create_state(1), create_state(2)),
            child2=IsobarNode(resonance, create_state(3), create_state(4)),
        )
    )
