from __future__ import annotations

import numpy as np
import pytest
from qrules.topology import Topology, create_isobar_topologies

from ampform.kinematics.lorentz import FourMomenta, create_four_momentum_symbols


@pytest.fixture(scope="session")
def topology_and_momentum_symbols(
    data_sample: dict[int, np.ndarray]
) -> tuple[Topology, FourMomenta]:
    n = len(data_sample)
    assert n == 4
    topologies = create_isobar_topologies(n)
    topology = topologies[1]
    momentum_symbols = create_four_momentum_symbols(topology)
    return topology, momentum_symbols
