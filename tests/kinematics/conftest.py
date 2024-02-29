from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qrules.topology import Topology, create_isobar_topologies

from ampform._qrules import get_qrules_version
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
