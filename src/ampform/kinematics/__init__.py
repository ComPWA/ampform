"""Classes and functions for relativistic four-momentum kinematics.

.. autolink-preface::

    import sympy as sp
    from ampform.kinematics import create_four_momentum_symbols
"""

from __future__ import annotations

import itertools
from collections import abc
from functools import singledispatch
from typing import TYPE_CHECKING

import attrs
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition

from ampform._qrules import get_qrules_version
from ampform.helicity.decay import assert_isobar_topology
from ampform.kinematics.angles import compute_helicity_angles
from ampform.kinematics.lorentz import (
    compute_invariant_masses,
    create_four_momentum_symbols,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy as sp


class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.create_expressions` method forms the bridge between four-momentum data for the
    decay you are studying and the kinematic variables that are in the `.HelicityModel`.
    These are invariant mass (see :func:`.get_invariant_mass_symbol`) and the
    :math:`\theta` and :math:`\phi` helicity angles (see
    :func:`.get_helicity_angle_symbols`).
    """

    def __init__(
        self,
        transitions: ReactionInfo | Iterable[Topology | StateTransition],
    ) -> None:
        self.__topologies = _extract_topologies(transitions)
        for topology in self.__topologies:
            assert_isobar_topology(topology)

    def register_transition(self, transition: StateTransition) -> None:
        topology = _get_topology(transition)
        self.register_topology(topology)

    def register_topology(self, topology: Topology) -> None:
        assert_isobar_topology(topology)
        if self.__topologies:
            existing = next(iter(self.__topologies))
            if topology.incoming_edge_ids != existing.incoming_edge_ids:
                msg = "Initial state ID mismatch those of existing topologies"
                raise ValueError(msg)
            if topology.outgoing_edge_ids != existing.outgoing_edge_ids:
                msg = "Final state IDs mismatch those of existing topologies"
                raise ValueError(msg)
        self.__topologies.add(topology)

    @property
    def registered_topologies(self) -> frozenset[Topology]:
        return frozenset(self.__topologies)

    def permutate_registered_topologies(self) -> None:
        """Register outgoing edge permutations of all `registered_topologies`.

        See :ref:`usage/amplitude:Extend kinematic variables`.
        """
        for topology in set(self.__topologies):
            final_state_ids = topology.outgoing_edge_ids
            for permutation in itertools.permutations(final_state_ids):
                id_mapping = dict(zip(topology.outgoing_edge_ids, permutation))
                permuted_topology = attrs.evolve(
                    topology,
                    edges={
                        id_mapping.get(i, i): edge for i, edge in topology.edges.items()
                    },
                )
                self.__topologies.add(permuted_topology)

    def create_expressions(self) -> dict[sp.Symbol, sp.Expr]:
        output = {}
        for topology in self.__topologies:
            momenta = create_four_momentum_symbols(topology)
            output.update(compute_helicity_angles(momenta, topology))
            output.update(compute_invariant_masses(momenta, topology))
        return output


@singledispatch
def _extract_topologies(
    obj: ReactionInfo | Iterable[Topology | StateTransition],
) -> set[Topology]:
    msg = f"Cannot extract topologies from a {type(obj).__name__}"
    raise TypeError(msg)


@_extract_topologies.register(ReactionInfo)
def _(transitions: ReactionInfo) -> set[Topology]:
    return _extract_topologies(transitions.transitions)


@_extract_topologies.register(abc.Iterable)
def _(transitions: abc.Iterable) -> set[Topology]:
    return {_get_topology(t) for t in transitions}


@singledispatch
def _get_topology(obj) -> Topology:
    msg = f"Cannot create a {Topology.__name__} from a {type(obj).__name__}"
    raise TypeError(msg)


@_get_topology.register(Topology)
def _(obj: Topology) -> Topology:
    return obj


def __get_state_transition(obj: StateTransition) -> Topology:
    return obj.topology


if get_qrules_version() < (0, 10):
    _get_topology.register(StateTransition)(__get_state_transition)
else:
    from qrules.topology import FrozenTransition

    _get_topology.register(FrozenTransition)(__get_state_transition)
