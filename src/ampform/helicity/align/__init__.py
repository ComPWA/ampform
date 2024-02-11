"""Different formalisms for aligning amplitudes in different sub-systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import sympy as sp

from ampform.helicity.decay import get_outer_state_ids, group_by_topology
from ampform.helicity.naming import create_amplitude_base, create_spin_projection_symbol

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo


class SpinAlignment(ABC):
    @abstractmethod
    def formulate_amplitude(self, reaction: ReactionInfo) -> sp.Expr: ...

    @abstractmethod
    def define_symbols(self, reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]: ...


class NoAlignment(SpinAlignment):
    """Sum the amplitudes *without* any spin alignment."""

    @staticmethod
    def formulate_amplitude(reaction: ReactionInfo) -> sp.Expr:
        outer_state_ids = get_outer_state_ids(reaction)
        topology_groups = group_by_topology(reaction.transitions)
        indices = [create_spin_projection_symbol(i) for i in outer_state_ids]
        amplitude_symbols = [
            create_amplitude_base(topology)[indices] for topology in topology_groups
        ]
        return sp.Add(*amplitude_symbols)

    @staticmethod
    def define_symbols(reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]:
        return {}
