"""Different formalisms for aligning amplitudes in different sub-systems."""
from __future__ import annotations

from abc import ABC, abstractmethod

import sympy as sp
from qrules.transition import ReactionInfo

from ampform.helicity.decay import get_outer_state_ids, group_by_topology
from ampform.helicity.naming import create_amplitude_base, create_spin_projection_symbol


class SpinAlignment(ABC):
    @abstractmethod
    def formulate_amplitude(self, reaction: ReactionInfo) -> sp.Expr:
        ...

    @abstractmethod
    def define_symbols(self, reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]:
        ...


class NoAlignment(SpinAlignment):
    """Sum the amplitudes *without* any spin alignment."""

    def formulate_amplitude(self, reaction: ReactionInfo) -> sp.Expr:
        outer_state_ids = get_outer_state_ids(reaction)
        topology_groups = group_by_topology(reaction.transitions)
        indices = [create_spin_projection_symbol(i) for i in outer_state_ids]
        amplitude_symbols = [
            create_amplitude_base(topology)[indices] for topology in topology_groups
        ]
        return sp.Add(*amplitude_symbols)

    def define_symbols(self, reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]:
        return {}
