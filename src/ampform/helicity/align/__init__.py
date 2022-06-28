"""Different formalisms for aligning amplitudes in different sub-systems."""
import sympy as sp
from qrules.transition import ReactionInfo

from ampform.helicity.decay import get_outer_state_ids, group_by_topology
from ampform.helicity.naming import create_amplitude_base, create_spin_projection_symbol


def sum_amplitudes(reaction: ReactionInfo) -> sp.Add:
    """Sum the amplitudes *without* any spin alignment."""
    outer_state_ids = get_outer_state_ids(reaction)
    topology_groups = group_by_topology(reaction.transitions)
    indices = [create_spin_projection_symbol(i) for i in sorted(outer_state_ids)]
    amplitudes = [
        create_amplitude_base(topology)[indices] for topology in topology_groups
    ]
    return sp.Add(*amplitudes)
