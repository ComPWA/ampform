from qrules import ReactionInfo

from ampform.helicity.naming import _render_float, generate_transition_label


def test_generate_transition_label(reaction: ReactionInfo):
    for transition in reaction.transitions:
        label = generate_transition_label(transition)
        jpsi_spin = _render_float(transition.states[-1].spin_projection)
        gamma_spin = _render_float(transition.states[0].spin_projection)
        assert label == (
            fR"J/\psi(1S)_{{{jpsi_spin}}} \to \gamma_{{{gamma_spin}}}"
            R" \pi^{0}_{0} \pi^{0}_{0}"
        )
