import qrules

from ampform import get_builder


def test_generate_model():
    result = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types=["strong", "EM"],
        formalism="helicity",
    )
    model_builder = get_builder(result)
    original_model = model_builder.formulate()
    assert original_model is not None
