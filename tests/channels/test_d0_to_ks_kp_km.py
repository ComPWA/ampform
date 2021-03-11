import expertsystem as es


def test_script():
    result = es.generate_transitions(
        initial_state="D0",
        final_state=["K~0", "K+", "K-"],
        allowed_intermediate_particles=[
            "a(0)(980)",
            "a(2)(1320)-",
            "phi(1020)",
        ],
        number_of_threads=1,
    )
    assert len(result.transitions) == 5
    assert result.get_intermediate_particles().names == {
        "a(0)(980)+",
        "a(0)(980)-",
        "a(0)(980)0",
        "a(2)(1320)-",
        "phi(1020)",
    }
    model_builder = es.amplitude.get_builder(result)
    model = model_builder.generate()
    assert len(model.parameters) == 5
