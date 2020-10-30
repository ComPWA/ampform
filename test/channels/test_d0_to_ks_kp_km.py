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
    assert len(result.solutions) == 5
    assert result.get_intermediate_particles().names == {
        "a(0)(980)+",
        "a(0)(980)-",
        "a(0)(980)0",
        "a(2)(1320)-",
        "phi(1020)",
    }
    model = es.generate_amplitudes(result)
    assert len(model.parameters) == 12
    es.io.write(model, "D0_to_K0sKpKm.xml")
    es.io.write(model, "D0_to_K0sKpKm.yml")


if __name__ == "__main__":
    test_script()
