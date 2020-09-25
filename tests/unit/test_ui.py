from expertsystem.ui import get_intermediate_state_names


def test_get_intermediate_state_names(jpsi_to_gamma_pi_pi_helicity_solutions):
    states = get_intermediate_state_names(
        jpsi_to_gamma_pi_pi_helicity_solutions
    )
    assert states == {"f(0)(1500)", "f(0)(980)"}
