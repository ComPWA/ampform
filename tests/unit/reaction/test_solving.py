# pylint: disable=no-self-use

from expertsystem.reaction import Result


class TestResult:
    def test_get_intermediate_state_names(
        self, jpsi_to_gamma_pi_pi_helicity_solutions: Result
    ):
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        intermediate_particles = result.get_intermediate_particles()
        assert intermediate_particles.names == {"f(0)(1500)", "f(0)(980)"}
