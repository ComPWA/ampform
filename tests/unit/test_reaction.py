from expertsystem.reaction import StateTransitionManager


class TestStateTransitionManager:
    @staticmethod
    def test_allowed_intermediate_particles():
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["p", "p~", "eta"],
        )
        particle_name = "N(753)"
        try:
            stm.set_allowed_intermediate_particles([particle_name])
        except LookupError as exception:
            assert particle_name in exception.args[0]
