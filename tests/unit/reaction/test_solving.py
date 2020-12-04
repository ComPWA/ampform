# pylint: disable=no-self-use

from expertsystem.particle import ParticleCollection
from expertsystem.reaction import Result


class TestResult:
    def test_get_intermediate_state_names(
        self, jpsi_to_gamma_pi_pi_helicity_solutions: Result
    ):
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        intermediate_particles = result.get_intermediate_particles()
        assert intermediate_particles.names == {"f(0)(1500)", "f(0)(980)"}

    def test_get_particle_graphs(
        self,
        particle_database: ParticleCollection,
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
    ):
        pdg = particle_database
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        particle_graphs = result.get_particle_graphs()
        assert len(particle_graphs) == 2
        assert particle_graphs[0].get_edge_props(1) == pdg["f(0)(980)"]
        assert particle_graphs[1].get_edge_props(1) == pdg["f(0)(1500)"]
        assert len(particle_graphs[0].edges) == 5
        for edge_id in (0, 2, 3, 4):
            assert particle_graphs[0].get_edge_props(
                edge_id
            ) is particle_graphs[1].get_edge_props(edge_id)

    def test_collapse_graphs(
        self,
        jpsi_to_gamma_pi_pi_helicity_solutions: Result,
        particle_database: ParticleCollection,
    ):
        pdg = particle_database
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        particle_graphs = result.get_particle_graphs()
        assert len(particle_graphs) == 2
        collapsed_graphs = result.collapse_graphs()
        assert len(collapsed_graphs) == 1
        graph = collapsed_graphs[0]
        edge_id = graph.get_intermediate_state_edge_ids()[0]
        f_resonances = pdg.filter(
            lambda p: p.name in ["f(0)(980)", "f(0)(1500)"]
        )
        intermediate_states = graph.get_edge_props(edge_id)
        assert isinstance(intermediate_states, ParticleCollection)
        assert intermediate_states == f_resonances
