from expertsystem.topology.graph import InteractionNode
from expertsystem.topology.topologybuilder import (
    SimpleStateTransitionTopologyBuilder,
)


class TestSimpleStateBuilder:  # pylint: disable=no-self-use
    def test_two_body_states(self):
        two_body_decay_node = InteractionNode("TwoBodyDecay", 1, 2)

        simple_builder = SimpleStateTransitionTopologyBuilder(
            [two_body_decay_node]
        )

        all_graphs = simple_builder.build_graphs(1, 3)

        assert len(all_graphs) == 1
