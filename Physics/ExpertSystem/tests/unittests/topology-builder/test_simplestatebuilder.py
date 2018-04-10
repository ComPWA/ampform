from expertsystem.topology.graph import InteractionNode
from expertsystem.topology.topologybuilder import (
    SimpleStateTransitionTopologyBuilder)


class TestSimpleStateBuilder(object):
    def test_two_body_states(self):
        TwoBodyDecayNode = InteractionNode("TwoBodyDecay", 1, 2)

        SimpleBuilder = SimpleStateTransitionTopologyBuilder(
            [TwoBodyDecayNode])

        all_graphs = SimpleBuilder.build_graphs(1, 3)

        assert len(all_graphs) is 1
