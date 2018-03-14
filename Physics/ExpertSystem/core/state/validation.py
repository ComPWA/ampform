class ParticleStateTransitionGraphValidator():
    def __init__(self, graph):
        self.graph = graph
        self.node_conservation_laws = {}

    def assign_conservation_laws_to_all_nodes(self, conservation_laws,
                                              quantum_number_domains):
        for node_id in self.graph.nodes:
            self.assign_conservation_laws_to_node(
                node_id, conservation_laws, quantum_number_domains)

    def assign_conservation_laws_to_node(self, node_id, conservation_laws,
                                         quantum_number_domains):
        if node_id not in self.node_conservation_laws:
            self.node_conservation_laws[node_id] = (
                {'strict': [],
                 'non-strict': []
                 },
                {}
            )
        (cl, qnd) = self.node_conservation_laws[node_id]
        if 'strict' in conservation_laws:
            cl['strict'].extend(conservation_laws['strict'])
        if 'non-strict' in conservation_laws:
            cl['non-strict'].extend(conservation_laws['non-strict'])
        qnd.update(quantum_number_domains)
