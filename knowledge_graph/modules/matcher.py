from py2neo.matching import NodeMatcher, RelationshipMatcher
from base_classes.graph_elements.knowledge_graph import KnowledgeGraph

class GraphNodeMatcher(NodeMatcher):
    def __init__(self, graph: KnowledgeGraph):
        super().__init__(graph)

    def find_node_by_property(self, label, property_name, value):
        return self.match(label).where(f"_.{property_name} = '{value}'").first()

    def find_all_by_label(self, label):
        return list(self.match(label))

class GraphRelationshipMatcher(RelationshipMatcher):
    def __init__(self, graph: KnowledgeGraph):
        super().__init__(graph)

    def find_relationship(self, node1, node2, rel_type=None):
        query = self.match(nodes=(node1, node2), r_type=rel_type)
        return query.first()

    def find_relationship_by_type(self, rel_type):
        return list(self.match(r_type=rel_type))