from typing import List

from knowledge_graph.modules.node import ContentNode, EntityNode

class EntityTypeListIsEmptyError(Exception):
    def __init__(self) -> None:
        self.message = "No entity types have been set."
        super().__init__(self.message)

class EntityDuplicationInOneContentNodeError(Exception):
    def __init__(self, entity_node: EntityNode, content_node: ContentNode) -> None:
        self.message = f"Entity {entity_node.name} already exists in content node {content_node.name} (global id: {content_node.content_global_id})."
        super().__init__(self.message)

class EntitiesAreDangerForMergingError(Exception):
    def __init__(self, entities: List[EntityNode]) -> None:
        self.message = f"Entities {', '.join([entity.name for entity in entities])} are dangerous for merging."
        super().__init__(self.message)