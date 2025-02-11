from datetime import datetime
from py2neo import Relationship

from knowledge_graph.modules.node import *
from base_classes.graph_elements.relationship import GraphRelationship

# Hierarchical relationships
# class IsDirectSubHeadingOfRelationship(GraphRelationship):
#     def __init__(self, start_node: HeadingNode, end_node: HeadingNode, **properties):
#         is_sub_heading_of_defaults = {}
#         super().__init__(start_node, end_node, '__IS_DIRECT_SUB_HEADING_OF__', defaults=is_sub_heading_of_defaults, **properties)

# class IsDirectSubCommunityOfRelationship(GraphRelationship):
#     def __init__(self, start_node: CommunityNode, end_node: CommunityNode, **properties):
#         is_sub_community_of_defaults = {}
#         super().__init__(start_node, end_node, '__IS_DIRECT_SUB_COMMUNITY_OF__', defaults=is_sub_community_of_defaults, **properties)

# Document citation relationships
# class IsAttachedTo(GraphRelationship):
#     def __init__(self, start_node: DocumentNode, end_node: DocumentNode, **properties):
#         is_attached_to_defaults = {}
#         super().__init__(start_node, end_node, '__IS_ATTACHED_TO__', defaults=is_attached_to_defaults, **properties)

# class IsSubjectedTo(GraphRelationship):
#     def __init__(self, start_node: DocumentNode, end_node: DocumentNode, **properties):
#         is_subjected_to_defaults = {}
#         super().__init__(start_node, end_node, '__IS_SUBJECTED_TO__', defaults=is_subjected_to_defaults, **properties)

# Containment relationships
# class HasDirectContentRelationship(GraphRelationship):
#     def __init__(self, start_node: HeadingNode, end_node: ContentNode, **properties):
#         has_content_defaults = {}
#         super().__init__(start_node, end_node, '__HAS_DIRECT_CONTENT__', defaults=has_content_defaults, **properties)

# class HasDirectChunkOfRelationship(GraphRelationship):
#     def __init__(self, start_node: ContentNode, end_node: ChunkNode, **properties):
#         has_chunk_defaults = {}
#         super().__init__(start_node, end_node, '__HAS_DIRECT_CHUNK__', defaults=has_chunk_defaults, **properties)

class HasEntityRelationship(GraphRelationship):
    def __init__(self, start_node: ContentNode, end_node: EntityNode, **properties):
        has_entity_defaults = {}
        super().__init__(start_node, end_node, '__HAS_ENTITY__', defaults=has_entity_defaults, **properties)

# class IsAHeadingInDocumentRelationship(GraphRelationship):
#     def __init__(self, start_node: HeadingNode, end_node: DocumentNode, **properties):
#         is_a_heading_in_document_defaults = {}
#         super().__init__(start_node, end_node, '__IS_A_HEADING_IN_DOCUMENT__', defaults=is_a_heading_in_document_defaults, **properties)

# class IsAContentInDocumentRelationship(GraphRelationship):
#     def __init__(self, start_node: ContentNode, end_node: DocumentNode, **properties):
#         is_a_content_in_document_defaults = {}
#         super().__init__(start_node, end_node, '__IS_A_CONTENT_IN_DOCUMENT__', defaults=is_a_content_in_document_defaults, **properties)

# class InCommunityRelationship(GraphRelationship):
#     def __init__(self, start_node: EntityNode, end_node: CommunityNode, **properties):
#         is_a_content_in_document_defaults = {}
#         super().__init__(start_node, end_node, '__IN_COMMUNITY__', defaults=is_a_content_in_document_defaults, **properties)

# Order relationships
# class IsNextHeadingRelationship(GraphRelationship):
#     def __init__(self, start_node: HeadingNode, end_node: HeadingNode, **properties):
#         is_next_heading_defaults = {}
#         super().__init__(start_node, end_node, '__IS_NEXT_HEADING_OF__', defaults=is_next_heading_defaults, **properties)

# class IsNextChunkRelationship(GraphRelationship):
#     def __init__(self, start_node: ChunkNode, end_node: ChunkNode, **properties):
#         is_next_chunk_of_defaults = {}
#         super().__init__(start_node, end_node, '__IS_NEXT_CHUNK_OF__', defaults=is_next_chunk_of_defaults, **properties)

# class IsNextContentRelationship(GraphRelationship):
#     def __init__(self, start_node: ContentNode, end_node: ContentNode, **properties):
#         is_next_chunk_of_defaults = {}
#         super().__init__(start_node, end_node, '__IS_NEXT_CONTENT_OF__', defaults=is_next_chunk_of_defaults, **properties)

# Entity relationships
class BaseEntityRelationship(GraphRelationship):
    def __init__(self, start_node: EntityNode, end_node: EntityNode, **properties):
        base_entity_defaults = {
            'description': str,
            'strength': int,
            'name': str
        }
        super().__init__(start_node, end_node, '>>>', defaults=base_entity_defaults, **properties)