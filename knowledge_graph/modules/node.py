from typing import Optional, List, Union
from types import NoneType

from base_classes.graph_elements.node import GraphNode

# class DocumentNode(GraphNode):
#     document_id: str
#     title: str
#     version: str
#     last_update: str
#     attached_to: list
#     document_directory: str
#     def __init__(self, **properties):
#         document_defaults = {
#             'document_id': str,
#             'title': str,
#             'version': str,
#             'last_update': str,
#             'attached_to': list,
#             'document_directory': str,
#         }
#         super().__init__('__Document__', defaults=document_defaults, **properties)
#     @property
#     def id(self):
#         return self['document_id']
#     @property
#     def title(self):
#         return self['title']
#     @property
#     def version(self):
#         return self['version']
#     @property
#     def last_update(self):
#         return self['last_update']
#     @property
#     def attached_to(self):
#         return self['attached_to']
#     @property
#     def document_directory(self):
#         return self['document_directory']
    
#     @id.setter
#     def id(self, value):
#         self['document_id'] = value
#     @title.setter
#     def title(self, value):
#         self['title'] = value
#     @version.setter
#     def version(self, value):
#         self['version'] = value
#     @last_update.setter
#     def last_update(self, value):
#         self['last_update'] = value
#     @attached_to.setter
#     def attached_to(self, value):
#         self['attached_to'] = value
#     @document_directory.setter
#     def document_directory(self, value):
#         self['document_directory'] = value

# class HeadingNode(GraphNode):
#     source_document_id: str
#     title: str
#     heading_global_id: str
#     def __init__(self, **properties):
#         heading_defaults = {
#             'source_document_id': str,
#             'title': str,
#             'heading_global_id': str,
#         }
#         super().__init__('__Heading__', defaults=heading_defaults, **properties)

#     @property
#     def title(self):
#         return self['title']
#     @property
#     def source_document_id(self):
#         return self['source_document_id']
#     @property
#     def heading_global_id(self):
#         return self['heading_global_id']
    
#     @title.setter
#     def title(self, value):
#         self['title'] = value
#     @source_document_id.setter
#     def source_document_id(self, value):
#         self['source_document_id'] = value

class ContentNode(GraphNode):
    source_document_id: str
    name: str
    content: str
    content_global_id: str
    content_emb: list
    def __init__(self, **properties):
        content_defaults = {
            'source_document_id': str,
            'name': str,
            'content': str,
            'content_global_id': str,
            'content_emb': list,
        }
        super().__init__('__Content__', defaults=content_defaults, **properties)
    @property
    def content(self):
        return self['content']
    @property
    def source_document_id(self):
        return self['source_document_id']
    @property
    def content_global_id(self):
        return self['content_global_id']
    @property
    def name(self):
        return self['name']
    @property
    def content_emb(self):
        return self['content_emb']
    
    @content.setter
    def content(self, value):
        self['content'] = value
    @source_document_id.setter
    def source_document_id(self, value):
        self['source_document_id'] = value
    @content_global_id.setter
    def content_global_id(self, value):
        self['content_global_id'] = value

# class ChunkNode(GraphNode):
#     def __init__(self, **properties):
#         chunk_defaults = {
#             'name': str,
#             'content': str,
#             'vector_emb': List[float],
#         }
#         super().__init__('__Chunk__', defaults=chunk_defaults, **properties)

# class CommunityNode(GraphNode):
#     community_id: str
#     level: int
#     rank: int
#     community_id: str
#     summary_content: str
#     vector_emb: list
#     def __init__(self, **properties):
#         community_defaults = {
#             'community_id': str,
#             'level': int,
#         }
#         super().__init__('__Community__', defaults=community_defaults, **properties)

#     @property
#     def community_id(self):
#         return self['community_id']
#     @property
#     def level(self):
#         return self['level']
#     @property
#     def rank(self):
#         return self['rank']
#     @property
#     def summary_content(self):
#         return self['summary_content']
#     @property
#     def vector_emb(self):
#         return self['vector_emb']
    
#     @vector_emb.setter
#     def vector_emb(self, value):
#         self['vector_emb'] = value

class EntityNode(GraphNode):
    name: str
    definition: str
    description: str
    types: list
    source_content_ids: list
    name_emb: list
    definition_emb: list
    description_emb: list
    def __init__(self, **properties):
        entity_defaults = {
            'name': str,
            'types': list,
            'source_content_ids': list,
        }
        super().__init__('__Entity__', defaults=entity_defaults, **properties)
    @property
    def name(self) -> str:
        return self['name']
    @property
    def types(self) -> list:
        return self['types']
    @property
    def definition(self) -> str:
        return self['definition']
    @property
    def description(self) -> str:
        return self['description']
    @property
    def source_content_ids(self) -> list:
        return self['source_content_ids']
    @property
    def name_emb(self) -> list:
        return self['name_emb']
    @property
    def definition_emb(self) -> list:
        return self['definition_emb']
    @property
    def description_emb(self) -> list:
        return self['description_emb']
    
    @types.setter
    def types(self, value):
        self['types'] = value
    @definition.setter
    def definition(self, value):
        self['definition'] = value
    @description.setter
    def description(self, value):
        self['description'] = value
    @source_content_ids.setter
    def source_content_ids(self, value):
        self['source_content_ids'] = value
    @name_emb.setter
    def name_emb(self, value):
        self['name_emb'] = value
    @definition_emb.setter
    def definition_emb(self, value):
        self['definition_emb'] = value
    @description_emb.setter
    def description_emb(self, value):
        self['description_emb'] = value

