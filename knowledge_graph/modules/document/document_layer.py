from typing import Dict, List
from py2neo import Subgraph

from base_classes.graph_elements.knowledge_graph import KnowledgeGraph
from base_classes.embedding_model import AbstractEmbeddingModel
from base_classes.graph_elements.layer import KnowledgeGraphLayer
from knowledge_graph.modules.node import DocumentNode, ContentNode, HeadingNode
from knowledge_graph.modules.relationship import IsAttachedTo
from knowledge_graph.modules.document.document_handler import DocumentHandler
from exception.node_exception import DocumentNodeHaveAlreadyBeenAddedError, NodeNotFoundError

class DocumentLayer(KnowledgeGraphLayer):
    def __init__(self, graph: KnowledgeGraph) -> None:
        super().__init__(graph)

        document_citation_subgraph_cypher = """
        MATCH (n:__Document__)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """

        document_citation_subgraph = self.graph.run(document_citation_subgraph_cypher)
        nodes = set()
        relationships = set()
        for record in document_citation_subgraph:
            nodes.add(record['n'])
            if record['m']:
                nodes.add(record['m'])
            if record['r']:
                relationships.add(record['r'])
        
        self._document_subgraph = Subgraph(nodes, relationships)

    # Graph modification methods
    def add_document_node_to_graph(self, document_node: DocumentNode) -> DocumentNode:
        list_of_document_nodes = self.graph.get_document_nodes()
        list_of_document_nodes_ids = [document_node['n']['document_id'] for document_node in list_of_document_nodes]

        if document_node.id in list_of_document_nodes_ids:
            raise DocumentNodeHaveAlreadyBeenAddedError(document_node)
        
        self.graph.create(document_node)
        
        if document_node.attached_to:
            for attached_document_id in document_node.attached_to:
                attached_document_node = self.graph.get_document_node_by_id(attached_document_id)
                if not attached_document_node:
                    raise NodeNotFoundError(attached_document_id)

                self.graph.create(IsAttachedTo(document_node, attached_document_node))
        
        return document_node
    
    # Operations on specific document nodes
    def create_document_tree(self, document_node: DocumentNode) -> None:
        document_handler = DocumentHandler(document_node, self.graph)
        document_handler.load_embedding_model(self.emb_model)
        document_handler.create_document_tree()
    
    # Getter methods
    def get_document_node_by_id(self, document_id: str) -> DocumentNode:
        _document_node = self.graph.get_document_node_by_id(document_id)
        document_node = DocumentNode(**_document_node) if _document_node else None
        return document_node
    
    def get_heading_nodes_by_document(self, document_node: DocumentNode) -> List[Dict[str, ContentNode]]:
        raw_heading_nodes = self.graph.get_heading_nodes_by_document(document_node)
        heading_nodes = [HeadingNode(**raw_heading_node) for raw_heading_node in raw_heading_nodes]
        return heading_nodes
    
    def get_content_node_by_id(self, content_id: str) -> ContentNode:
        _content_node = self.graph.get_content_node_by_id(content_id)
        content_node = ContentNode(**_content_node) if _content_node else None
        return content_node
    
    def get_content_nodes_by_document(self, document_node: DocumentNode) -> List[Dict[str, ContentNode]]:
        _content_nodes = self.graph.get_content_nodes_by_document(document_node)
        content_nodes = [ContentNode(**content_node) for content_node in _content_nodes]
        return content_nodes
    
    def get_content_nodes_by_heading(self, heading_node: HeadingNode) -> List[Dict[str, ContentNode]]:
        _content_nodes = self.graph.get_content_nodes_by_heading(heading_node)
        content_nodes = [ContentNode(**content_node) for content_node in _content_nodes]
        return content_nodes
    
    def get_content_nodes(self) -> List[Dict[str, ContentNode]]:
        _content_nodes = self.graph.get_content_nodes()
        content_nodes = [ContentNode(**content_node) for content_node in _content_nodes]
        return content_nodes
    
    