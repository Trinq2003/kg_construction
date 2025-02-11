from typing import Optional, List
import pandas as pd
import os

from base_classes.graph_elements.knowledge_graph import KnowledgeGraph
from base_classes.embedding_model import AbstractEmbeddingModel
from knowledge_graph.modules.node import HeadingNode, ContentNode, DocumentNode
from knowledge_graph.modules.relationship import IsDirectSubHeadingOfRelationship, HasDirectContentRelationship, IsNextHeadingRelationship, IsNextContentRelationship, IsAHeadingInDocumentRelationship, IsAContentInDocumentRelationship
from exception.document_exception import DocumentDirectoryNotFoundError
from exception.embedding_exception import EmbeddingModelNotFoundError

class DocumentHandler:
    heading_id: int = 0
    content_id: int = 0
    emb_model: AbstractEmbeddingModel = None
    def __init__(self, document_node: DocumentNode, knowledge_graph: KnowledgeGraph) -> None:
        self._parent_document_node = document_node
        
        self.doc_dir = document_node.document_directory
        if not os.path.exists(self.doc_dir):
            raise DocumentDirectoryNotFoundError(document_node)
        
        self.document_name = os.path.basename(self.doc_dir)
        json2csv_path = "json2csv_" + self.document_name + ".csv"
        self.csv_path = os.path.join(self.doc_dir, json2csv_path)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file '{self.csv_path}' does not exist.")
        
        self.df = pd.read_csv(self.csv_path)

        self.graph = knowledge_graph
        self.headings = []
        self.subheading_relationships = []
        self.has_content_relationships = {}
        self.headings_by_level = {}

        self.nodes = {}
    def extract_data_from_csv(self):
        """
        Extract headings, relationships, content relations, and headings by level from the CSV file template (produced by data_processing.ipynb)
        """
        headings = set()
        for index, row in self.df.iterrows():
            path = row['Metadata'].split("-->")
            content = row['Content'].strip()

            for i in range(len(path)):
                heading = path[i].strip()
                headings.add(heading)
                level = i 
                parent = path[i-1].strip() if i > 0 else None
                if (parent, level) not in self.headings_by_level:
                    self.headings_by_level[(parent, level)] = []
                self.headings_by_level[(parent, level)].append(heading)

                if i > 0:
                    self.subheading_relationships.append((path[i-1].strip(), heading))

            heading = path[-1].strip()
            if heading not in self.has_content_relationships:
                self.has_content_relationships[heading] = []
            self.has_content_relationships[heading].append(content)

        self.headings = list(headings)

        return self.headings, self.subheading_relationships, self.has_content_relationships, self.headings_by_level
    
    def create_document_tree(self):
        self.extract_data_from_csv()
        self._sort_headings()
        self._add_heading_nodes_and_heading_relationships_to_graph()
        self._add_content_nodes_and_content_relationships_to_graph()

    def load_embedding_model(self, emb_model: AbstractEmbeddingModel):
        self.emb_model = emb_model

    def _sort_headings(self, headings_list:Optional[List] = None):
        def extract_indices(heading):
            parts = heading.split()
            indices = parts[0].split('.')
            return [int(part) if part.isdigit() else float('inf') for part in indices]
        
        if headings_list is None:
            headings_list = list(set(self.headings))
            self.headings = sorted(headings_list, key=extract_indices, reverse=True)

            return self.headings
        else:
            headings_list = list(set(headings_list))
            return sorted(headings_list, key=extract_indices, reverse=True)
    
    def _add_heading_nodes_and_heading_relationships_to_graph(self):
        for heading in self.headings:
            heading_node = HeadingNode(title=heading, source_document_id=self._parent_document_node.id, heading_global_id=self._generate_heading_id())
            self.graph.create(heading_node)
            self.nodes[heading] = heading_node

            is_a_heading_in_document_relationship = IsAHeadingInDocumentRelationship(start_node=heading_node, end_node=self._parent_document_node)
            self.graph.create(is_a_heading_in_document_relationship)

        for parent, child in self.subheading_relationships:
            subheading_relationship = IsDirectSubHeadingOfRelationship(start_node=self.nodes[child], end_node=self.nodes[parent])
            self.graph.create(subheading_relationship)

        for (parent, level), headings_list in self.headings_by_level.items():
            sorted_headings = self._sort_headings(headings_list)
            for i in range(len(sorted_headings) - 1):
                next_heading_relationship = IsNextHeadingRelationship(start_node=self.nodes[sorted_headings[i]], end_node=self.nodes[sorted_headings[i + 1]])
                self.graph.create(next_heading_relationship)
                
    def _add_content_nodes_and_content_relationships_to_graph(self):
        for heading, contents in self.has_content_relationships.items():
            previous_content_node = None
            for i, content in enumerate(contents):
                if self.emb_model:
                    content_emb: list = self.emb_model.encode(content).tolist()
                    content_node = ContentNode(name=f"{heading}_content_{i+1}", content=content, source_document_id=self._parent_document_node.id, content_global_id=self._generate_content_id(), content_emb=content_emb)
                else:
                    raise EmbeddingModelNotFoundError()
                self.graph.create(content_node)

                has_content_relationship = HasDirectContentRelationship(start_node=self.nodes[heading], end_node=content_node)
                self.graph.create(has_content_relationship)

                is_a_content_in_document_relationship = IsAContentInDocumentRelationship(start_node=content_node, end_node=self._parent_document_node)
                self.graph.create(is_a_content_in_document_relationship)

                if previous_content_node:
                    is_next_content_relationship = IsNextContentRelationship(start_node=previous_content_node, end_node=content_node)
                    self.graph.create(is_next_content_relationship)
                
                previous_content_node = content_node

    def _generate_heading_id(self):
        self.heading_id += 1
        return f"D{self._parent_document_node.id}_H{self.heading_id}"
    def _generate_content_id(self):
        self.content_id += 1
        return f"D{self._parent_document_node.id}_C{self.content_id}"