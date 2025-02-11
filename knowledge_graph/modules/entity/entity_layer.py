from typing import List
from collections import defaultdict

from base_classes.graph_elements.knowledge_graph import KnowledgeGraph
from base_classes.embedding_model import AbstractEmbeddingModel
from base_classes.graph_elements.layer import KnowledgeGraphLayer
from knowledge_graph.modules.node import EntityNode, ContentNode, HeadingNode, DocumentNode
from knowledge_graph.modules.relationship import HasEntityRelationship, BaseEntityRelationship
from knowledge_graph.modules.entity.er_extractor import ERExtractor
from exception.entity_exception import EntityDuplicationInOneContentNodeError, EntitiesAreDangerForMergingError
from exception.embedding_exception import EmbeddingModelNotFoundError

class EntityLayer(KnowledgeGraphLayer):
    er_extractor: ERExtractor = None
    def __init__(self, graph: KnowledgeGraph) -> None:
        super().__init__(graph)

    def load_er_extractor(self, er_extractor: ERExtractor) -> None:
        self._er_extractor = er_extractor
        self._er_extractor.load_embedding_model(self.emb_model)

    # Getter methods
    def get_entity_nodes_by_content(self, content_node: ContentNode) -> List[EntityNode]:
        _entity_nodes = self.graph.get_entity_nodes_by_content(content_node)
        entity_nodes = [EntityNode(**raw_entity_node) for raw_entity_node in _entity_nodes]
        return entity_nodes
    
    def get_entity_nodes_by_heading(self, heading_node: HeadingNode) -> List[EntityNode]:
        _entity_nodes = self.graph.get_entity_nodes_by_heading(heading_node)
        entity_nodes = [EntityNode(**raw_entity_node) for raw_entity_node in _entity_nodes]
        return entity_nodes
    
    def get_entity_nodes(self) -> List[EntityNode]:
        _entity_nodes = self.graph.get_entity_nodes()
        entity_nodes = [EntityNode(**raw_entity_node) for raw_entity_node in _entity_nodes]
        return entity_nodes

    # Graph modification methods
    def _add_entity_node_to_graph(self, entity_node: EntityNode, source_content_node: ContentNode) -> EntityNode:
        available_entity_nodes_from_content_node_cypher = f"""
        MATCH (m:__Content__ {{content_global_id: '{source_content_node.content_global_id}'}})-[:__HAS_ENTITY__]->(n:__Entity__)
        RETURN n
        """
        available_entity_nodes_from_content_node = self.graph.run(available_entity_nodes_from_content_node_cypher).data()
        
        if entity_node.name.lower() in [entity_node['n']['name'].lower() for entity_node in available_entity_nodes_from_content_node]:
            raise EntityDuplicationInOneContentNodeError(entity_node, source_content_node)
        
        self.graph.create(entity_node)
        self.graph.create(HasEntityRelationship(self.graph.get_content_node_by_id(source_content_node.content_global_id), entity_node))

        return entity_node
    
    # Entity and relationship extraction
    def er_process_content_node(self, content_node: ContentNode, num_trials: int=1) -> None:
        entities_and_relationships = self._er_extractor.er_extraction(content_node=content_node, num_trials=num_trials)
        entity_list = entities_and_relationships['entities']
        relationship_list = entities_and_relationships['relationships']

        if len(entity_list) == 0:
            return

        nodes_dict = {}

        for entity in entity_list:
            entity_node = EntityNode(name=entity['name'],
                                        definition='',
                                        description=entity['description'],
                                        types=[entity['type']],
                                        source_content_ids=[content_node.content_global_id],
                                        name_emb=self.emb_model.encode(entity['name']).tolist(),
                                        definition_emb=[],
                                        description_emb=self.emb_model.encode(entity['description']).tolist())
            
            self._add_entity_node_to_graph(entity_node, content_node)

            nodes_dict[entity['name']] = entity_node

        for relationship in relationship_list:
            source_node = nodes_dict.get(relationship['source'])
            target_node = nodes_dict.get(relationship['target'])
            try:
                relationship = BaseEntityRelationship(start_node = source_node, end_node = target_node, description = relationship['relationship'], strength = relationship['relationship_strength'])
                self.graph.create(relationship)
            except Exception as e:
                print(e)
                continue

    # Entity graph resolution
    def entity_similarity(self, entity_node_1: EntityNode, entity_node_2: EntityNode) -> float:
        if self.emb_model is None:
            raise EmbeddingModelNotFoundError()

        name_similarity = self.emb_model.similarity(entity_node_1.name, entity_node_2.name)
        definition_similarity = self.emb_model.similarity(entity_node_1.definition, entity_node_2.definition)
        description_similarity = self.emb_model.similarity(entity_node_1.description, entity_node_2.description)

        return {"name_similarity": name_similarity, "definition_similarity": definition_similarity, "description_similarity": description_similarity}
    
    def merge_entity_nodes_from_id_list(self, entity_node_ids: list[int], forced: False) -> EntityNode:
        """
        Merge all entity nodes in the provided list of IDs into a single node in the graph database.
        If forced is False and the names of the nodes do not match, an exception is raised.
        Otherwise, the nodes are merged with their properties combined.

        Args:
            entity_node_ids (list[int]): A list of entity node IDs to merge.
            forced (bool): If True, forces the merge even if the names don't match.

        Returns:
            EntityNode: The newly merged node.
        """
        if not entity_node_ids:
            raise ValueError("The entity_node_ids list cannot be empty.")

        entity_nodes = []
        for node_id in entity_node_ids:
            result = self.graph.run(
                f"MATCH (n:__Entity__) WHERE id(n) = {node_id} RETURN n"
            ).data()
            if result:
                entity_nodes.append(result[0]['n'])
            else:
                raise ValueError(f"Node with ID {node_id} not found")

        base_node = entity_nodes[0]

        if not forced:
            for node in entity_nodes[1:]:
                if node['name'] != base_node['name']:
                    raise EntitiesAreDangerForMergingError(entity_nodes)

        print(f"Merging {len(entity_nodes)} entity nodes")

        combined_definitions = [node['definition'] for node in entity_nodes]
        combined_descriptions = [node['description'] for node in entity_nodes]
        combined_types = set()
        combined_source_content_ids = set()

        for node in entity_nodes:
            combined_types.update(node['types'])
            combined_source_content_ids.update(node['source_content_ids'])

        new_node = EntityNode(
            name=base_node['name'],
            definition="\n\n".join(combined_definitions),
            description="\n\n".join(combined_descriptions),
            types=list(combined_types),
            source_content_ids=list(combined_source_content_ids),
            name_emb=base_node['name_emb'],
            definition_emb=self.emb_model.encode("\n\n".join(combined_definitions)).tolist(),
            description_emb=self.emb_model.encode("\n\n".join(combined_descriptions)).tolist()
        )

        self.graph.create(new_node)

        new_node_id = self.graph.run(
            f"MATCH (n:__Entity__ {{name: '{new_node.name}', description: '{new_node.description}'}}) RETURN id(n)"
        ).data()[0]['id(n)']

        merging_node_id = entity_node_ids[0]
        for node_id in entity_node_ids[1:]:
            if node_id == merging_node_id:
                continue

            cypher_merge_nodes = f"""
            MATCH (n1:__Entity__), (n2:__Entity__)
            WHERE id(n1) = {merging_node_id} AND id(n2) = {node_id}
            CALL apoc.refactor.mergeNodes([n1, n2], {{properties: 'overwrite'}})
            YIELD node
            RETURN node        
            """

            try:
                self.graph.run(cypher_merge_nodes)
                print(f"Merged node {node_id} into the new node {merging_node_id}")
            except Exception as e:
                print(f"Error while merging node {node_id}: {e}")

        cypher_update_merged_node_properties = f"""
        MATCH (n1:__Entity__), (n2:__Entity__)
        WHERE id(n1) = {merging_node_id} AND id(n2) = {new_node_id}
        CALL apoc.refactor.mergeNodes([n1, n2], {{properties: 'overwrite'}})
        YIELD node
        RETURN node         
        """
        self.graph.run(cypher_update_merged_node_properties)

        print("Completed merging all entity nodes")
        
        return new_node
    
    def _similarity_analysis(self, document_node: DocumentNode, similarity_threshold: float = 0.95):
        projection_cypher_query = f"""CALL gds.graph.project.cypher(
        'Similar_Entity_Graph',
        "MATCH (d:__Document__ {{document_id: '{document_node.id}'}})
        MATCH (c:__Content__)-[:__IS_A_CONTENT_IN_DOCUMENT__]->(d)
        MATCH (c)-[:__HAS_ENTITY__]->(e:__Entity__)
        RETURN id(e) AS id, e.name_emb AS embedding",
        'MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        RETURN id(e1) AS source, id(e2) AS target, type(r) AS type'
        )
        """
        self.graph_ds.run_cypher(projection_cypher_query)

        cypher_query_knn = """
        CALL gds.knn.mutate(
        'Similar_Entity_Graph',  
        {
            nodeProperties: ['embedding'],         
            mutateRelationshipType: 'SIMILAR',     
            mutateProperty: 'score',               
            similarityCutoff: 0.95
        }
        )
        YIELD relationshipsWritten
        RETURN relationshipsWritten
        """
        self.graph_ds.run_cypher(cypher_query_knn)

        cypher_query_wcc = """
        CALL gds.wcc.write(
        'Similar_Entity_Graph',
        {
            writeProperty: 'wcc',
            relationshipTypes: ['SIMILAR']
        }
        )
        YIELD nodePropertiesWritten, componentCount
        RETURN nodePropertiesWritten, componentCount
        """
        self.graph_ds.run_cypher(cypher_query_wcc)

        self.graph_ds.graph.drop("Similar_Entity_Graph")

    def _duplicate_candidates_detection(self, document_node: DocumentNode, word_edit_distance:int = 3, prefix_length:int = 3):
        detection_query = f"""
        MATCH (d:__Document__ {{document_id: '{document_node.id}'}})
        MATCH (e:__Entity__)<-[:__HAS_ENTITY__]-(c:__Content__)-[:__IS_A_CONTENT_IN_DOCUMENT__]->(d)
        WHERE size(e.name) > {prefix_length}
        WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
        WHERE count > 1
        UNWIND nodes AS node
        WITH DISTINCT
        [n IN nodes WHERE apoc.text.distance(toLower(node.name), toLower(n.name)) < {word_edit_distance} | id(n)] AS intermediate_results
        WHERE size(intermediate_results) > 1
        WITH collect(intermediate_results) AS results
        UNWIND range(0, size(results)-1, 1) AS index
        WITH results, index, results[index] AS result
        WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
        )) AS combinedResult
        WITH DISTINCT(combinedResult) AS combinedResult
        WITH collect(combinedResult) AS allCombinedResults
        UNWIND range(0, size(allCombinedResults)-1, 1) AS combinedResultIndex
        WITH allCombinedResults[combinedResultIndex] AS combinedResult, combinedResultIndex, allCombinedResults
        WHERE NOT any(x IN range(0, size(allCombinedResults)-1, 1)
            WHERE x <> combinedResultIndex
            AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
        )
        RETURN combinedResult
        """

        duplicate_candidate = self.graph.query(detection_query).data()

        return [res['combinedResult'] for res in duplicate_candidate]

    def find_similar_entity_nodes_in_given_document_node(self, document_node: DocumentNode, similarity_threshold: float=0.95, word_edit_distance:int = 3, prefix_length:int = 3):
        self._similarity_analysis(document_node = document_node, similarity_threshold = similarity_threshold)
        duplicate_candidate = self._duplicate_candidates_detection(document_node = document_node, word_edit_distance = word_edit_distance, prefix_length = prefix_length)

        return duplicate_candidate

    def entity_clustering(self) -> None:
        projection_cypher_query = """
        CALL gds.graph.project(
        'Entity_Clustering',
        '__Entity__',
        {
            _ALL_: {
            type: '*',
            orientation: 'UNDIRECTED',
            properties: {
                    weight: {
                    property: '*',
                    aggregation: 'COUNT'
                    }
                }
            }
        }
        )
        YIELD graphName, nodeCount, relationshipCount;
        """
        self.graph_ds.run_cypher(projection_cypher_query)

        clustering_cypher_query = """
        CALL gds.leiden.write(
        'Entity_Clustering',
        {
            writeProperty: 'communities',
            includeIntermediateCommunities: true,
            relationshipWeightProperty: 'weight'
        }
        )
        YIELD ranLevels, communityCount, modularity, modularities
        RETURN ranLevels, communityCount, modularity, modularities;   
        """
        ranLevels, communityCount, modularity, modularities = self.graph_ds.run_cypher(clustering_cypher_query)

        self.graph_ds.graph.drop("Entity_Clustering")

        return ranLevels, communityCount, modularity, modularities

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        if node not in self.parent:
            self.parent[node] = node
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1

