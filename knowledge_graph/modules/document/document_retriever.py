import torch

from base_classes.retriever import Retriever
from base_classes.embedding_model import AbstractEmbeddingModel
from knowledge_graph.modules.document.document_layer import DocumentLayer

class DocumentRetriever(Retriever):
    def __init__(self, emb_model: AbstractEmbeddingModel, layer: DocumentLayer):
        super().__init__(emb_model=emb_model, layer=layer)
    
    def content_retrieve(self, query: str, k:int = 3):
        query_emb = torch.tensor(self._embed_query(query))
        content_nodes = self._layer.get_content_nodes()

        similarities = []
        for node in content_nodes:
            node_emb = torch.tensor(node["content_emb"])
            similarity = self.compute_similarity(query_emb, node_emb)
            similarities.append(({'node_id': node['content_global_id'], 'content': node["content"]}, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = [item[0] for item in similarities[:k]]

        return top_k