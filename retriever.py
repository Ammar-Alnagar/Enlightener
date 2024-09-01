from typing import List
import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings: List[np.ndarray]):
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))

    

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[int]:
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0].tolist()
