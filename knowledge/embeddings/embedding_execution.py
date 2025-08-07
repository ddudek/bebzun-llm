from abc import ABC, abstractmethod
from typing import List

class EmbeddingExecution(ABC):
    @abstractmethod
    def init_embeddings(self, model: str):
        pass

    @abstractmethod
    def generate_query_embedding(self, text: str, vector_dimension: int) -> List[float]:
        pass

    @abstractmethod
    def generate_documents_embedding(self, texts: List[str], vector_dimension: int) -> List[List[float]]:
        pass