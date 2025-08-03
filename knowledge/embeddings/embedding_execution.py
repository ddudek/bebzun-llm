from abc import ABC, abstractmethod
from typing import List

class EmbeddingExecution(ABC):
    @abstractmethod
    def init_embeddings(self, model: str):
        pass

    @abstractmethod
    def generate_embeddings(self, text: str, vector_dimension: int) -> List[float]:
        pass