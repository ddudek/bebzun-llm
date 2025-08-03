import logging
import ollama
from typing import List
from .embedding_execution import EmbeddingExecution

class OllamaEmbeddingExecution(EmbeddingExecution):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.embedding_model = None

    def init_embeddings(self, model: str):
        self.embedding_model = model

    def generate_embeddings(self, text: str, vector_dimension: int) -> List[float]:
        if not self.embedding_model:
            raise Exception("Embeddings model not initialized")

        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        
        embeddings = response.get('embedding', [])
        
        if len(embeddings) != vector_dimension:
            self.logger.warning(f"Warning: Expected embedding dimension {vector_dimension}, got {len(embeddings)}")
        
        return embeddings