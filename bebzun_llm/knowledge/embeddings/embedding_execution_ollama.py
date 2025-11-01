import logging
import ollama
from typing import List
from .embedding_execution import EmbeddingExecution

class OllamaEmbeddingExecution(EmbeddingExecution):
    def __init__(self, logger: logging.Logger, url: str):
        self.logger = logger
        self.embedding_model = None
        self.client = ollama.Client(host=url)
        self.logger.info(f"Initializing ollama embeddings at: {url}")

    def init_embeddings(self, model: str):
        self.embedding_model = model

    def _generate_embedding(self, text: str, vector_dimension: int) -> List[float]:
        if not self.embedding_model:
            raise Exception("Embeddings model not initialized")
        
        self.logger.info(f"Processing embeddings for:\n ------- \n{text}\n -------")

        response = self.client.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        
        embeddings = response.get('embedding', [])
        
        if len(embeddings) != vector_dimension:
            self.logger.warning(f"Warning: Expected embedding dimension {vector_dimension}, got {len(embeddings)}")
        
        return embeddings

    def generate_query_embedding(self, text: str, vector_dimension: int) -> List[float]:
        query = text
        
        if "nomic" in self.embedding_model:
            query = "search_query: " + text

        return self._generate_embedding(query, vector_dimension)

    def generate_documents_embedding(self, texts: List[str], vector_dimension: int) -> List[List[float]]:
        return [self._generate_embedding(text, vector_dimension) for text in texts]