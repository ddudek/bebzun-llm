import logging
import openai
from typing import List
from .embedding_execution import EmbeddingExecution

class OpenAIEmbeddingExecution(EmbeddingExecution):
    def __init__(self, logger: logging.Logger, url: str):
        self.logger = logger
        self.embedding_model = None
        self.client = openai.OpenAI(api_key="", base_url=url)
        logger.info(f"Initialized embeddings at {url}")

    def init_embeddings(self, model: str):
        self.embedding_model = model

    def _generate_embedding(self, text: str, vector_dimension: int) -> List[float]:
        if not self.embedding_model:
            raise Exception("Embeddings model not initialized")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        
        embedding = response.data[0].embedding
        
        if len(embedding) != vector_dimension:
            self.logger.warning(f"Warning: Expected embedding dimension {vector_dimension}, got {len(embedding)}")
        
        return embedding

    def generate_query_embedding(self, text: str, vector_dimension: int) -> List[float]:
        return self._generate_embedding("search_query: " + text, vector_dimension)

    def generate_documents_embedding(self, texts: List[str], vector_dimension: int) -> List[List[float]]:
        if not self.embedding_model:
            raise Exception("Embeddings model not initialized")

        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            logger_text = '\n'.join(batch_texts)
            self.logger.info(f"Processing batch of embeddings:\n ------- \n{logger_text}\n -------")

            self.logger.info(f"Progress: {i}..{i + batch_size} / {len(texts)}")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text for text in batch_texts],
            )


            batch_embeddings = [None] * len(batch_texts)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding

            for embedding in batch_embeddings:
                if len(embedding) != vector_dimension:
                    self.logger.warning(f"Warning: Expected embedding dimension {vector_dimension}, got {len(embedding)}")
            
            all_embeddings.extend(batch_embeddings)

        return all_embeddings