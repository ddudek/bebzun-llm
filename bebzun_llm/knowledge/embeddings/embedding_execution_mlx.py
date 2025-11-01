import logging
from typing import List, Optional
import mlx.core as mx
from mlx_embeddings.utils import load as load_embed_model
from mlx_embeddings.tokenizer_utils import TokenizerWrapper
from .embedding_execution import EmbeddingExecution

class MlxEmbeddingExecution(EmbeddingExecution):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.embedding_model = None
        self.embedding_tokenizer: TokenizerWrapper = None

    def init_embeddings(self, model: str):
        self.logger.info(f"Loading embeddings model: {model}...\n")
        self.embedding_model, self.embedding_tokenizer = load_embed_model(model)

    def _generate_embedding(self, text: str, vector_dimension: int) -> List[float]:
        if self.embedding_model is None or self.embedding_tokenizer is None:
            raise Exception("Embeddings model not initialized")

        input_ids = self.embedding_tokenizer.encode(text, return_tensors="mlx")
        outputs = self.embedding_model(input_ids)
        embeddings_te: Optional[mx.array] = outputs.text_embeds
        embeddings = embeddings_te[1].tolist()

        mx.clear_cache()

        if len(embeddings) != vector_dimension:
            self.logger.warning(f"Warning: Expected embedding dimension {vector_dimension}, got {len(embeddings)}")
        
        return embeddings

    def generate_query_embedding(self, text: str, vector_dimension: int) -> List[float]:
        return self._generate_embedding("search_query: " + text, vector_dimension)

    def generate_documents_embedding(self, texts: List[str], vector_dimension: int) -> List[List[float]]:
        return [self._generate_embedding("search_document: " + text, vector_dimension) for text in texts]