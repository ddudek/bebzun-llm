from typing import List, Optional
import logging
from core.config.config import Config
from core.search.reranker_execution_transformers import RerankerExecutionTransformers
from core.search.reranker_execution_llama import RerankerExecutionLlama

class Reranker:
    def __init__(self, config: Config, logger: logging.Logger):
        self.logger = logger
        self.execution = None
        if config.reranker.model == "experimental-transfrormers-qwen3":
            self.execution = RerankerExecutionTransformers(logger)
        elif config.reranker.model == "llama_rerank":
            self.execution = RerankerExecutionLlama(logger, config.reranker.url)

    def isEnabled(self) -> bool:
        return self.execution is not None

    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        if not self.isEnabled():
            raise Exception("Reranker not enabled in config")
        
        self.logger.debug(f"Reranking {len(documents)} documents for query: '{query}'")
        scores = self.execution.rerank(query, documents, instruction)
        self.logger.debug(f"Reranking scores: {scores}")
        return scores
