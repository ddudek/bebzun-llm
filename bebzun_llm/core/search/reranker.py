from typing import List, Optional
import logging
from bebzun_llm.core.config.config import Config
from bebzun_llm.core.search.reranker_execution_transformers import RerankerExecutionTransformers
from bebzun_llm.core.search.reranker_execution_llama import RerankerExecutionLlama

class Reranker:
    def __init__(self, config: Config, logger: logging.Logger):
        self.logger = logger
        self.execution = None
        if config.reranker.mode == "experimental-transfrormers-qwen3":
            self.execution = RerankerExecutionTransformers(logger)
        elif config.reranker.mode == "llama_rerank":
            self.execution = RerankerExecutionLlama(logger, config.reranker.url, config.reranker.model)

    def isEnabled(self) -> bool:
        return self.execution is not None

    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        if not self.isEnabled():
            raise Exception("Reranker not enabled in config")
        
        self.logger.debug(f"Reranking {len(documents)} documents for query: '{query}'")
        scores = self.execution.rerank(query, documents, instruction)
        self.logger.debug(f"Reranking scores: {scores}")
        return scores
