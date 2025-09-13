from typing import Dict, List, Optional
import os
import sys
import json
import logging
import mlx.core as mx

from core.config.config import Config, EmbeddingsConfig
from core.search.embedding_entry import EmbeddingEntry
from knowledge.embeddings.embedding_execution import EmbeddingExecution
from knowledge.embeddings.embedding_execution_mlx import MlxEmbeddingExecution
from knowledge.embeddings.embedding_execution_ollama import OllamaEmbeddingExecution
from knowledge.embeddings.embedding_execution_openai import OpenAIEmbeddingExecution

class Embeddings:
    def __init__(self, config: Config, logger: logging.Logger):
        self.logger = logger
        self.storage_path: Optional[str] = None
        self.data: Dict[str, EmbeddingEntry] = {}
        self.config = config
        
        embeddings_config = getattr(config, 'embeddings', EmbeddingsConfig())
        
        self.embedding_model = embeddings_config.model
        self.vector_dimension = embeddings_config.vector_dimension
        self.execution_mode = embeddings_config.execution
        self.embedding_execution: EmbeddingExecution

        if self.execution_mode == 'mlx':
            self.embedding_execution = MlxEmbeddingExecution(logger=logger)
        elif self.execution_mode == 'ollama':
            self.embedding_execution = OllamaEmbeddingExecution(logger=logger, url=embeddings_config.url)
        elif self.execution_mode == 'openai':
            self.embedding_execution = OpenAIEmbeddingExecution(logger=logger, url=embeddings_config.url)
        else:
            raise ValueError(f"Unsupported embeddings execution mode: {self.execution_mode}")

    def initialize(self, base_dir: str, create: bool = False):
        """Initialize JSON file storage and load embeddings"""
        
        self.storage_path = os.path.join(base_dir, ".ai-agent", "db_embeddings.json")
        
        self.logger.info(f"Using embeddings file: {self.storage_path}")
        
        if create:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        else:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        loaded_data = json.loads(content)
                        self.data = {k: EmbeddingEntry(**v) for k, v in loaded_data.items()}
                    else:
                        self.data = {}
                print(f"Successfully loaded {len(self.data)} embeddings from {self.storage_path}")
            else:
                raise Exception("db_embeddings.json file not found")

        self.embedding_execution.init_embeddings(self.embedding_model)
        return True

    def is_loaded(self) -> bool:
        """Check if the embeddings data is loaded"""
        return self.data is not None

    def generate_documents_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Generate document embeddings using the configured execution method
        
        Args:
            texts: The texts to embed
            
        Returns:
            List of embedding values
        """
        try:
            return self.embedding_execution.generate_documents_embedding(texts, self.vector_dimension)
        except Exception as e:
            self.logger.error(f"Error generating document embeddings: {str(e)}")
            raise

    def generate_query_embedding(self, text: str) -> List[float]:
        """
        Generate query embeddings using the configured execution method
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        try:
            query_text = f"Implementation of a class, property or method related to this query: {text}"
            return self.embedding_execution.generate_query_embedding(query_text, self.vector_dimension)
        except Exception as e:
            self.logger.error(f"Error generating query embeddings: {str(e)}")
            raise
        
    def store_all_classes(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            data_to_store = {k: v.to_dict() for k, v in self.data.items()}
            json.dump(data_to_store, f, indent=2)

    def store_class_description_embeddings(self, type: str, classname: str, detail: str, summary: str, filecontext: str, rel_path: str, timestamp: int):
        """
        Store embeddings in the JSON file
        
        Args:
            classname: The full class name
            summary: The class summary
            filecontext: The file content
        """
        self.logger.info(f"Store embeddings for {classname}:")
        
        # Combine summary and content as the text to embed
        text_to_embed = f"{summary}\n{filecontext}" if filecontext else summary
        self.logger.info(f"Text to embedd:\n ------- \n{text_to_embed}\n -------")
        
        try:
            # Generate embeddings
            embedding_vector = self.generate_documents_embedding([text_to_embed])[0]
            
            if not embedding_vector:
                self.logger.error(f"Failed to generate embeddings for {classname}")
                return
            
            entry = EmbeddingEntry(
                type=type,
                detail=detail,
                full_classname=classname,
                rel_path=rel_path,
                embedding=embedding_vector,
                version=timestamp
            )
            key = f"{classname}.{detail}" if type != 'class' else classname
            self.data[key] = entry
            
            # self.logger.info(f"Successfully stored embeddings for {classname} in {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error storing embeddings for {classname}: {str(e)}")

    def search_similar(self, query: str, limit: int = 5) -> List[tuple[EmbeddingEntry, float]]:
        """
        Search for similar documents in the vector store
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            if not self.is_loaded() or not self.data:
                self.logger.error("Embeddings not available or empty, cannot search")
                return []

            # Generate embeddings for the query
            query_embedding = self.generate_query_embedding(query)
            
            if not query_embedding:
                self.logger.error(f"Failed to generate embeddings for query")
                return []
            
            query_embedding_mx = mx.array(query_embedding)

            stored_entries = list(self.data.values())
            stored_embeddings = [entry.embedding for entry in stored_entries]
            if not stored_embeddings:
                return []
            stored_embeddings_mx = mx.array(stored_embeddings)

            # Cosine similarity calculation
            query_norm = query_embedding_mx / mx.linalg.norm(query_embedding_mx)
            stored_norm = stored_embeddings_mx / mx.linalg.norm(stored_embeddings_mx, axis=1, keepdims=True)
            
            similarities = mx.matmul(stored_norm, query_norm.T).squeeze()
            
            # Get top results
            if similarities.ndim == 0: # Handle case with one stored embedding
                sorted_indices = mx.array([0])
            else:
                sorted_indices = mx.argsort(similarities)[::-1]

            results = []
            entries_log = ""
            for i in range(min(limit, len(sorted_indices))):
                idx = sorted_indices[i].item()
                entry: EmbeddingEntry = stored_entries[idx]
                score = float(similarities[idx].item())
                entries_log += f"\n{i}. {round(score, 3)}: {entry.full_classname}{' [`' + entry.detail+'`]' if entry.detail else ''}"
                results.append((entry, score))
            
            self.logger.debug(f"Query '{query}': found {len(results)} embedding entries: {entries_log}\n")
            return results

        except Exception as e:
            self.logger.error(f"Error searching for similar documents: {str(e)}")
            raise

    def get_all_documents(self) -> Dict[str, EmbeddingEntry]:
        """
        Retrieve all documents from the storage.
        
        Returns:
            List of all documents with their content and metadata.
        """
        if not self.is_loaded():
            self.logger.error("Embeddings storage not available, cannot retrieve documents")
            return []
        
        return self.data