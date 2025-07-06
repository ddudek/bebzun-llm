from typing import Dict, List, Optional
import os
import json
import ollama
import mlx.core as mx
from core.config.config import Config, EmbeddingsConfig

class Embeddings:
    def __init__(self, config: Config):
        self.storage_path: Optional[str] = None
        self.data: List[Dict] = []
        self.config = config
        
        # Handle missing embeddings config gracefully
        embeddings_config = getattr(config, 'embeddings', None)
        if embeddings_config is None:
            embeddings_config = EmbeddingsConfig()
        
        self.embedding_model = embeddings_config.model
        self.vector_dimension = embeddings_config.vector_dimension

    def initialize(self, base_dir: str):
        """Initialize JSON file storage and load embeddings"""
        try:
            # Create storage path
            self.storage_path = os.path.join(base_dir, ".ai-agent", "embeddings.json")
            
            print(f"Using embeddings file: {self.storage_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Load data from JSON file
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        self.data = json.loads(content)
                    else:
                        self.data = []
                print(f"Successfully loaded {len(self.data)} embeddings from {self.storage_path}")
            else:
                self.data = []
                print(f"No existing embeddings file found. A new one will be created at {self.storage_path}")
            
            return True
        except Exception as e:
            print(f"Error initializing embeddings storage: {str(e)}")
            self.data = []
            return False

    def is_loaded(self) -> bool:
        """Check if the embeddings data is loaded"""
        return self.data is not None

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using the configured execution method
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        try:
            embeddings_config = getattr(self.config, 'embeddings', None)
            if embeddings_config is None:
                embeddings_config = EmbeddingsConfig()
            
            if embeddings_config.execution == "ollama":
                # Call ollama API directly to generate embeddings
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                
                # Extract embeddings from response
                embeddings = response.get('embedding', [])
                
                # Verify embedding dimension
                if len(embeddings) != self.vector_dimension:
                    print(f"Warning: Expected embedding dimension {self.vector_dimension}, got {len(embeddings)}")
                
                return embeddings
            else:
                raise ValueError(f"Unsupported embeddings execution method: {embeddings_config.execution}")
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []

    def store_class_description_embeddings(self, classname: str, summary: str, filecontext: str, metadata: dict, timestamp: str):
        """
        Store embeddings in the JSON file
        
        Args:
            classname: The full class name
            summary: The class summary
            filecontext: The file content
            metadata: Additional metadata
        """
        print(f"Store embeddings for {classname}:")
        
        # Combine summary and content as the text to embed
        text_to_embed = f"{summary}\n{filecontext}" if filecontext else summary
        print(f"Text to embedd:\n ------- \n{text_to_embed}\n -------")
        
        # Add classname to metadata
        metadata["classname"] = classname
        
        try:
            # Generate embeddings using ollama
            embedding_vector = self.generate_embeddings(text_to_embed)
            
            if not embedding_vector:
                print(f"Failed to generate embeddings for {classname}")
                return
            
            # Remove existing entry for the classname if it exists
            self.data = [entry for entry in self.data if entry.get("metadata", {}).get("classname") != classname]

            # Add new entry
            entry = {
                "full_classname": classname,
                "metadata": metadata,
                "embedding": embedding_vector,
                "last_modified_at": timestamp
            }
            self.data.append(entry)
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            
            print(f"Successfully stored embeddings for {classname} in {self.storage_path}")
        except Exception as e:
            print(f"Error storing embeddings for {classname}: {str(e)}")

    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
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
                print("Embeddings not available or empty, cannot search")
                return []

            # Generate embeddings for the query
            query_embedding = self.generate_embeddings(query)
            
            if not query_embedding:
                print(f"Failed to generate embeddings for query")
                return []
            
            query_embedding_mx = mx.array(query_embedding)

            stored_embeddings = [entry["embedding"] for entry in self.data]
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
            for i in range(min(limit, len(sorted_indices))):
                idx = sorted_indices[i].item()
                results.append({
                    "metadata": self.data[idx]["metadata"],
                    "score": float(similarities[idx].item())
                })
            
            return results

        except Exception as e:
            print(f"Error searching for similar documents: {str(e)}")
            return []

    def get_all_documents(self) -> List[Dict]:
        """
        Retrieve all documents from the storage.
        
        Returns:
            List of all documents with their content and metadata.
        """
        if not self.is_loaded():
            print("Embeddings storage not available, cannot retrieve documents")
            return []
        
        return [
            {
                "metadata": entry["metadata"]
            }
            for entry in self.data
        ]