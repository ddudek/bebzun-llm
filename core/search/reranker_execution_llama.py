import requests
import logging
from typing import List, Optional
import json

class RerankerExecutionLlama:
    def __init__(self, logger: logging.Logger, url: str):
        self.logger = logger
        self.url = url

    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        self.logger.debug(f"Reranking {len(documents)} documents for query: '{query}' using llama reranker at {self.url}")
    
        json_doc = {
                "query": query,
                "documents": [d.replace('\'', '').replace("\"","").replace("`","").replace("\n","") for d in documents],
            }
        data = json.dumps(json_doc, indent=2)
        self.logger.debug(f"Reranking request: {data}")
        response = requests.post(
            self.url,
            data=data,
        )

        self.logger.debug(f"Reranking results: {response}, {response.reason}")

        response.raise_for_status()
        results = response.json()
        
        # Initialize scores with 0.0 for all documents
        scores = [0.0] * len(documents)
        
        # Update scores based on the reranker's response
        for result in results['results']:
            print(result)
            index = result['index']
            score = result['relevance_score']
            if 0 <= index < len(scores):
                scores[index] = score
            else:
                self.logger.warning(f"Received out-of-bounds index {index} from reranker.")

        return scores