from rank_bm25 import BM25Okapi
from knowledge.embeddings import Embeddings, EmbeddingEntry
from knowledge.knowledge_store import KnowledgeStore
from typing import List, Dict
from core.search.search_result import CombinedSearchResults, SearchResult

class KnowledgeSearch:
    def __init__(self, embeddings: Embeddings, knowledge_store: KnowledgeStore):
        self.embeddings = embeddings
        self.knowledge_store = knowledge_store
        self.documents = list(self.embeddings.get_all_documents().values())
        
        corpus_for_bm25 = []
        for doc in self.documents:
            classname = doc.full_classname
            summary = ""
            if classname:
                class_info = self.knowledge_store.get_class_description(classname)
                if class_info:
                    summary = class_info.summary
                    if class_info.methods:
                        for method in class_info.methods:
                            summary += f"- Method `{method.method_name}`: {method.method_summary}\n"
                    
                    if class_info.variables:
                        for var in class_info.variables:
                            summary += f"- Variable `{var.variable_name}`: {var.variable_summary}\n"
            
            corpus_for_bm25.append(summary.lower())

        self.tokenized_corpus = [text.split() for text in corpus_for_bm25]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def vector_search(self, query_embeddings: str, limit: int = 5) -> List[Dict]:
        """
        Performs vector similarity search.
        """
        return self.embeddings.search_similar(query_embeddings, limit=limit)

    def bm25_search(self, query_bm25: str, limit: int = 5) -> List[Dict]:
        """
        Performs BM25 search.
        """
        tokenized_query = query_bm25.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        max_bm25_score = max(doc_scores) if any(doc_scores) else 0.0
        
        if max_bm25_score > 0:
            normalized_scores = [score / max_bm25_score for score in doc_scores]
        else:
            normalized_scores = [0.0] * len(doc_scores)
        
        bm25_results = []
        for i, score in enumerate(normalized_scores):
            if score > 0 and i < len(self.documents):
                item = self.documents[i]
                bm25_results.append({
                    "item": item,
                    "score": score
                })
        
        return sorted(bm25_results, key=lambda x: x['score'], reverse=True)[:limit]

    def hybrid_search(self, query_embeddings: str, query_bm25: str) -> List[SearchResult]:
        vector_results = self.vector_search(query_embeddings, limit=5)
        bm25_results = self.bm25_search(query_bm25, limit=5)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in vector_results:
            combined_results.add_vector_result(result["item"], result["score"])

        for result in bm25_results:
            combined_results.add_bm25_result(result["item"], result["score"])

        return combined_results.get_sorted_results()