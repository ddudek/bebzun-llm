from rank_bm25 import BM25Okapi
from knowledge.embeddings import Embeddings
from core.search.embedding_entry import EmbeddingEntry
from knowledge.knowledge_store import KnowledgeStore
from typing import List, Dict
from core.config.config import Config
from core.search.reranker import Reranker
from core.search.search_result import CombinedSearchResults, SearchResult
import logging

class KnowledgeSearch:
    def __init__(self, embeddings: Embeddings, knowledge_store: KnowledgeStore, config: Config, logger: logging.Logger):
        self.reranker = None
        if config.reranker.model == "experimental-transfrormers-qwen3":
            self.reranker = Reranker(logger)
        self.embeddings = embeddings
        self.knowledge_store = knowledge_store
        self.documents = list(self.embeddings.get_all_documents().values())
        
        corpus_for_bm25 = []
        for doc in self.documents:
            summary = self._get_document_summary(doc)
            corpus_for_bm25.append(summary.lower())

        self.tokenized_corpus = [text.split() for text in corpus_for_bm25]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_document_summary(self, item: EmbeddingEntry) -> str:
        summary = ""
        class_info = self.knowledge_store.get_class_description(item.full_classname)

        if not class_info:
            print (f"Warn: no description found for embedding entry: {item.full_classname}")

        if item.type == 'class':
                summary = class_info.describe("- ") + "\n"

        if item.type == 'method':
            if class_info:
                detail = class_info.find_method(item.detail)
                summary = f"Class {class_info.summary}\n"
                if detail:
                    summary += f"Method: `{detail.method_name}`: {detail.method_summary}"

        if item.type == 'property':
            if class_info:
                detail = class_info.find_property(item.detail)
                summary = f"Class {class_info.summary}\n"
                if detail:
                    summary += f"Method: `{detail.property_name}`: {detail.property_summary}"
        
        return summary

    def vector_search(self, query_embeddings: str, limit: int = 10) -> List[SearchResult]:
        """
        Performs vector similarity search.
        """
        return self.embeddings.search_similar(query_embeddings, limit=limit)
    
    def vector_search_combined(self, query_embeddings: str, limit: int = 10) -> List[SearchResult]:
        """
        Performs vector similarity search.
        """
        results = self.embeddings.search_similar(query_embeddings, limit=limit)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in results:
            class_info = self.knowledge_store.get_class_description(result.entry.full_classname)
            combined_results.add_detail(result.entry, class_info)
            combined_results.add_vector_result(result.entry, result.vector_score)

        return combined_results.get_sorted_results()

    def bm25_search(self, query_bm25: str, limit: int = 10) -> List[SearchResult]:
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
        
        bm25_results: List[SearchResult] = []
        for i, score in enumerate(normalized_scores):
            if score > 0 and i < len(self.documents):
                item = self.documents[i]
                bm25_results.append(SearchResult(entry=item, bm25_score=score, details=[]))
        
        return sorted(bm25_results, key=lambda x: x.bm25_score, reverse=True)[:limit]
    

    def bm25_search_combined(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Performs vector similarity search.
        """
        results = self.bm25_search(query, limit=limit)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in results:
            class_info = self.knowledge_store.get_class_description(result.entry.full_classname)
            combined_results.add_detail(result.entry, class_info)
            combined_results.add_vector_result(result.entry, result.vector_score)

        return combined_results.get_sorted_results()

    def hybrid_search(self, query_embeddings: str, query_bm25: str, limit: int = 15) -> List[SearchResult]:
        vector_results = self.vector_search(query_embeddings, limit)
        bm25_results = self.bm25_search(query_bm25, limit - 5)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in vector_results:
            class_info = self.knowledge_store.get_class_description(result.entry.full_classname)
            combined_results.add_detail(result.entry, class_info)
            combined_results.add_vector_result(result.entry, result.vector_score)

        for result in bm25_results:
            class_info = self.knowledge_store.get_class_description(result.entry.full_classname)
            combined_results.add_detail(result.entry, class_info)
            combined_results.add_bm25_result(result.entry, result.bm25_score)

        sorted_results = combined_results.get_sorted_results()

        return sorted_results
    
    def rerank_results(self, sorted_results: List[SearchResult], query: str, rerank_limit: int = 20) -> List[SearchResult]:
        if self.reranker:
            results = sorted_results[-rerank_limit:]

            print("\n\nReranking:")
            for idx, item in enumerate(results):
                print(f"{idx}.{item.describe_content()}\n")

            docs_to_rerank = [res.describe_content() for res in results]
            
            instruction = "Given a user query, retrieve documents of code summaries that are relevant to the query"
            rerank_scores = self.reranker.rerank(query, docs_to_rerank, instruction=instruction)
            
            for i, res in enumerate(results):      
                res.rerank_score = rerank_scores[i]      

            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

            results_limited = [x for x in results if x.rerank_score > 0.0001]
            print("\n\nResults limited:")
            for idx, item in enumerate(results_limited):
                print(f"{idx}.{item.rerank_score}, {item.describe_content()}\n")

            return results_limited

        return sorted_results