from rank_bm25 import BM25Okapi
from knowledge.embeddings_store import Embeddings
from core.search.embedding_entry import EmbeddingEntry
from knowledge.knowledge_store import KnowledgeStore
from knowledge.model import ClassDescription
from typing import List, Dict
from core.config.config import Config
from core.search.reranker import Reranker
from core.search.search_result import CombinedSearchResults, SearchResult
import logging

class KnowledgeSearch:
    def __init__(self, embeddings: Embeddings, knowledge_store: KnowledgeStore, config: Config, logger: logging.Logger):
        self.logger = logger
        self.reranker = Reranker(config, self.logger)
        self.embeddings = embeddings
        self.knowledge_store = knowledge_store
        self.embedding_entries: List[EmbeddingEntry] = []

        documents = []
        corpus_for_bm25 = []
        for doc in self.embeddings.get_all_documents().values():
            class_info = self.knowledge_store.get_class_description(doc.full_classname)
            if not class_info:
                print (f"Warn: no description found for embedding entry: {doc.full_classname}")
                continue

            summary = self._get_document_summary(doc, class_info)
            corpus_for_bm25.append(summary.lower())
            documents.append(doc)

        self.embedding_entries = documents
        self.tokenized_corpus = [text.split() for text in corpus_for_bm25]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_document_summary(self, item: EmbeddingEntry, class_info: ClassDescription) -> str:
        summary = ""

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
        items =  self.embeddings.search_similar(query_embeddings, limit=limit)

        search_results = []
        for item, score in items:
            class_info = self.knowledge_store.get_class_description(item.full_classname)

            if not class_info:
                print (f"Warn: no description found for embedding entry: {item.full_classname}")
                continue
                
            search_result = SearchResult(
                full_classname=item.full_classname,
                file=item.rel_path,
                details=[],
                class_description=class_info,
                vector_score=score
            )
            search_result.add_detail_embedding(item, class_info)
            search_results.append(search_result)

        return search_results
    
    def vector_search_combined(self, query_embeddings: str, limit: int = 10) -> List[SearchResult]:
        """
        Performs vector similarity search.
        """
        results: List[tuple[EmbeddingEntry, float]] = self.embeddings.search_similar(query_embeddings, limit=limit)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for (item, vector_score) in results:
            class_info = self.knowledge_store.get_class_description(item.full_classname)
            if not class_info:
                print (f"Warn: no description found for embedding entry: {item.full_classname}")
                continue

            search_result = SearchResult(
                full_classname=item.full_classname,
                file=item.rel_path,
                details=[],
                class_description=class_info
            )
            search_result.add_detail_embedding(item, class_info)
            combined_results.merge_search_result(search_result, class_info)
            combined_results.add_vector_result(item.full_classname, vector_score)

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
            if score > 0 and i < len(self.embedding_entries):
                item = self.embedding_entries[i]

                class_info = self.knowledge_store.get_class_description(item.full_classname)
                if not class_info:
                    print (f"Warn: no description found for embedding entry: {item.full_classname}")
                    continue

                search_result = SearchResult(
                    full_classname=item.full_classname, 
                    file = item.rel_path,
                    class_description=class_info,
                    bm25_score=score, 
                    details=[]
                )
                search_result.add_detail_embedding(item, class_info)
                bm25_results.append(search_result)
        
        return sorted(bm25_results, key=lambda x: x.bm25_score, reverse=True)[:limit]
    

    def bm25_search_combined(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Performs vector similarity search.
        """
        results = self.bm25_search(query, limit=limit)

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in results:
            class_info = self.knowledge_store.get_class_description(result.full_classname)
            combined_results.merge_search_result(result, class_info)
            combined_results.add_vector_result(result.full_classname, result.vector_score)

        return combined_results.get_sorted_results()

    def hybrid_search(self, query_embeddings: List[str], query_bm25: List[str], limit: int = 15) -> List[SearchResult]:
        vector_results: List[SearchResult] = []
        for query in query_embeddings:
            vector_results.extend(self.vector_search(query, limit))

        bm25_results: List[SearchResult] = []
        for query in query_bm25:
            bm25_results.extend(self.bm25_search(query, limit - 5))

        # Combine and re-rank results
        combined_results = CombinedSearchResults()

        for result in vector_results:
            class_info = self.knowledge_store.get_class_description(result.full_classname)
            combined_results.merge_search_result(result, class_info)
            combined_results.add_vector_result(result.full_classname, result.vector_score)

        for result in bm25_results:
            class_info = self.knowledge_store.get_class_description(result.full_classname)
            combined_results.merge_search_result(result, class_info)
            combined_results.add_bm25_result(result.full_classname, result.bm25_score)

        sorted_results = combined_results.get_sorted_results()

        return sorted_results
    
    def rerank_results(self, sorted_results: List[SearchResult], query: str, rerank_limit: int = 20) -> List[SearchResult]:
        if self.reranker.isEnabled():
            results = sorted_results[-rerank_limit:]

            print("\nReranking...")
            for idx, item in enumerate(results):
                print(f"{idx}. {item.full_classname}")
                self.logger.debug(f"{idx}. {item.describe_content()}")

            docs_to_rerank = [res.describe_content() for res in results]
            
            instruction = "Given a user query, score documents of class summaries that are relevant to the query"
            rerank_scores = self.reranker.rerank(query, docs_to_rerank, instruction=instruction)
            
            for i, res in enumerate(results):      
                res.rerank_score = rerank_scores[i]      

            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

            results_limited = results #[x for x in results if x.rerank_score > 0.0001]
            print("\n\nResults limited:")
            for idx, item in enumerate(results_limited):
                print(f"{idx}. {round(item.rerank_score, 3)}: {item.full_classname}")

            return results_limited

        return sorted_results