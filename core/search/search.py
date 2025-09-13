from rank_bm25 import BM25Okapi
from knowledge.embeddings_store import Embeddings
from core.search.embedding_entry import EmbeddingEntry
from knowledge.knowledge_store import KnowledgeStore
from knowledge.model import ClassDescription, VariableDescription
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

            summary = self._get_document_summary_bm25(doc, class_info)
            corpus_for_bm25.append(summary.lower())
            documents.append(doc)

        self.embedding_entries = documents
        self.tokenized_corpus = [text.split() for text in corpus_for_bm25]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_document_summary_bm25(self, item: EmbeddingEntry, class_info: ClassDescription) -> str:
        summary = ""

        if item.type == 'class':
            summary = class_info.describe("- ") + "\n"

        if item.type == 'method':
            if class_info:
                detail = class_info.find_method(item.detail)
                summary = f"Class {class_info.full_classname}\n"
                if detail:
                    summary += f"Method: `{detail.method_name}`: {detail.method_summary}"

        if item.type == 'property':
            if class_info:
                detail = class_info.find_property(item.detail)
                summary = f"Class {class_info.full_classname}\n"
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
        
        max_bm25_score = float(max(doc_scores) if any(doc_scores) else 0.0)
        
        if max_bm25_score > 0:
            normalized_scores = [float(score) / max_bm25_score for score in doc_scores]
        else:
            normalized_scores = [0.0] * len(doc_scores)
        
        bm25_results: List[SearchResult] = []
        for i, score in enumerate(normalized_scores):
            if score > 0.2 and i < len(self.embedding_entries):
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
                
        entries_log = ""
        result = sorted(bm25_results, key=lambda x: x.bm25_score, reverse=True)[:limit]
        for idx, search_result in enumerate(result):
            entries_log += f"\n{idx}. {round(search_result.bm25_score, 3)}: {search_result.describe_content_compact()}"

        self.logger.debug(f"Query '{query_bm25}'(BM25) found {len(result)} entries: {entries_log}\n")

        return result
    

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
            search_result = self.vector_search(query, limit)

            # This tries to squash similar results together.
            # Nearby results which are similar should be safe to process later together.
            squashed_search_result = self.squash_nearby_results(search_result)
            if self.logger.level == logging.DEBUG:
                print(f"After squashing:")
                self.print_results(squashed_search_result, True)
                print(f"\n")
            vector_results.extend(squashed_search_result)

        max_similarity_score = max([item.vector_score for item in vector_results])

        bm25_results: List[SearchResult] = []
        for query in query_bm25:
            search_result = self.bm25_search(query, limit - 5)

            # let's normalize bm25 scores
            for result in search_result:
                result.bm25_score = result.bm25_score * max_similarity_score

            # This tries to squash similar results together.
            # Nearby results which are similar should be safe to process later together.
            squashed_search_result = self.squash_nearby_results(search_result)
            if self.logger.level == logging.DEBUG:
                print(f"After squashing:")
                self.print_results(squashed_search_result, False)
                print(f"\n")
            bm25_results.extend(squashed_search_result)

        # Now we try to merge both vector and bm25 lists together.
        combined_results: List[SearchResult] = []
        for result in vector_results:
            self.merge_hybrid_result(combined_results, result)

        for result in bm25_results:
            self.merge_hybrid_result(combined_results, result)
        
        for result in combined_results:
            result.calculate_total_score()

        sorted_results = sorted(combined_results, key=lambda x: x.total_score, reverse=True)

        if self.logger.level == logging.DEBUG:
            entries_log = "All results combined:"
            for idx, search_result in enumerate(sorted_results):
                entries_log += f"\n{idx}. {round(search_result.total_score, 3)}: {search_result.describe_content_compact()}"
            print(entries_log)

        results_no_duplicates = self.squash_duplicates(sorted_results)

        for result in results_no_duplicates:
            result.calculate_total_score()
        
        entries_log = "\nHybrid search sorted results:"
        for idx, search_result in enumerate(results_no_duplicates):
            if self.logger.level == logging.DEBUG:
                entries_log += f"\n{idx}. {round(search_result.total_score, 3)}: {search_result.describe_content_compact()}"
            else:
                entries_log += f"\n{idx}. {round(search_result.total_score, 3)}: {search_result.describe_content_compact()}"

        print(entries_log)
        print("\n")

        return results_no_duplicates

    def squash_nearby_results(self, original_results: List[SearchResult]) -> List[SearchResult]:
        output_results: List[SearchResult] = []
        last_squashed: SearchResult = None
        last_last_squashed: SearchResult = None
        for result in original_results:
            if last_squashed is not None and result.full_classname == last_squashed.full_classname:
                last_squashed.bm25_score = max(last_squashed.bm25_score, result.bm25_score)
                last_squashed.vector_score = max(last_squashed.vector_score, result.vector_score)
                last_squashed.rerank_score = max(last_squashed.rerank_score, result.rerank_score)
                last_squashed.calculate_total_score()
                last_squashed.merge_search_result(result.details)
            elif last_last_squashed is not None and result.full_classname == last_last_squashed.full_classname:
                last_last_squashed.bm25_score = max(last_last_squashed.bm25_score, result.bm25_score)
                last_last_squashed.vector_score = max(last_last_squashed.vector_score, result.vector_score)
                last_last_squashed.rerank_score = max(last_last_squashed.rerank_score, result.rerank_score)
                last_last_squashed.merge_search_result(result.details)
                last_last_squashed.calculate_total_score()
            else:
                last_last_squashed = last_squashed
                last_squashed = result
                output_results.append(last_squashed)
        return output_results
    
    def squash_duplicates(self, original_results: List[SearchResult]) -> List[SearchResult]:
        output_results: List[SearchResult] = []
        for item in original_results:
            already_in_output = False
            for existing_result in output_results:
                
                is_existing_better_or_equal = item.is_better_or_equal(existing_result)

                if is_existing_better_or_equal:
                    already_in_output = True
                    existing_result.bm25_score = max(existing_result.bm25_score, item.bm25_score)
                    existing_result.vector_score = max(existing_result.vector_score, item.vector_score)
                    existing_result.rerank_score = max(existing_result.rerank_score, item.rerank_score)
                    existing_result.calculate_total_score()

            if not already_in_output:
                output_results.append(item)

        return output_results

    def print_results(self, sorted_results: List[SearchResult], vector: bool):
        for idx, item in enumerate(sorted_results):
            print(f"{idx}. {round(item.vector_score if vector else item.bm25_score, 3)}: {item.describe_content_compact()}")

    def merge_hybrid_result(self, combined_results: List[SearchResult], result: SearchResult):
        already_present = False
        for existing_result in combined_results:
            is_existing_better_or_equal = result.is_better_or_equal(existing_result)

            if is_existing_better_or_equal:
                existing_result.vector_score = max(result.vector_score, existing_result.vector_score)
                existing_result.bm25_score = max(result.bm25_score, existing_result.bm25_score)
                existing_result.rerank_score = max(result.rerank_score, existing_result.rerank_score)
                existing_result.calculate_total_score()
                already_present = True

        if not already_present:
            combined_results.append(result)

    def is_equal(self, result: SearchResult, other_result: SearchResult):
        entry_equal = False
        if result.full_classname == other_result.full_classname:
            if len(result.details) == 0 and len(other_result.details) == 0:
                entry_equal = True

            if len(result.details) >= 1 and len(other_result.details) >= 1 and result.details[0].getName() == other_result.details[0].getName():
                entry_equal = True
        return entry_equal
    
    def rerank_results(self, sorted_results: List[SearchResult], query: str, rerank_limit: int = 100, rerank_result_limit = 30) -> List[SearchResult]:
        if self.reranker.isEnabled():
            results = sorted_results[:rerank_limit]

            query = f"{query}"
            print(f"\nReranking with query: '{query}'...")
            for idx, item in enumerate(results):
                if self.logger.level != logging.DEBUG:
                    print(f"{idx}. {item.describe_content_compact()}")
                else:
                    print(f"\n{idx}. {item.describe_content()}")

            docs_to_rerank = [res.describe_content_rerank() for res in results]
            
            instruction = "Given a user query, score documents of class summaries that are relevant to the query"
            rerank_scores = self.reranker.rerank(query, docs_to_rerank, instruction=instruction)
            
            for i, res in enumerate(results):      
                res.rerank_score = rerank_scores[i]      

            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

            results_limited = results[:rerank_result_limit]
            print("Final reranking results:")
            for idx, item in enumerate(results_limited):
                print(f"{idx}. {round(item.rerank_score, 3)}: {item.describe_content_compact()}")
            print("")

            return results_limited

        return sorted_results