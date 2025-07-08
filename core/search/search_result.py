from dataclasses import dataclass, field
from typing import Dict, Any, List
from knowledge.embeddings import EmbeddingEntry

@dataclass
class SearchResult:
    item: EmbeddingEntry
    vector_score: float = 0.0
    bm25_score: float = 0.0
    total_score: float = 0.0

@dataclass
class CombinedSearchResults:
    results: Dict[str, SearchResult] = field(default_factory=dict)

    def add_vector_result(self, item: EmbeddingEntry, score: float):
        if item.full_classname not in self.results:
            self.results[item.full_classname] = SearchResult(item=item)
        self.results[item.full_classname].vector_score = score

    def add_bm25_result(self, item: EmbeddingEntry, score: float):
        if item.full_classname not in self.results:
            self.results[item.full_classname] = SearchResult(item=item)
        self.results[item.full_classname].bm25_score = score

    def get_sorted_results(self) -> List[SearchResult]:
        for result in self.results.values():
            result.total_score = result.vector_score + result.bm25_score
        
        return sorted(self.results.values(), key=lambda x: x.total_score, reverse=True)