from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from core.search.embedding_entry import EmbeddingEntry
from knowledge.model import ClassDescription, MethodDescription, VariableDescription

@dataclass
class SearchResult:
    entry: EmbeddingEntry
    details: List[MethodDescription | VariableDescription]
    class_description: Optional[ClassDescription] = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    total_score: float = 0.0

    def describe_content(self) -> str:
        content = f"{self.class_description.summary}"
        for detail in self.details:
            if isinstance(detail, MethodDescription):
                content += f"\n Method: `{detail.method_name}`: {detail.method_summary}"
            if isinstance(detail, VariableDescription):
                content += f"\n Property: `{detail.property_name}`: {detail.property_summary}"
        return content
    
    def add_detail(self, item: EmbeddingEntry, class_info: ClassDescription):

        if not class_info:
            print (f"Warn: no description found for embedding entry: {item.full_classname}")

        methods = [method.method_name for method in self.details if isinstance(method, MethodDescription)]
        properties = [property.property_name for property in self.details if isinstance(property, VariableDescription)]

        if item.type == 'method':
            detail = class_info.find_method(item.detail)
            if detail and detail.method_name not in methods:
                self.details.append(detail)
            
        if item.type == 'property':
            detail = class_info.find_property(item.detail)
            if detail and detail.property_name not in properties:
                self.details.append(detail)

@dataclass
class CombinedSearchResults:
    results: Dict[str, SearchResult] = field(default_factory=dict)

    def add_detail(self, item: EmbeddingEntry, class_info: ClassDescription):

        if not class_info:
            print (f"Warn: no description found for embedding entry: {item.full_classname}")

        if item.full_classname not in self.results:
            result = SearchResult(entry=item, details=[], class_description=class_info)
            self.results[item.full_classname] = result
        else:
            result = self.results[item.full_classname]

        result.add_detail(item, class_info)

    def add_vector_result(self, item: EmbeddingEntry, score: float):
        if item.full_classname not in self.results:
            self.results[item.full_classname] = SearchResult(entry=item, details=[])
        self.results[item.full_classname].vector_score += score

    def add_bm25_result(self, item: EmbeddingEntry, score: float):
        if item.full_classname not in self.results:
            self.results[item.full_classname] = SearchResult(entry=item, details=[])
        self.results[item.full_classname].bm25_score += score

    def add_rerank_result(self, item: EmbeddingEntry, score: float):
        if item.full_classname not in self.results:
            self.results[item.full_classname] = SearchResult(entry=item, details=[])
        self.results[item.full_classname].rerank_score = +score

    def get_sorted_results(self) -> List[SearchResult]:
        for result in self.results.values():
            result.total_score = result.vector_score + result.bm25_score
        
        return sorted(self.results.values(), key=lambda x: x.total_score, reverse=True)
    
    def get_sorted_by_rerank_results(self) -> List[SearchResult]:
        return sorted(self.results.values(), key=lambda x: x.rerank_score, reverse=True)