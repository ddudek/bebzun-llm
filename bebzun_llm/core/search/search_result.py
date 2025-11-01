from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from bebzun_llm.core.search.embedding_entry import EmbeddingEntry
from bebzun_llm.knowledge.model import ClassDescription, MethodDescription, VariableDescription

@dataclass
class SearchResult:
    full_classname: str
    file: str
    details: List[MethodDescription | VariableDescription]
    class_description: Optional[ClassDescription] = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    total_score: float = 0.0

    def describe_content_rerank(self) -> str:
        return self.describe_content()

    def describe_content(self) -> str:
        content = f"{self.class_description.summary}"
        for detail in self.details:
            if isinstance(detail, MethodDescription):
                content += f"\n{self.class_description.simple_classname} contains method: `{detail.method_name}`: {detail.method_summary}."
            if isinstance(detail, VariableDescription):
                content += f"\n{self.class_description.simple_classname} contains property: `{detail.property_name}`: {detail.property_summary}."
        return content
    
    def describe_content_compact(self) -> str:
        details = ",".join(['`'+detail.getName()+'`' for detail in self.details])
        return f"{self.full_classname}" + (f" [{details}]" if details else "")

    def add_detail_embedding(self, item: EmbeddingEntry, class_info: ClassDescription):        
        self.add_detail(item.full_classname, item.type, item.detail, class_info)

    def add_detail(self, classname, type, detail, class_info: ClassDescription):

        if not class_info:
            print (f"Warn: no description found for embedding entry: {classname}")

        methods = [method.method_name for method in self.details if isinstance(method, MethodDescription)]
        properties = [property.property_name for property in self.details if isinstance(property, VariableDescription)]

        if type == 'method':
            detail = class_info.find_method(detail)
            if detail and detail.method_name not in methods:
                self.details.append(detail)
            
        if type == 'property':
            detail = class_info.find_property(detail)
            if detail and detail.property_name not in properties:
                self.details.append(detail)

    def merge_search_result(self, details: List):
        for detail in details:
            if isinstance(detail, MethodDescription):
                self.add_detail(self.full_classname, "method", detail.method_name, self.class_description)
            if isinstance(detail, VariableDescription):
                self.add_detail(self.full_classname, "property", detail.property_name, self.class_description)

    def calculate_total_score(self):
        self.total_score = self.vector_score + self.bm25_score
        # if both searches found the same result, average it, but with a small advantage
        if self.vector_score > 0.1 and self.bm25_score > 0.1:
            self.total_score = max(self.vector_score, self.bm25_score, (self.vector_score + self.bm25_score) * 0.55)

    def is_better_or_equal(self, existing_result:"SearchResult"):
        is_existing_better_or_equal = False
        if existing_result.full_classname == self.full_classname:
            if len(self.details) == 0:
                is_existing_better_or_equal = True
            elif len(self.details) > 0 and len(existing_result.details) >= len(self.details):
                contains_all_details = True
                for detail in self.details:
                    if detail.getName() not in [detail.getName() for detail in existing_result.details]:
                        contains_all_details = False
                if contains_all_details:
                    is_existing_better_or_equal = True
            else:
                is_existing_better_or_equal = True
        return is_existing_better_or_equal

@dataclass
class CombinedSearchResults:
    results: Dict[str, SearchResult] = field(default_factory=dict)

    def merge_search_result(self, search_result: SearchResult, class_info: ClassDescription):

        if not class_info:
            print (f"Warn: no description found for embedding entry: {search_result.full_classname}")

        if search_result.full_classname not in self.results:
            search_result.class_description = class_info
            self.results[search_result.full_classname] = search_result
            result = search_result
        else:
            result = self.results[search_result.full_classname]
            result.merge_search_result(search_result.details)
                        
    def add_vector_result(self, full_classname: str, score: float):
        if full_classname not in self.results:
            return
        
        prev_score = self.results[full_classname].vector_score
        self.results[full_classname].vector_score = max(prev_score, score)

    def add_bm25_result(self, full_classname: str, score: float):
        if full_classname not in self.results:
            return
        
        prev_score = self.results[full_classname].bm25_score
        self.results[full_classname].bm25_score += max(prev_score, score)

    def add_rerank_result(self, full_classname: str, score: float):
        if full_classname not in self.results:
            return
        
        prev_score = self.results[full_classname].rerank_score
        self.results[full_classname].rerank_score = max(prev_score, score)

    def get_sorted_results(self) -> List[SearchResult]:
        for result in self.results.values():
            result.total_score = result.vector_score + result.bm25_score
        
        return sorted(self.results.values(), key=lambda x: x.total_score, reverse=True)
    
    def get_sorted_by_rerank_results(self) -> List[SearchResult]:
        return sorted(self.results.values(), key=lambda x: x.rerank_score, reverse=True)