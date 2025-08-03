import json
import os
from typing import Dict, List, Union
from core.utils.file_utils import get_file_content_safe
from knowledge.knowledge_store import DepUsage, KnowledgeStore
from knowledge.model import ClassDescriptionExtended
from static_analysis.model.model import ClassStructure
from math import trunc
from dataclasses import dataclass
from core.search.search_result import SearchResult
from knowledge.model import ClassDescription, MethodDescription, VariableDescription

def chars_to_tokens(chars: int) -> int:
    return trunc(chars / 4.8)

def tokens_to_chars(tokens: int) -> int:
    return trunc(tokens * 4.8)

@dataclass
class MemoryItemFullFile:
    file_path: str
    file_content: str
    file_context: str

@dataclass
class MemoryItemClassSummary:
    class_name: str
    rel_path: str
    search_result: SearchResult
    file_size: int

@dataclass
class MemoryItemDependency:
    class_name: str
    class_data: ClassDescriptionExtended

class Memory:
    def __init__(self, knowledge_store: KnowledgeStore):
        self.knowledge_store = knowledge_store
        self.items: List[Union[MemoryItemFullFile, MemoryItemClassSummary]] = []
        self.max_memory_size: int = tokens_to_chars(32768)

    def add_search_result(self, search_result: SearchResult, rel_path: str, file_size: int):
        for i in self.items:
            if  isinstance(i, MemoryItemClassSummary):
                if i.search_result.full_classname == search_result.full_classname:
                    i.search_result.merge_search_result(search_result.details)
                    return
            
        self.items.append(MemoryItemClassSummary(class_name=search_result.full_classname, search_result=search_result, rel_path=rel_path, file_size=file_size))

    def remove_class_memory(self, class_name: str):
        for i in self.items:
            if  isinstance(i, MemoryItemClassSummary):
                if class_name in i.search_result.class_description.full_classname:
                    self.items.remove(i)
                    return

    def remove_file_memory(self, file_name: str):
        for i in self.items:
            if  isinstance(i, MemoryItemFullFile):
                if file_name in i.file_path:
                    self.items.remove(i)
                    return

    def add_file(self, file_path: str, file_content: str, base_dir: str):
        for i in self.items:
            if  isinstance(i, MemoryItemFullFile):
                if file_path in i.file_path:
                    # file is already in memory
                    return
                
        dependencies: Dict[str, DepUsage] = self.knowledge_store.get_file_dependencies(file_path)
        usages: Dict[str, DepUsage] = self.knowledge_store.get_file_usages(file_path)
        file_context = self.knowledge_store.prepare_file_context(
                                                base_dir,
                                                file_path, 
                                                dependencies,
                                                usages,
                                                file_content,
                                                True,
                                                False,
                                                False,
                                                False,
                                                8)
        
        self.items.append(MemoryItemFullFile(file_path=file_path, file_content=file_content, file_context=file_context))

    def get_formatted_memory(self) -> str:
        memory_size: int = 0
        
        memory_content = f"# Your current knowledge memory (items gathered so far):\n"
        
        if not self.items:
            memory_content += "- No items\n"

        for idx, item in enumerate(self.items):
            if  isinstance(item, MemoryItemClassSummary):
                class_summary = item.search_result.describe_content()
                class_fullname = item.class_name
                file_path = item.rel_path
                file_size = item.file_size
                file_info = f"\nFile with implementation: `{file_path}` (size: {chars_to_tokens(file_size)} tokens)"
                raw_content = f"Item #{idx}: Class summary for: {class_fullname}:{file_info}\n{class_summary}\n"
            
            if isinstance(item, MemoryItemFullFile):
                raw_content = f"Item #{idx}: File: {item.file_path}\n```\n{item.file_content}\n```\n{item.file_context}\n"
                
            memory_content += raw_content
            memory_content += "\n"
            memory_size += len(raw_content)
        
        memory_content += f"End of knowledge memory. Size: {chars_to_tokens(memory_size)} tokens /{chars_to_tokens(self.max_memory_size)} tokens max.\n"
        
        return memory_content
    
    def get_formatted_memory_compact(self) -> str:
        if not self.items:
            return "- No items\n"
        
        memory_size: int = 0
        memory_content = f""

        for idx, item in enumerate(self.items):
            if  isinstance(item, MemoryItemClassSummary):
                raw_content = f"- Item #{idx}: Class summary: `{item.class_name}`"
            
            if isinstance(item, MemoryItemFullFile):
                raw_content = f"- Item #{idx}: Full file: `{item.file_path}` ({chars_to_tokens(len(item.file_content))} tk)"
                
            memory_content += raw_content
            memory_content += "\n"
            memory_size += len(raw_content)
        
        return memory_content

    def is_empty(self) -> bool:
        return not self.items