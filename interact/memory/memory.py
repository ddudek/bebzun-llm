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
        self.max_memory_size: int = tokens_to_chars(51250)

    def add_search_result(self, search_result: SearchResult, rel_path: str, file_size: int):
        for i in self.items:
            if  isinstance(i, MemoryItemClassSummary):
                if i.search_result.full_classname == search_result.full_classname:
                    i.search_result.merge_search_result(search_result.details)
                    return
            
        self.items.append(MemoryItemClassSummary(class_name=search_result.full_classname, search_result=search_result, rel_path=rel_path, file_size=file_size))

    def remove_class_memory(self, query: str):
        removed = []
        for i in self.items:
            if  isinstance(i, MemoryItemClassSummary):
                if query.lower() in i.class_name.lower():
                    removed.append(i)
        
        for i in removed:
            self.items.remove(i)

        return [r.class_name for r in removed]

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
                
        memory_classes = self.items.copy()
        for i in memory_classes:
            if  isinstance(i, MemoryItemClassSummary):
                if file_path in i.search_result.file:
                    self.items.remove(i)
                
        dependencies: Dict[str, DepUsage] = self.knowledge_store.get_file_dependencies(file_path)
        dependencies_filtered = dependencies.copy()
        for dep in dependencies.values():
            for i in self.items:
                if isinstance(i, MemoryItemClassSummary) and i.search_result.full_classname == dep.dependency_structure.full_classname:
                    print(f"--- Ignoring dependency already in mem: {dep.dependency_structure.full_classname}")
                    del dependencies_filtered[dep.dependency_structure.full_classname]

        usages: Dict[str, DepUsage] = self.knowledge_store.get_file_usages(file_path)
        file_context = self.knowledge_store.prepare_file_context(
                                                base_dir,
                                                file_path, 
                                                dependencies_filtered,
                                                usages,
                                                file_content,
                                                True,
                                                False,
                                                False,
                                                False,
                                                6)
        
        file_context_size = chars_to_tokens(len(file_context))
        if file_context_size > 15000:
            file_context = self.knowledge_store.prepare_file_context(
                                                base_dir,
                                                file_path, 
                                                dependencies_filtered,
                                                usages,
                                                file_content,
                                                False,
                                                False,
                                                False,
                                                False,
                                                2)
            print(f"--- Limiting context for {file_path}: {file_context_size} -> {chars_to_tokens(len(file_context))}")

        
        self.items.append(MemoryItemFullFile(file_path=file_path, file_content=file_content, file_context=file_context))

    def get_formatted_memory(self) -> str:
        memory_size: int = 0
        
        memory_content = f"# Your current knowledge memory (items gathered so far):\n"
        
        if not self.items:
            memory_content += "- No items\n"

        other_files = set()
        other_files_already_covered = set()

        for idx, item in enumerate(self.items):
            if  isinstance(item, MemoryItemClassSummary):
                class_summary = item.search_result.describe_content()
                class_fullname = item.class_name
                file_path = item.rel_path
                file_size = item.file_size
                file_info = f"\nFile with implementation: `{file_path}` (size: {chars_to_tokens(file_size)} tokens)"
                raw_content = f"Item #{idx}: Class summary for: {class_fullname}:{file_info}\n{class_summary}\n\n"
                other_files_already_covered.add(file_path)
            
            if isinstance(item, MemoryItemFullFile):
                raw_content = f"Item #{idx}: File content: {item.file_path}\n```\n{item.file_content}\n```\n{item.file_context}\n"
                
                other_files_already_covered.add(item.file_path)
                deps = self.knowledge_store.get_file_dependencies(item.file_path)
                usages = self.knowledge_store.get_file_usages(item.file_path)
                for i in deps.values():
                    if i.dependency_description:
                        other_files.add(i.dependency_description.file)

                for i in usages.values():
                    if i.parent_description:
                        other_files.add(i.parent_description.file)
                
            memory_content += raw_content

            memory_size += len(raw_content)

        if other_files:
            memory_content += "\n"
            files = list(other_files - other_files_already_covered)
            files.sort(key=lambda x: x, reverse=True)
            other_files_formatted = "\n".join(files) 
            memory_content += "Other existing files:\n" + other_files_formatted + "\n\n"

            memory_size += len(other_files_formatted)
        
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
                raw_content = f"- Item #{idx}: Full file: `{item.file_path}` ({chars_to_tokens(len(item.file_content))} tk + {chars_to_tokens(len(item.file_context))} tk context)"
                
            memory_content += raw_content
            memory_content += "\n"
            memory_size += len(raw_content)
        
        return memory_content

    def is_empty(self) -> bool:
        return not self.items