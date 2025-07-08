import json
from typing import List, Union
from knowledge.model import ClassDescriptionExtended
from math import trunc
from dataclasses import dataclass

def chars_to_tokens(chars: int) -> int:
    return trunc(chars / 4.8)

def tokens_to_chars(tokens: int) -> int:
    return trunc(tokens * 4.8)

@dataclass
class MemoryItemFullFile:
    file_path: str
    file_content: str

@dataclass
class MemoryItemClassSummary:
    class_name: str
    class_data: ClassDescriptionExtended

@dataclass
class MemoryItemDependency:
    class_name: str
    class_data: ClassDescriptionExtended

class Memory:
    def __init__(self):
        self.items: List[Union[MemoryItemFullFile, MemoryItemClassSummary]] = []
        self.max_memory_size: int = tokens_to_chars(32768)

    def add_class(self, class_name: str, class_data: ClassDescriptionExtended):
        self.items.append(MemoryItemClassSummary(class_name=class_name, class_data=class_data))

    def add_file(self, file_path: str, file_content: str):
        self.items.append(MemoryItemFullFile(file_path=file_path, file_content=file_content))

    def get_formatted_memory(self, step_number: int) -> str:
        memory_size: int = 0
        if (step_number <= 1):
            step_message = "current step"
        elif (step_number == 2):
            step_message = "step 1 and current step"
        else:
            step_message = "step 1 and step 2"
        memory_content = f"# Your current knowledge memory (items gathered in {step_message}):\n"
        
        if not self.items:
            memory_content += "- No items\n"

        for idx, item in enumerate(self.items):
            if  isinstance(item, MemoryItemClassSummary):
                class_summary = item.class_data
                file_path = class_summary.file
                raw_content = f"Item #{idx}: Class summary: {class_summary.class_summary.full_classname}\nFile with implementation: {file_path}\n{class_summary.class_summary.describe()}\n"
            
            if isinstance(item, MemoryItemFullFile):
                raw_content = f"Item #{idx}: File: {item.file_path}\n```\n{item.file_content}\n```\n"
                
            memory_content += raw_content
            memory_content += "\n"
            memory_size += len(raw_content)
        
        memory_content += f"End of knowledge memory. Size: {chars_to_tokens(memory_size)} tokens /{chars_to_tokens(self.max_memory_size)} tokens max.\n"
        
        return memory_content

    def is_empty(self) -> bool:
        return not self.items