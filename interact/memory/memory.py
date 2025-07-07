import json
from typing import Dict
from knowledge.model import ClassDescriptionExtended
from math import trunc

def chars_to_tokens(chars: int) -> int:
    return trunc(chars / 4.8)

def tokens_to_chars(tokens: int) -> int:
    return trunc(tokens * 4.8)

class Memory:
    def __init__(self):
        self.classes: Dict[str, ClassDescriptionExtended] = {}
        self.files: Dict[str, str] = {}
        self.max_memory_size: int = tokens_to_chars(32768)

    def add_class(self, class_name: str, class_data: ClassDescriptionExtended):
        self.classes[class_name] = class_data

    def add_file(self, file_path: str, file_content: str):
        self.files[file_path] = file_content

    def update_memory(self, memory_update: Dict[str, any]):
        if "classes" in memory_update:
            self.classes.update(memory_update["classes"])
        if "files" in memory_update:
            self.files.update(memory_update["files"])

    def get_formatted_memory(self, step_number: int) -> str:
        memory_size: int = 0
        if (step_number <= 1):
            step_message = "current step"
        elif (step_number == 2):
            step_message = "step 1 and current step"
        else:
            step_message = "step 1 and step 2"
        memory_content = f"# Your current knowledge memory (items gathered in {step_message}):\n"
        if not self.classes and not self.files:
            memory_content += "- No items\n"
        if self.classes:
            memory_content += "## Class Summaries\n"
            
            raw_content = ""
            for idx, class_summary in enumerate(self.classes.values()):
                file_path = class_summary.file
                raw_content += f"\nItem #{idx}: Class: {class_summary.class_summary.full_classname}\nFile with implementation: {file_path}\n{class_summary.class_summary.describe()}\n"
            
            memory_content += raw_content
            memory_content += "\n"
            memory_size += len(raw_content)
        if self.files:
            memory_content += "## File Contents\n"
            raw_content = ""
            for idx, (file_path, file_content) in enumerate(self.files.items()):
                raw_content += f"\n Item #{len(self.classes) + idx}: File: {file_path}\n```\n{file_content}\n```\n"
            memory_content += raw_content
            memory_size += len(raw_content)
        
        memory_content += f"End of knowledge memory. Size: {chars_to_tokens(memory_size)} tokens /{chars_to_tokens(self.max_memory_size)} tokens max.\n"
        
        return memory_content

    def is_empty(self) -> bool:
        return not self.classes and not self.files