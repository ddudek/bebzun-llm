import os
import re
from typing import Dict, Any, Tuple
from interact.memory.memory import Memory
from interact.chat_state import ChatState

class GetFileContentTool:
    name: str = "read_file"
    description: str = (
        "Gets the content of a specific file in the specified directory.\n"
        "Parameters:\n"
        "- path (required): Provide a relative path to the file.\n"
        "Example:\n"
        "```\n"
        "<read_file>\n"
        "<path>com/example/data/ExampleClass.kt</path>\n"
        "</read_file>\n"
        "```"
        )
    
    def __init__(self, base_path: str, source_dirs: list[str]):
        self.base_path = base_path
        self.source_dirs = [os.path.normpath(d) for d in source_dirs]

    
    def run(self, chat_state: ChatState, path: str = "") -> str:
        """
        Get the content of a specific file in the specified directory.
        Provide a relative path to the file.
        """
        print(f"\n---- GetFileContentTool Debug Info ----")
        print(f"Input path parameter (raw): '{path}'")
        
        path = self._clean_path_input(path)
        print(f"Input path parameter (cleaned): '{path}'")
        
        print(f"Base path: '{self.base_path}'")
        
        try:
            if not path:
                print(f"Empty path provided")
                return "Error: Please provide a file path.", {}
            
            original_path = path
            path = path.lstrip("/")
            if original_path != path:
                print(f"Removed leading slashes: '{original_path}' -> '{path}'")
                
            full_path = os.path.normpath(os.path.join(self.base_path, path))
            print(f"Resolved full path: '{full_path}'")
            
            # Security check: ensure the requested path is within one of the allowed source directories
            abs_path = os.path.abspath(full_path)
            is_allowed = False
            for src_dir in self.source_dirs:
                abs_src_dir = os.path.abspath(os.path.join(self.base_path, src_dir))
                if abs_path.startswith(abs_src_dir):
                    is_allowed = True
                    break
            
            if not is_allowed:
                print(f"Safety check failed: Path is not within allowed source directories")
                return f"Error: Cannot access path outside of the allowed source directories. Please use a relative path from one of: {self.source_dirs}", {}
            
            if not os.path.exists(full_path):
                print(f"Path does not exist: '{full_path}'")
                return f"Error: The file '{path}' does not exist within the base directory.", {}
                
            if not os.path.isfile(full_path):
                print(f"Path is not a file: '{full_path}'")
                return f"Error: The path '{path}' is not a file.", {}
                
            with open(full_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                
            chat_state.get_file_used_count += 1
            print(f"Successfully read file: '{full_path}'")
            print(f"---- End Debug Info ----\n")
            
            observation = f"- File added to the memory: '{path}'"
            memory_update = {"files": {path: content}}
            chat_state.memory.update_memory(memory_update)
            
            return observation
            
        except UnicodeDecodeError:
            print(f"Unicode decode error: File may be binary")
            print(f"---- End Debug Info ----\n")
            return f"Error: The file '{path}' appears to be a binary file and cannot be displayed as text.", {}
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"---- End Debug Info ----\n")
            return f"Error reading file: {str(e)}", {}
    
    def _clean_path_input(self, path: str) -> str:
        path = re.sub(r'[\n`]+$', '', path)
        path = re.sub(r'\n```$', '', path)
        path = path.strip()
        return path