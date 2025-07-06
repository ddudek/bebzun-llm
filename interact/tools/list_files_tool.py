import os
from interact.chat_state import ChatState

class ListFilesTool:
    name: str = "list_files"
    description: str = (
        "Lists all Kotlin (.kt) and Java (.java) files in the project directory. E.g.: `<list_files></list_files>`")
    
    def __init__(self, base_path: str, source_dirs: list[str]):
        self.base_path = base_path
        self.source_dirs = source_dirs

    def run(self, chat_state: ChatState, **kwargs) -> str:
        """
        List all files in the project directory recursively.
        The path parameter is ignored as this tool always lists all files.
        """
        print(f"\n---- ListFilesTool Debug Info ----")
        print(f"Base path: '{self.base_path}'")
        print(f"Source directories: {self.source_dirs}")
        
        try:
            files = []
            for src_dir in self.source_dirs:
                abs_src_dir = os.path.join(self.base_path, src_dir)
                for root, dirs, filenames in os.walk(abs_src_dir):
                    rel_path = os.path.relpath(root, self.base_path)
                    if rel_path == '.':
                        rel_path = ''
                        
                    for filename in filenames:
                        if filename.endswith(('.kt', '.java')):
                            file_path = os.path.join(rel_path, filename)
                            files.append(file_path)
            
            if not files:
                print(f"No files found in source directories: {self.source_dirs}")
                return "No files found in the project directory."
                
            print(f"Found {len(files)} files")
            print(f"---- End Debug Info ----\n")
            return "\n".join(sorted(files))
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"---- End Debug Info ----\n")
            return f"Error listing files: {str(e)}"