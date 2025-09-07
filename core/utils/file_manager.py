import os
from typing import List, Optional
from logging import Logger
from knowledge.model import FileInfo
from core.config.config import Config
from core.utils.file_utils import format_file_size


class FileManager:
    def __init__(self):
        self.base_dir: str = ""
        self.file_infos: List[FileInfo] = []
        self.source_dirs: List[str] = []
        self.exclude: List[str] = []
        self.extensions: tuple[str] = ()
        self.is_filtering_enabled = False

    def load(self, logger: Logger, base_dir: str, config: Config, name_filter: str):
        self.base_dir = base_dir
        self.is_filtering_enabled = name_filter != None

        invert_filter = False
        filter_text = name_filter
        if name_filter and name_filter.startswith("!"):
            invert_filter = True
            filter_text = name_filter[1:]

        self.source_dirs = config.source_dirs
        self.extensions = config.file_extensions
        self.exclude = config.exclude

        for src_dir in self.source_dirs:
            abs_src_path = os.path.join(base_dir, src_dir)
            if not os.path.exists(abs_src_path):
                logger.warning(f"The source directory '{abs_src_path}' does not exist.")
                continue
                
            for root, _, files in os.walk(abs_src_path):
                for file_path in files:
                    if file_path.endswith(self.extensions):
                        abs_file_path = os.path.join(root, file_path)
                        rel_file_path = os.path.relpath(abs_file_path, base_dir)
                        file_size = os.path.getsize(abs_file_path)
                        version = file_size # will be enough for now

                        if self.is_excluded(file_path):
                            continue
                        
                        is_allowed = True
                        if filter_text:
                            contains_filter = filter_text in file_path
                            is_allowed = not contains_filter if invert_filter else contains_filter
                        
                        file_info = FileInfo(
                            filepath=rel_file_path,
                            file_size=file_size,
                            version=version,
                            is_allowed_by_filter=is_allowed
                        )

                        self.file_infos.append(file_info)

    def is_allowed(self, rel_path: str):
        # Security check: ensure the requested path is within one of the allowed source directories
        full_path = os.path.normpath(os.path.join(self.base_dir, rel_path))
        abs_path = os.path.abspath(full_path)
        is_allowed = False
        for src_dir in self.source_dirs:
            abs_src_dir = os.path.abspath(os.path.join(self.base_path, src_dir))
            if abs_path.startswith(abs_src_dir) and not self.is_excluded(rel_path):
                is_allowed = True          
                break

        return is_allowed

    def print_files_info(self, logger: Logger):
        logger.info("Filtered files found:")
        for i, file_info in enumerate(self.file_infos, 1):
            formatted_size = format_file_size(file_info.file_size)
            status = "[WILL PROCESS]" if file_info.is_allowed_by_filter else "[FILTERED OUT]"
            logger.info(f"{i}. {file_info.filepath} ({formatted_size}) {status}")

        allowed_count = sum(1 for f in self.file_infos if f.is_allowed_by_filter)
        logger.info(f"Total: {len(self.file_infos)} files found, {allowed_count} match filter")

    def get_all_supported_files(self, only_filtered: bool) -> List[str]:
        return [f.filepath for f in self.file_infos if (not only_filtered) or f.is_allowed_by_filter]
    
            
    def list_all_files(self, base_dir: str) -> List[str]:
        """
        Get a list of all files in the specified source directories, ignoring specified directories and files.
        """
        all_files = []
        for src_dir in self.source_dirs:
            abs_src_path = os.path.join(base_dir, src_dir)
            if not os.path.isdir(abs_src_path):
                continue

            for root, _, files in os.walk(abs_src_path, topdown=True):
                # Exclude ignored directories from traversal
                
                for file_path in files:
                    # Exclude ignored files
                    if self.is_excluded(file_path):
                        continue  
                    
                    rel_path = os.path.relpath(os.path.join(root, file_path), base_dir)
                    all_files.append(rel_path)

        return all_files

    def is_excluded(self, file_path):
        excluded = False
        for excl in self.exclude:
            if excl in file_path:
                excluded = True
        return excluded