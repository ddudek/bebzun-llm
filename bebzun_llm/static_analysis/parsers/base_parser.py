"""
Abstract base class for language-specific parsers.

This module defines the common interface that all parser implementations must follow,
ensuring consistent behavior across different language parsers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Tuple

from bebzun_llm.static_analysis.model.model import ClassStructure

class BaseParser(ABC):
    """
    Abstract base class for language-specific parsers.
    
    Defines the interface that all parser implementations must follow.
    """
    

    @abstractmethod
    def extract_classes(self, file_path: Path, input_dir: Path, version: int) -> List[ClassStructure]:
        """
        Parse a source file and extract class information.
        
        Args:
            file_path: Path to the source file to parse
            input_dir: Path to the root of the codebase
            
        Returns:
            List of ClassSummaryOutput objects representing classes found in the file
        """
        pass

    
    @abstractmethod
    def extract_dependencies(
        self, 
        file_content: str, 
        cls: ClassStructure, 
        known_classes: Set[str]
    ) -> List[Tuple[str, str, List[int]]]:
        """
        Extract dependencies from a class.
        
        Args:
            file_content: Content of the source file
            cls: class to extract dependencies for
            known_classes: Set of all known class names in the codebase
            
        Returns:
            List of tuples (simple_name, full_name, usage_lines) for each dependency
        """
        pass
    
    @abstractmethod
    def extract_methods(
        self, 
        file_content: str, 
        class_name: str
    ) -> List[Tuple[str, int, int]]:
        """
        Extract public methods from a class.
        
        Args:
            file_content: Content of the source file
            class_name: Name of the class to extract methods for
            
        Returns:
            List of tuples (method_name, start_line, end_line) for each public method
        """
        pass