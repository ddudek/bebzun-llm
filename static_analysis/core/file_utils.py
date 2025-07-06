"""
File system utilities for the Kotlin and Java analyzer.

This module provides functionality for file system operations, such as finding
Java and Kotlin source files in a directory.
"""

import os
from pathlib import Path
from typing import Generator, List, Set


def find_source_files(
    root_dirs: List[str], input_dir: str, extensions: Set[str] = {'.java', '.kt'}
) -> Generator[Path, None, None]:
    """
    Find all source files with specified extensions in a directory tree.
    
    Args:
        root_dirs: Root directories to search in
        input_dir: The root of the codebase, used for making relative paths
        extensions: Set of file extensions to include
        
    Yields:
        Path objects for each matching source file
    """
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        
        if not root_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_dir}")
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in extensions:
                    yield file_path


def get_java_files(root_dirs: List[str], input_dir: str) -> List[Path]:
    """
    Get all Java source files in a directory tree.
    
    Args:
        root_dirs: Root directories to search in
        input_dir: The root of the codebase, used for making relative paths
        
    Returns:
        List of Path objects for Java source files
    """
    return list(find_source_files(root_dirs, input_dir, {'.java'}))


def get_kotlin_files(root_dirs: List[str], input_dir: str) -> List[Path]:
    """
    Get all Kotlin source files in a directory tree.
    
    Args:
        root_dirs: Root directories to search in
        input_dir: The root of the codebase, used for making relative paths
        
    Returns:
        List of Path objects for Kotlin source files
    """
    return list(find_source_files(root_dirs, input_dir, {'.kt'}))


def get_all_source_files(root_dirs: List[str], input_dir: str) -> List[Path]:
    """
    Get all Java and Kotlin source files in a directory tree.
    
    Args:
        root_dirs: Root directories to search in
        input_dir: The root of the codebase, used for making relative paths
        
    Returns:
        List of Path objects for all source files
    """
    return list(find_source_files(root_dirs, input_dir))