"""
Output formatter for the Kotlin and Java analyzer.

This module provides functionality for formatting analysis results as JSON
and writing them to a file.
"""

import json
from pathlib import Path
from typing import Dict, List

from bebzun_llm.static_analysis.model.model import FileStructure, ClassStructure


def format_as_json(analysis_output: FileStructure) -> str:
    """
    Format analysis output as a JSON string.
    
    Args:
        analysis_output: Analysis output to format
        
    Returns:
        JSON string representation of the analysis output
    """
    return analysis_output.model_dump_json(indent=2)


def write_json_to_file(analysis_output: FileStructure, output_path: Path) -> None:
    """
    Write analysis output to a JSON file.
    
    Args:
        analysis_output: Analysis output to write
        output_path: Path to write the output to
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the JSON output
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(format_as_json(analysis_output))
    
    print(f"Analysis output written to {output_path}")


def format_dependencies_for_output(
    dependencies: List[tuple[str, str, List[int]]]
) -> List[Dict]:
    """
    Format dependencies for output.
    
    Args:
        dependencies: List of dependencies as (simple_name, full_name, usage_lines) tuples
        
    Returns:
        List of formatted dependency dictionaries
    """
    return [
        {
            "simple_classname": simple_name,
            "full_classname": full_name,
            "usage_lines": usage_lines
        }
        for simple_name, full_name, usage_lines in dependencies
    ]


def format_methods_for_output(
    methods: List[tuple[str, int, int]]
) -> List[Dict]:
    """
    Format methods for output.
    
    Args:
        methods: List of methods as (name, start_line, end_line) tuples
        
    Returns:
        List of formatted method dictionaries
    """
    return [
        {
            "name": name,
            "definition_start": start_line,
            "definition_end": end_line
        }
        for name, start_line, end_line in methods
    ]