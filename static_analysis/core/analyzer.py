"""
Main analyzer module for Kotlin and Java code analysis.

This module provides the core functionality for analyzing Kotlin and Java
codebases, including orchestration of parsing, dependency analysis, and output generation.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Set

from static_analysis.core.file_utils import get_all_source_files, get_java_files, get_kotlin_files
from static_analysis.core.output_formatter import format_dependencies_for_output, format_methods_for_output
from static_analysis.model.model import FileStructure, ClassStructure, ClassStructureDependency, ClassStructureMethod
from static_analysis.parsers.java_parser import JavaParser
from static_analysis.parsers.kotlin_parser import KotlinParser

class CodebaseAnalyzer:
    """
    Main analyzer class for Kotlin and Java codebases.
    
    This class orchestrates the analysis process, including parsing source files,
    analyzing dependencies, and generating output.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the analyzer with parsers."""
        self.verbose = verbose
        self.java_parser = JavaParser(verbose=self.verbose)
        self.kotlin_parser = KotlinParser(verbose=self.verbose)
    
    def analyze_codebase(self, source_dirs: List[str], input_dir: str, output_file: str) -> FileStructure:
        """
        Analyze a Kotlin and Java codebase.
        
        Args:
            source_dirs: Input directories containing source files
            input_dir: The root of the codebase, used for making relative paths
            output_file: Output file path for the analysis results
            
        Returns:
            AnalysisOutput object containing the analysis results
        """
        print(f"Starting analysis of files in:\n" + "\n".join(source_dirs))
        start_time = time.time()
        
        # Find all source files
        print("Finding source files...")
        java_files = get_java_files(source_dirs, input_dir)
        kotlin_files = get_kotlin_files(source_dirs, input_dir)
        
        print(f"Found {len(java_files)} Java files and {len(kotlin_files)} Kotlin files.")
        
        # Initial pass: extract class information
        print("Extracting class information...")
        all_classes = self._extract_all_classes(java_files, kotlin_files, Path(input_dir))
        print(f"Found {len(all_classes)} class.")
        
        # Build a set of known class names for dependency analysis
        known_classes = {cls.simple_classname for cls in all_classes}
        
        # Second pass: analyze dependencies and methods
        print("Analyzing dependencies and methods...")
        enriched_classes = self._analyze_dependencies_and_methods(
            all_classes, java_files, kotlin_files, known_classes, input_dir
        )
        
        # Create output object for return value
        analysis_output = FileStructure(classes=enriched_classes)

        # Prepare data structure for serialization
        metadata = {
            "source_directories": [str(Path(s).relative_to(input_dir)) for s in source_dirs],
            "total_classes_analyzed": len(enriched_classes),
            "java_files_analyzed": len(java_files),
            "kotlin_files_analyzed": len(kotlin_files),
        }
        
        storage_data = {
            "metadata": metadata,
            "pre_process": {},
        }
        
        # Convert pre_process to serializable format
        # storage_obj is ClassSummaryOutput
        for storage_obj in enriched_classes:
            storage_data["pre_process"][storage_obj.full_classname] = storage_obj.dict()
        
        # Write to file
        filepath = Path(output_file)
        with open(filepath, "w") as f:
            json.dump(storage_data, f, indent=2)
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
        
        return analysis_output
    
    def _extract_all_classes(
        self, java_files: List[Path], kotlin_files: List[Path], input_dir: Path
    ) -> List[ClassStructure]:
        """
        Extract class information from all source files.
        
        Args:
            java_files: List of Java source files
            kotlin_files: List of Kotlin source files
            
        Returns:
            List of ClassSummaryOutput objects
        """
        all_classes = []
        
        # Process Java files
        for file_path in java_files:
            classes = self.java_parser.parse_file(file_path, input_dir)
            all_classes.extend(classes)
        
        # Process Kotlin files
        for file_path in kotlin_files:
            classes = self.kotlin_parser.parse_file(file_path, input_dir)
            all_classes.extend(classes)
        
        return all_classes
    
    def _analyze_dependencies_and_methods(
        self,
        classes: List[ClassStructure],
        java_files: List[Path],
        kotlin_files: List[Path],
        known_classes: Set[str],
        input_dir: str
    ) -> List[ClassStructure]:
        """
        Analyze dependencies and methods for all classes.
        
        Args:
            classes: List of ClassSummaryOutput objects
            java_files: List of Java source files
            kotlin_files: List of Kotlin source files
            known_classes: Set of known class names
            
        Returns:
            List of enriched ClassSummaryOutput objects
        """
        # Enrich class information with dependencies and methods
        enriched_classes = []
        total_classes = len(classes)
        for i, cls in enumerate(classes):
            source_file_relative = cls.source_file
            source_file_absolute = Path(input_dir) / source_file_relative
            print(f"Processing \"{cls.full_classname}\" ({i + 1}/{total_classes})")
            
            # Read file content only when needed
            try:
                with open(source_file_absolute, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                print(f"Warning: could not read {source_file_absolute}: {e}")
                enriched_classes.append(cls)
                continue
            
            # Determine which parser to use based on file extension
            parser = self.java_parser if source_file_absolute.suffix.lower() == '.java' else self.kotlin_parser
            
            # Extract dependencies
            dependencies = parser.extract_dependencies(
                file_content, cls.simple_classname, known_classes
            )
            
            # Extract methods
            methods = parser.extract_methods(file_content, cls.simple_classname)
            if self.verbose:
                print(f"Found {len(dependencies)} dependency entries and {len(methods)} methods")
            
            # Create dependency structs
            dependency_structs = [
                ClassStructureDependency(
                    simple_classname=simple_name,
                    full_classname=full_name,
                    usage_lines=usage_lines
                )
                for simple_name, full_name, usage_lines in dependencies
            ]
            
            # Create method structs
            method_structs = [
                ClassStructureMethod(
                    name=name,
                    definition_start=start_line,
                    definition_end=end_line
                )
                for name, start_line, end_line in methods
            ]
            
            # Update class with dependencies and methods
            cls.dependencies = dependency_structs
            cls.public_methods = method_structs
            
            enriched_classes.append(cls)
        
        return enriched_classes