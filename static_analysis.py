"""
Command-line interface for the Kotlin and Java analyzer.

This module provides a command-line interface for running the analyzer on a Kotlin and Java codebase.
"""

import argparse
import sys
import os
from pathlib import Path

from static_analysis.core.analyzer import CodebaseAnalyzer
from core.config.config import load_config


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze a Kotlin and Java codebase and generate a JSON report.'
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        required=True,
        help='Directory containing Kotlin and Java source files to analyze'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    try:
        args = parse_args()
        
        # Check if input directory exists
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        if not input_dir.is_dir():
            print(f"Error: {input_dir} is not a directory")
            sys.exit(1)
        
        # Load configuration
        config_file_path = input_dir / ".ai-agent" / "config.json"
        if not config_file_path.exists():
            print(f"Error: Configuration file not found at {config_file_path}")
            sys.exit(1)
            
        config = load_config(str(config_file_path))
        
        source_dirs = [str(input_dir / src) for src in config.source_dirs]
        
        for source_dir in source_dirs:
            if not Path(source_dir).exists():
                print(f"Warn: Source directory does not exist: {source_dir}")
                sys.exit(1)
            if not Path(source_dir).is_dir():
                print(f"Warn: {source_dir} is not a directory")
                sys.exit(1)

        # Create analyzer and run analysis
        analyzer = CodebaseAnalyzer(verbose=args.verbose)
        output = input_dir / ".ai-agent" / "preprocess.json"
        analyzer.analyze_codebase(source_dirs, str(input_dir), output)
        
        print(f"Analysis complete. Results written to {output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()