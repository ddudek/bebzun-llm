import argparse
import os
import sys
import json
import logging
import traceback
from typing import List, Dict

from core.config.config import load_config
from knowledge.knowledge_store import KnowledgeStore
from knowledge.embeddings import Embeddings
from knowledge.model import MethodDescription, VariableDescription
from core.search.embedding_entry import EmbeddingEntry
from core.search.search import KnowledgeSearch
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from core.utils.file_utils import get_file_content, get_all_files
from core.utils.logging_utils import setup_logging, setup_llm_logger

# Global instances
llm_execution = None
knowledge_store = KnowledgeStore()
embeddings = None
logger = None

def similarity_search(args, config):
    """
    Performs vector similarity search using the knowledge base.
    """
    global embeddings, knowledge_store
    search = KnowledgeSearch(embeddings, knowledge_store, config, logger)
    results = search.vector_search_combined(args.query, limit=args.limit)
    results = search.rerank_results(results, args.query)
    
    output_results = []
    for res in results:
        item: EmbeddingEntry = res.entry
        content = res.describe_content()        
        output_results.append({
            'full_classname': item.full_classname,
            'content': content,
            'file_path': item.rel_path,
            'score': res.vector_score,
            'rerank_score': res.rerank_score
        })
    
    logger.info(json.dumps(output_results, indent=2))

def bm25_search(args, config):
    """
    Performs BM25 search using the knowledge base.
    """
    global embeddings, knowledge_store
    search = KnowledgeSearch(embeddings, knowledge_store, config, logger)
    results = search.bm25_search(args.query, limit=args.limit)
    results = search.rerank_results(results, args.query)
    
    output_results = []
    for res in results:
        item: EmbeddingEntry = res.entry
        content = res.describe_content() 
        output_results.append({
            'full_classname': item.full_classname,
            'content': content,
            'file_path': item.rel_path,
            'score': res.vector_score,
            'rerank_score': res.rerank_score
        })
        
    logger.info(json.dumps(output_results, indent=2))

def summarize_project(args, config):
    """
    Generates a summary of the entire project based on the knowledge base.
    """
    global llm_execution, knowledge_store
    logger.info("Generating project summary...")

    # 1. Load project context
    project_context_path = os.path.join(args.input_dir, ".ai-agent", "project_context.txt")
    project_context = ""
    if os.path.exists(project_context_path):
        project_context = get_file_content(project_context_path)

    # 2. Get all files filtered
    all_files = get_all_files(
        args.input_dir, 
        config.source_dirs, 
        ignored_dirs={'.ai-agent', '.ai-agent-bak', '.git', '__pycache__', '.idea', 'build', 'gradle'},
        ignored_files={'.DS_Store'}
    )
    
    # 3. Get all class summaries, interating by each file
    file_entries = []
    for file_path in all_files:
        classes = knowledge_store.get_file_description(file_path)
        file_entry = f"- {file_path}"
        if classes:
            file_entry += "\nContains:\n"
            for class_summary in classes:
                file_entry += f"Class: {class_summary.full_classname}: {class_summary.summary}\n"

        file_entries.append(file_entry)

    class_summaries_str = "\n".join(file_entries)

    # 4. Prepare prompt
    system_prompt = "You are a senior software engineer tasked with creating a high-level summary of a software project. Your summary will be read by new developers to quickly understand the project."
    prompt = f"""
Based on the information provided below, please generate summary of the project.

# Project Context
{project_context}

# File List with summaries
{class_summaries_str}

---
**Project Summary Task:**

Based on all the information above (project context, file list, and class summaries), please provide a comprehensive yet concise summary of the project. The summary should cover:
1.  **Project Purpose:** What is the main goal of the project?
2.  **All Features:** List all features and components. For each write instructions where are the main files to understand how feature works.
3.  **Architecture:** Briefly describe the high-level architecture or design patterns used.
4.  **Key Technologies:** What are the main languages, frameworks, and libraries used?
"""

    # 5. Invoke LLM
    if llm_execution:
        llm_execution.on_load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        summary = llm_execution.llm_chat(messages)
        logger.info(json.dumps({"project_summary": summary}, indent=2))
    else:
        logger.error(json.dumps({"error": "LLM execution not initialized."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description='CLI for querying codebase knowledge.')
    parser.add_argument('-i', '--input-dir', required=True, help='Directory containing project and .ai-agent folder.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument('--log-file', help='Path to the log file.')
    parser.add_argument('--llm-log-file', help='Path to the LLM log file.')

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Similarity search command
    search_parser = subparsers.add_parser('similarity_search', help='Perform similarity search using vector embeddings.')
    search_parser.add_argument('query', help='The search query.')
    search_parser.add_argument('--limit', type=int, default=15, help='Number of results to return.')
    search_parser.set_defaults(func=similarity_search)

    # BM25 search command
    bm25_parser = subparsers.add_parser('bm25_search', help='Perform keyword-based BM25 search.')
    bm25_parser.add_argument('query', help='The search query.')
    bm25_parser.add_argument('--limit', type=int, default=15, help='Number of results to return.')
    bm25_parser.set_defaults(func=bm25_search)

    # Summarize project command
    summarize_parser = subparsers.add_parser('summarize_project', help='Generate a summary of the entire project.')
    summarize_parser.set_defaults(func=summarize_project)

    args = parser.parse_args()
    
    # Initialize logging
    global logger
    global llm_logger
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    llm_logger = setup_llm_logger(log_level=args.log_level, log_file=args.llm_log_file)

    input_dir = os.path.abspath(args.input_dir)

    # Load configuration
    config_file_path = os.path.join(input_dir, ".ai-agent", "config.json")
    if not os.path.exists(config_file_path):
        logger.error(f"Error: Configuration file not found at {config_file_path}")
        sys.exit(1)
    config = load_config(config_file_path)

    # Initialize components
    global llm_execution, embeddings
    if config.llm.mode == 'mlx':
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature, logger=llm_logger)
    elif config.llm.mode == 'ollama':
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url, logger=llm_logger)
    elif config.llm.mode == 'anthropic':
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key, logger=llm_logger)
    
    embeddings = Embeddings(config, logger)
    embeddings.initialize(input_dir)

    # Load knowledge store data
    final_json_path = os.path.join(input_dir, ".ai-agent", "db_final.json")
    if not os.path.exists(final_json_path):
        logger.error(f"Error: Knowledge base file not found at {final_json_path}")
        logger.error("Please run 'build_knowledge.py' first.")
        sys.exit(1)
    knowledge_store.read_storage_final(final_json_path)

    # Execute the function for the chosen command
    if hasattr(args, 'func'):
        args.func(args, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected error occurred:")
        sys.exit(1)