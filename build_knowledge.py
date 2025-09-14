import argparse
import os
import sys
from typing import List, Dict, Set
import logging
from core.utils.logging_utils import setup_logging, setup_llm_logger
from core.utils.token_utils import tokens_to_chars, chars_to_tokens

from datetime import datetime, timezone
from core.config.config import load_config
from core.utils.file_utils import get_file_content, get_file_content_safe
from core.utils.file_manager import FileManager
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from core.llm.llm_execution_openai import OpenAILlmExecution
from knowledge.knowledge_store import KnowledgeStore, DepUsage
from knowledge.model import FileDescription, ClassDescription, ClassDescriptionExtended, FileInfo
from knowledge.embeddings_store import Embeddings
from static_analysis.core.analyzer import CodebaseAnalyzer
from static_analysis.model.model import ClassStructure, ClassStructureDependency
from core.llm.prepare_prompts import params_add_field_descriptions, params_add_ignored_packaged, params_get_project_context, params_add_containing_classes
from core.search.embedding_entry import EmbeddingEntry


# Initialize Storage as a global instance
knowledge_store = KnowledgeStore()
file_manager = FileManager()
logger = None
embeddings = None  # Will be initialized after config is loaded
config = None

file_static_db = "db_preprocess.json"
file_final_db = "db_final.json"
file_config = "config.json"

def process_summary_llm(
        file_path: str, 
        content: str, 
        filecontext: str, 
        prompt_params: Dict, 
        prompt_templates: Dict
    ) -> FileDescription:
    """
    Generate a structured summary of the file
    Returns a FileFinalSummaryOutput object with 'summary' and 'dependencies' fields
    """
    # Skip empty or error content
    if not content or content.startswith('['):
        # Return a minimal valid structure for error cases
        return FileDescription(
            classes=[]
        )
    
    # Get the file extension and relative filepath
    _, file_extension = os.path.splitext(file_path)
    
    try:
        system_prompt = prompt_templates["system_prompt_template"]
        prompt = prompt_templates["final_prompt_template"].format(
                field_descriptions = prompt_params["field_descriptions_final"],
                file_extension = " " + file_extension if file_extension else "",
                filename = "File name: " + file_path,
                content = "File content:\n```\n" + content + "```\n",
                ignore_packages = prompt_params["ignore_packages"],
                contain_classes = prompt_params["contain_classes"],
                projectcontext = prompt_params["projectcontext"],
                filecontext = "" + filecontext + "" if filecontext is not None else ""
            )
        
        system_prompt = f"""
    You are helpful kotlin and java expert that outputs JSON response, analysing and summarizing files of the android software sorce code. Your output will help understand how the whole project works. Generated JSON need to strictly follow the provided JSON Schema.
    """

        schema = FileDescription.model_json_schema()
        logger.info(f"Processing with LLM ~{chars_to_tokens(len(prompt) + len(system_prompt) + len(schema))} tk...")
        response = llm_execution.llm_invoke(system_prompt, prompt=prompt, schema=schema)
        return FileDescription(**response)
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        logger.error(error_msg)
        sys.exit(1)
    

def format_prompt(prompt_params: dict, prompt_template: str, filename, fileext, content):
    return prompt_template.format(
        field_descriptions = prompt_params["field_descriptions_summary"],
        file_extension = fileext,
        filename = filename,
        content = content,
        ignore_packages = prompt_params["ignore_packages"]
        )

def calc_priority_for_dependency_files(rel_path: str, dependency_files_with_priority: dict) -> List[tuple[str, int]]:
    """
    Extract dependency files from a given file and return them with priority information.
    
    Args:
        rel_path: Relative path of the file
        
    Returns:
        List of tuples (dependency_file_path, priority) sorted by priority (highest first)
    """
    # Use memory to add additional context to the files
    file_classes = knowledge_store.get_file_structure(rel_path)
    
    # Dictionary to track dependency files and their usage count
    
    # Collect all distinct dependency files with their usage count
    for class_info in file_classes if file_classes else []:
        for dependency in class_info.dependencies:
            if dependency.full_classname in knowledge_store.class_structure_dict:
                dependency_storage = knowledge_store.class_structure_dict[dependency.full_classname]
                dep_file = dependency_storage.source_file
                # Use number of usage_lines as priority
                priority = len(dependency.usage_lines) if hasattr(dependency, 'usage_lines') else 0

                usages_fraction = 1 - (1/priority) if priority > 0 else 0
                
                # Update priority if this dependency has more usage lines
                if dep_file in dependency_files_with_priority:
                    dependency_files_with_priority[dep_file] = dependency_files_with_priority[dep_file] + 2 + usages_fraction
                else:
                    dependency_files_with_priority[dep_file] = 2 + usages_fraction
    
    # Convert to list of tuples and sort by priority (highest first)
    result = [(file, priority) for file, priority in dependency_files_with_priority.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result



def process_summary_for_file(rel_path: str, 
                             path_final_db: str,
                             prompt_params: Dict,
                             prompt_templates: Dict,
                             base_dir: str,
                             processed_counter: int,
                             total_files: int,
                             all_class_summaries: List[ClassDescriptionExtended]
    ) -> None:
    """
    Process a file for final LLM processing and update shared data
    
    Args:
        rel_path: Relative path of the file
        prompt_params: Parameters for the prompts
        prompt_templates: The prompt templates to use
        base_dir: Base directory (needed for file operations)
        processed_counter: Counter for tracking progress
        total_files: Total number of files for progress reporting
        all_class_summaries: List to collect class summaries
    """

    logger.info(f"Processing: {rel_path}")

    add_usages_summaries = True
    add_usages_methods = False
    add_dependency_summaries = True
    add_dependency_methods_and_properties = True


    # Create filecontext string with class information

    dependencies: Dict[str, DepUsage] = knowledge_store.get_file_dependencies(rel_path)

    # Convert relative path to absolute for file operations
    abs_file_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
    
    # Get file content
    content = get_file_content_safe(abs_file_path)

    usages: Dict[str, DepUsage] = knowledge_store.get_file_usages(rel_path)
    
    filecontext: str = knowledge_store.prepare_file_context(base_dir, rel_path, dependencies, usages, content, 
                                            add_dependency_summaries, 
                                            add_dependency_methods_and_properties,
                                            add_usages_summaries,
                                            add_usages_methods,
                                            diff_window=4)

    if chars_to_tokens(len(filecontext) + len(content)) > config.llm.warn_context:
        logger.info(f"Warn context reached, limiting {rel_path}")
        filecontext = knowledge_store.prepare_file_context(base_dir, rel_path, dependencies, usages, content, 
                            add_dependency_summaries, 
                            add_dependency_methods_and_properties = False,
                            add_usages_summaries = False,
                            add_usages_methods = False,
                            diff_window=2
                        )
    
    if chars_to_tokens(len(filecontext) + len(content)) > config.llm.max_context:
        logger.error(f"Max context reached, skipping {rel_path}")
        return
    
    if chars_to_tokens(len(filecontext) + len(content)) < config.llm.min_context:
        logger.error(f"Min context reached, skipping {rel_path}")
        return
    

    if content is None:
        logger.info(f"  [Binary file, skipping] {rel_path}")
        return
    
    prompt_params = params_add_containing_classes(prompt_params, rel_path, knowledge_store)
    
    # Print dependency files for debugging
    logger.info(f"""## Project context:
{prompt_params['projectcontext']}

## File name:
{rel_path}

## Dependencies and usages context:
{filecontext}

## Classes to describe:
{prompt_params['contain_classes']}

## Content of the file to process:
{content}
--- End of file ---""")
    
    # Generate summary
    logger.info(f"[{processed_counter}/{total_files}] File: {rel_path}")
    summary = process_summary_llm(rel_path, content, filecontext, prompt_params, prompt_templates)
    
    # Process results
    # Update processed counter

    logg_info = f"Done: [{processed_counter + 1}/{total_files}] {rel_path}\nClasses found:\n"
    for classs in summary.classes:
        logg_info += f"- Class: {classs.full_classname}\n"
        logg_info += f"  Summary: {classs.summary}\n"
        logg_info += f"  Category: {classs.category}\n"
    logg_info += f"========================================"
    
    # Print progress
    logger.info(logg_info)
    
    file_size = os.path.getsize(abs_file_path)
    version = file_size
    # Process and save results
    for classs in summary.classes:
        
        # Create ClassFinalSummaryStorage object
        storage_obj = ClassDescriptionExtended(
            class_summary=classs,
            file=rel_path,
            file_size=file_size,
            version=version,
        )
        
        # Add to collection
        all_class_summaries.append(storage_obj)
        knowledge_store.save_class_description(storage_obj)
    
    # Save after file processed
    knowledge_store.dump_write_storage_final(path_final_db)
    
    # Periodically report progress
    logger.info(f"Progress: {processed_counter}/{total_files} files processed")

def process_summaries_for_files(
        prompt_params: Dict, 
        prompt_templates: Dict, 
        base_dir: str,
        path_final_db: str,
        is_force = True,
        is_update_only = False
        ) -> List[ClassDescriptionExtended]:
    """
    Process a list of files and generate final summaries with context from storage
    Returns a list of ClassFinalSummaryStorage objects to be saved in storage
    
    Args:
        file_infos: List of FileInfo objects to process
        prompt_params: Parameters for the prompts
        prompt_templates: The prompt templates to use
        base_dir: Base directory (needed for file operations)
    """
    # Build a dependency graph for all files
    file_info_map = {f.filepath: f for f in file_manager.file_infos}
    priorities_for_dependency_files = {}
    
    include_dependencies_first = True
    # First, collect all dependencies for each file and calc its priorities
    for file_info in file_manager.file_infos:
        if file_info.is_allowed_by_filter or is_update_only:
            rel_path = file_info.filepath
            priorities_for_dependency_files[rel_path] = 0
            if include_dependencies_first:
                calc_priority_for_dependency_files(rel_path, priorities_for_dependency_files)

    # Convert to list of tuples and sort by priority (highest first)
    files_to_process_pre = [(file, priority) for file, priority in priorities_for_dependency_files.items()]
    files_to_process_pre.sort(key=lambda x: x[1], reverse=True)
    
    # List to collect all class summaries
    all_class_summaries = []
    files_to_process: List[FileInfo] = []

    only_missing = not is_force
    log_message_files = ""

    if is_update_only:
        existing_classes = set()
        new_classes = set()
        removed_classes = set()

        all_previous_entries = set(knowledge_store.descriptions_dict.keys())
        all_new_entries = set(knowledge_store.class_structure_dict.keys())
        
        new_classes = set(all_new_entries - all_previous_entries)
        removed_classes = set(all_previous_entries - all_new_entries)
        existing_classes = set(all_new_entries - new_classes - removed_classes)
        
        existing_modified_files = set()

        for deleted_classname in removed_classes:
            knowledge_store.remove_class_description(deleted_classname)
            embeddings.remove_embeddings(deleted_classname)

        for cls in new_classes:
            cls_new = knowledge_store.get_class_structure(cls)
            if cls_new:
                existing_modified_files.add(cls_new.source_file)

        for cls in existing_classes:
            cls_old = knowledge_store.get_class_description_extended(cls)
            cls_new = knowledge_store.get_class_structure(cls)
            
            prev_version = cls_old.version if cls_old else -1
            new_version = cls_new.version if cls_new else -1

            if cls_new and prev_version != new_version:
                existing_modified_files.add(cls_new.source_file)

        for file in files_to_process_pre:
            file_path = file[0]

            if file_path in existing_modified_files:
                file_info = file_info_map[file_path]
                files_to_process.append(file_info)
    else :
        for file in files_to_process_pre:
            file_path = file[0]
            file_priority = file[1]
            if file_path not in file_info_map:
                continue
            
            file_info = file_info_map[file_path]

            old_classes = knowledge_store.get_file_description(file_path)

            classes_in_file = knowledge_store.get_file_structure(file_path)

            is_missing_classes = len(classes_in_file) > len(old_classes)

            file_to_process = None
            if file_manager.is_filtering_enabled:
                if only_missing:
                    if is_missing_classes and file_info.is_allowed_by_filter:
                        file_to_process = file_info
                        
                else:
                    if file_info.is_allowed_by_filter:
                        file_to_process = file_info

            elif not old_classes or not only_missing:
                file_to_process = file_info

            if file_to_process:
                files_to_process.append(file_to_process)

    for file_to_process in files_to_process:
        log_message_files += f"\n- {file_to_process.filepath}"

    logger.info(f"Files after limiting:\n {log_message_files}")
    
    processed_counter = 0

    total_files = len(files_to_process)
    logger.info(f"Processing {total_files} files for final summary (including dependencies)")
    logger.info("Generating summaries with context...")

    prompt_params["projectcontext"] = params_get_project_context(base_dir)

    for i, file_info in enumerate(files_to_process):
        rel_path = file_info.filepath
        
        process_summary_for_file(rel_path,
                               path_final_db,
                               prompt_params, prompt_templates, base_dir,
                               processed_counter, total_files, all_class_summaries)
        
        processed_counter += 1

    logger.info(f"All files processed in dependency order")
    
    # Return the collected summaries
    return all_class_summaries

def process_embeddings(base_dir: str = None, is_update: bool = False) -> None:
    """
    Process embeddings for all classes in storage.final_process
    """
    embeddings.initialize(base_dir, create=(not is_update))

    total_classes = len(knowledge_store.descriptions_dict)
    logger.info(f"Processing {total_classes} classes for embeddings")
    logger.info("Generating embeddings...")
    
    embedding_entries_to_process: List[EmbeddingEntry] = []
    documents_to_embed: List[str] = []

    # 1. Create a list with EmbeddingEntry elements first with empty embeddings array
    # and a parallel list of documents to embed
    for classname, class_summary_extended in knowledge_store.descriptions_dict.items():
        rel_path = class_summary_extended.file
        class_summary = class_summary_extended.class_summary
        class_structure = knowledge_store.get_class_structure(classname)

        # delete missing entries
        old_key = class_summary.full_classname
        if is_update and not class_structure and old_key in embeddings.data:
            to_remove_keys = []
            for i in embeddings.data.keys():
                if old_key in i:
                    to_remove_keys.append(i)

            for key in to_remove_keys:
                del embeddings.data[key]
            continue

        # Class embedding
        filecontext_class: str = ""
        if class_summary.features:
            filecontext_class += "Used in features:\n"
            for feature in class_summary.features:
                filecontext_class += f"{feature}\n"
        if class_summary.questions:
            filecontext_class += "Example questions:\n"
            for question in class_summary.questions:
                filecontext_class += f"{question}\n"
        
        summary_class = f"{class_summary.summary}"
        text_to_embed = f"Summary of a `{class_summary.full_classname}` class implemented in a {rel_path} file: {summary_class}\n{filecontext_class}" if filecontext_class else summary_class

        embedding_version = calc_embedd_version(class_summary_extended, text_to_embed)

        old_key = class_summary.full_classname
        if is_update and old_key in embeddings.data and embeddings.data[old_key].version == embedding_version:
            continue
        
        documents_to_embed.append(text_to_embed)
        embedding_entries_to_process.append(EmbeddingEntry(
            type='class',
            detail='',
            full_classname=class_summary.full_classname,
            rel_path=rel_path,
            embedding=[], # empty for now
            version=embedding_version
        ))

        # Method embeddings
        for method in class_summary.methods:
            filecontext_method: str = ""
            if class_summary.features:
                filecontext_method += "Used in features:\n"
                for feature in class_summary.features:
                    filecontext_method += f"{feature}\n"
            
            summary_method = f"Method `{method.method_name}` of a `{class_summary.simple_classname}` class implemented in a {rel_path} file: {method.method_summary}"
            text_to_embed = f"{summary_method}\n{filecontext_method}" if filecontext_method else summary_method
            embedding_version = calc_embedd_version(class_summary_extended, text_to_embed)
            documents_to_embed.append(text_to_embed)
            embedding_entries_to_process.append(EmbeddingEntry(
                type='method',
                detail=method.method_name,
                full_classname=class_summary.full_classname,
                rel_path=rel_path,
                embedding=[], # empty for now
                version=embedding_version
            ))

        # Property embeddings
        for property in class_summary.properties:
            filecontext_property: str = ""
            if class_summary.features:
                filecontext_property += "Used in features:\n"
                for feature in class_summary.features:
                    filecontext_property += f"{feature}\n"

            summary_property = f"Property `{property.property_name}` of a `{class_summary.simple_classname}` class implemented in a {rel_path} file: {property.property_summary}"
            text_to_embed = f"{summary_property}\n{filecontext_property}" if filecontext_property else summary_property
            embedding_version = calc_embedd_version(class_summary_extended, text_to_embed)
            documents_to_embed.append(text_to_embed)
            embedding_entries_to_process.append(EmbeddingEntry(
                type='property',
                detail=property.property_name,
                full_classname=class_summary.full_classname,
                rel_path=rel_path,
                embedding=[], # empty for now
                version=embedding_version
            ))

    if documents_to_embed:
        # Generate embeddings in batch
        embedding_vectors = embeddings.generate_documents_embedding(documents_to_embed)
        
        # 3. assign embedding vectors from the result to the output list
        for i, entry in enumerate(embedding_entries_to_process):
            key = f"{entry.full_classname}.{entry.detail}" if entry.type != 'class' else entry.full_classname

            entry.embedding = embedding_vectors[i]
            embeddings.data[key] = entry

    # Save to file
    embeddings.store_all_classes()

def calc_embedd_version(class_storage, text_to_embed):
    return class_storage.version + len(text_to_embed)

llm_execution = None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate summaries for files in a directory.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument('--log-file', help='Path to the log file.')
    parser.add_argument('--llm-log-file', help='Path to the LLM log file.')
    parser.add_argument('-i', '--input-dir', help='Directory containing files to summarize. Overrides config file.')
    
    parser.add_argument('-m', '--mode', choices=['Pre', 'Final', 'Embedd'],
                        help='Processing mode: Pre (TreeSitter preprocessing), Final (final process only), Embedd (embeddings only)')
    
    parser.add_argument('-f', '--filter', help='Filter files by name (only process files containing this string in filename). '
                                        'Prefix with "!" to invert (exclude files containing the string)')
    
    parser.add_argument('-fo', '--force', action='store_true', help='Process even if already processed')
    parser.add_argument('--update', action='store_true', help='Process even if already processed')
    
    args = parser.parse_args()

    # Initialize logging
    global logger
    global llm_logger
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    llm_logger = setup_llm_logger(log_level=args.log_level, log_file=args.llm_log_file)

    # Determine input directory    
    base_dir = os.path.abspath(args.input_dir)
    
    path_config = os.path.join(base_dir, ".ai-agent", file_config)
    path_static_db = os.path.join(base_dir, ".ai-agent", file_static_db)
    path_final_db = os.path.join(base_dir, ".ai-agent", file_final_db)

    # Load configuration
    global config
    config = load_config(path_config)

    # Initialize LLM execution
    global llm_execution, embeddings
    if config.llm.mode == 'mlx':
        logger.info("Initializing connection to MLX...")
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature, logger=llm_logger)
    elif config.llm.mode == 'ollama':
        logger.info("Initializing connection to Ollama...")
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url, logger=llm_logger, max_context=config.llm.max_context)
    elif config.llm.mode == 'anthropic':
        logger.info("Initializing connection to Anthropic...")
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key, logger=llm_logger)
    elif config.llm.mode == 'openai':
        logger.info(f"Initializing connection to {config.llm.openai.url}")
        llm_execution = OpenAILlmExecution(
            model=config.llm.openai.model,
            temperature=config.llm.openai.temperature,
            key=config.llm.openai.key,
            base_url=config.llm.openai.url,
            logger=llm_logger
        )
    else:
        raise ValueError(f"Unsupported LLM mode: {config.llm.mode}")
    
    # Initialize embeddings with config
    embeddings = Embeddings(config, logger)
    
    # Load the prompt templates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    mode = args.mode  # Can be None if not specified
    
    # Get file information with optional filter
    file_manager.load(logger, base_dir, config, name_filter=args.filter)

    # Print out the filtered files with sizes
    file_manager.print_files_info(logger)

    # Load prompt templates for final processing
    final_prompt_path = os.path.join(script_dir, 'knowledge', 'prompts', 'final_proccess_prompt.txt')
    system_prompt_path = os.path.join(script_dir, 'knowledge', 'prompts', 'system_prompt.txt')
    system_prompt_template = get_file_content(system_prompt_path)
    final_prompt_template = get_file_content(final_prompt_path)
    
    # Create prompt templates dictionary
    prompt_templates = {
        "system_prompt_template": system_prompt_template,
        "final_prompt_template": final_prompt_template
    }
    
    # Create parameter dictionaries for both processes
    prompt_params = {}
    
    # Add params
    prompt_params = params_add_field_descriptions(prompt_params)
    prompt_params = params_add_ignored_packaged(prompt_params)

    is_update_only = args.update
    is_force = args.force 

    # Determine which processes to run based on mode
    run_pre_process = mode is None or mode == 'Pre'  or is_update_only
    run_final_process = mode is None or mode == 'Final' or is_update_only
    run_embeddings = mode is None or mode == 'Embedd' or is_update_only or is_force

    if run_final_process:
        llm_execution.on_load()
        
    # Pre-processing step with TreeSitter
    if run_pre_process:
        # Create analyzer and run analysis
        analyzer = CodebaseAnalyzer(verbose=args.log_level == 'DEBUG')
        analyzer.analyze_codebase(config.source_dirs, str(base_dir), path_static_db)
        knowledge_store.read_storage_pre_process(path_static_db)

    if not run_pre_process and (run_final_process or run_embeddings):
        # If we're not running pre-process but need the data for later steps
        logger.info("=== Loading pre-processed data from storage ===")
        knowledge_store.read_storage_pre_process(path_static_db)

    # Final processing step
    if run_final_process:
        logger.info("=== Running Final processing ===")
        knowledge_store.read_storage_final(path_final_db, base_dir)


        # Process with memory and get all class summaries
        class_summaries_final = process_summaries_for_files(
            prompt_params, 
            prompt_templates, 
            base_dir, 
            path_final_db = path_final_db,
            is_force=is_force, 
            is_update_only = is_update_only)
        
        # Save all class summaries to storage at once
        logger.info(f"Saving {len(class_summaries_final)} class summaries with memory to storage...")
        for classs in class_summaries_final:
            knowledge_store.save_class_description(classs)

        knowledge_store.dump_write_storage_final(path_final_db)

    if run_embeddings and not run_final_process:
        # If we're only running embeddings and not final process, we need to load final data
        logger.info("=== Loading final processed data from storage ===")
        knowledge_store.read_storage_final(path_final_db, base_dir)

    # Embeddings processing step
    if run_embeddings:
        logger.info("=== Running Embeddings processing ===")
        is_update = is_update_only or file_manager.is_filtering_enabled
        process_embeddings(base_dir=base_dir, is_update = is_update)

    logger.info("\nSummary generation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected error occurred:")
        sys.exit(1)