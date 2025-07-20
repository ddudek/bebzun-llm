import argparse
import os
import sys
from typing import List, Dict, Set
import logging
from core.utils.logging_utils import setup_logging, setup_llm_logger
from core.utils.token_utils import tokens_to_chars, chars_to_tokens

from datetime import datetime, timezone
from core.config.config import load_config
from core.utils.file_utils import get_file_content, is_binary_file, get_file_content_safe, format_file_size
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from knowledge.knowledge_store import KnowledgeStore, DepUsage
from knowledge.model import FileDescription, ClassDescription, ClassDescriptionExtended, FileInfo
from knowledge.embeddings import Embeddings
from static_analysis.model.model import ClassStructure, ClassStructureDependency
from core.llm.prepare_prompts import params_add_list_of_files, params_add_field_descriptions, params_add_ignored_packaged, params_add_project_context

# Initialize Storage as a global instance
knowledge_store = KnowledgeStore()
logger = None
embeddings = None  # Will be initialized after config is loaded
config = None

def final_process_file(file_path: str, content: str, filecontext: str, prompt_params: Dict, prompt_templates: Dict, base_dir: str = None) -> FileDescription:
    """
    Generate a structured summary of the file using Ollama
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
    
    # Format the list of files with tags
    list_of_files_txt = f"```{prompt_params['list_of_files']}\n```"
    
    try:
        system_prompt = prompt_templates["system_prompt_template"]
        prompt = prompt_templates["final_prompt_template"].format(
                field_descriptions = prompt_params["field_descriptions_final"],
                file_extension = " " + file_extension if file_extension else "",
                filename = "File name: " + file_path,
                content = "File content:\n```\n" + content + "```\n",
                list_of_files = list_of_files_txt,
                ignore_packages = prompt_params["ignore_packages"],
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
    

def get_filtered_files(input_dir: str, source_dirs: List[str], extensions: tuple = ('.kt', '.java'), name_filter: str = None) -> List[FileInfo]:
    """
    Get a list of files with specified extensions from multiple source directories recursively.

    Args:
        input_dir: The base directory of the project.
        source_dirs: List of source directories relative to the input_dir.
        extensions: Tuple of file extensions to filter by.
        name_filter: Optional string to filter files by name.
                     If name_filter starts with "!", the filter is inverted.

    Returns:
        List of FileInfo objects with filepaths relative to input_dir.
    """
    file_infos = []
    
    invert_filter = False
    filter_text = name_filter
    if name_filter and name_filter.startswith("!"):
        invert_filter = True
        filter_text = name_filter[1:]

    for src_dir in source_dirs:
        abs_src_path = os.path.join(input_dir, src_dir)
        if not os.path.exists(abs_src_path):
            logger.warning(f"The source directory '{abs_src_path}' does not exist.")
            continue
            
        for root, _, files in os.walk(abs_src_path):
            for filename in files:
                if filename.endswith(extensions):
                    abs_file_path = os.path.join(root, filename)
                    rel_file_path = os.path.relpath(abs_file_path, input_dir)
                    file_size = os.path.getsize(abs_file_path)
                    mtime = os.path.getmtime(abs_file_path)
                    modified_timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
                    
                    is_allowed = True
                    if filter_text:
                        contains_filter = filter_text in filename
                        is_allowed = not contains_filter if invert_filter else contains_filter
                    
                    file_info = FileInfo(
                        filepath=rel_file_path,
                        file_size=file_size,
                        modified_timestamp=modified_timestamp,
                        is_allowed_by_filter=is_allowed
                    )
                    file_infos.append(file_info)
                    
    return file_infos

def format_prompt(prompt_params: dict, prompt_template: str, filename, fileext, content):
    return prompt_template.format(
        field_descriptions = prompt_params["field_descriptions_summary"],
        file_extension = fileext,
        filename = filename,
        content = content,
        ignore_packages = prompt_params["ignore_packages"]
        )

def get_dependency_files(rel_path: str, dependency_files_with_priority: dict) -> List[tuple[str, int]]:
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



def llm_final_process_file(rel_path: str, dependency_files: set,
                                prompt_params: Dict, prompt_templates: Dict, base_dir: str,
                                processed_counter: int, total_files: int, all_class_summaries: List[ClassDescriptionExtended]) -> None:
    """
    Process a file for final LLM processing and update shared data
    
    Args:
        rel_path: Relative path of the file
        file_memory: Memory information for the file from cache
        dependency_files: Set of dependency files
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
    
    filecontext: str = prepare_file_context(base_dir, rel_path, dependencies, usages, content, 
                                            add_dependency_summaries, 
                                            add_dependency_methods_and_properties,
                                            add_usages_summaries,
                                            add_usages_methods,
                                            diff_window=4)

    if chars_to_tokens(len(filecontext) + len(content)) > config.llm.warn_context:
        logger.info(f"Warn context reached, limiting {rel_path}")
        filecontext = prepare_file_context(base_dir, rel_path, dependencies, usages, content, 
                            add_dependency_summaries, 
                            add_dependency_methods_and_properties = False,
                            add_usages_summaries = False,
                            add_usages_methods = False,
                            diff_window=2
                        )
    
    if chars_to_tokens(len(filecontext) + len(content)) > config.llm.max_context:
        logger.error(f"Max context reached, skipping {rel_path}")
        return
    
    if content is None:
        logger.info(f"  [Binary file, skipping] {rel_path}")
        return
    
    # Print dependency files for debugging
    logger.info(f"""## Project context:
{prompt_params['projectcontext']}

## File name:
{rel_path}

## Dependencies and usages context:
{filecontext}

## File content
{content}
--- End of file ---""")
    
    # Generate summary

    logger.info(f"[{processed_counter}/{total_files}] File: {rel_path}")
    summary = final_process_file(rel_path, content, filecontext, prompt_params, prompt_templates, base_dir)
    
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
    
    md_timestamp = int(os.path.getmtime(abs_file_path))
    # Process and save results
    for classs in summary.classes:
        
        # Create ClassFinalSummaryStorage object
        storage_obj = ClassDescriptionExtended(
            class_summary=classs,
            file=rel_path,
            timestamp=md_timestamp
        )
        
        # Add to collection
        all_class_summaries.append(storage_obj)
        knowledge_store.save_class_description(storage_obj)
    
    # Save after file processed
    knowledge_store.dump_write_storage_final(f"{base_dir}/.ai-agent/db_final.json")
    
    # Periodically report progress
    logger.info(f"Progress: {processed_counter}/{total_files} files processed")

def prepare_file_context(base_dir, rel_path, 
                         dependencies: Dict[str, DepUsage], 
                         usages: Dict[str, DepUsage],
                         content: str, 
                         add_dependency_summaries, 
                         add_dependency_methods_and_properties,
                         add_usages_summaries,
                         add_usages_methods,
                         diff_window
                         ) -> str:
    filecontext = " "

    for key, dependency in dependencies.items():
        if dependency.dependency_description and dependency.dependency_description.file != rel_path:
            filecontext += f"\n# `{dependency.parent_structure.full_classname}` is using `{dependency.dependency_structure.full_classname}` in lines: [{', '.join(map(str, dependency.dep.usage_lines))}]\n"
             
            if add_dependency_summaries:
                check_in_content = content if add_dependency_methods_and_properties else " "
                filecontext += f"{dependency.dependency_description.class_summary.describe('- ', check_in_content)}\n"
            continue
    
    for key, dep_usage in usages.items():

        parent_class_structure = dep_usage.parent_structure
        parent_class_description = dep_usage.parent_description

        dependency_class_name = dep_usage.dependency_structure.full_classname

        parent_file_rel_path = dep_usage.parent_structure.source_file # Relative path of the file where the usage occurs
        usage_line_numbers = dep_usage.dep.usage_lines # List of line numbers (1-based) where usage occurs

        # skip quoting if it's the same file.
        if parent_file_rel_path == rel_path:
            continue

        filecontext += "\n"

        filecontext += f"# `{dependency_class_name}` is used by `{parent_class_structure.full_classname}` in file '{parent_file_rel_path}'\n"
        if parent_class_description:
            if add_usages_summaries:
                filecontext += f"{parent_class_description.class_summary.simple_classname} summary: {parent_class_description.class_summary.summary}\n"
            if add_usages_methods:
                for method in parent_class_description.class_summary.methods:
                    if "get" not in method.method_name: #ignore getters
                        filecontext += f"- Method {method.method_name}: {method.method_summary}\n"

        # Construct absolute path for reading the file where usage occurs
        # base_dir is a parameter of llm_final_process_file
        abs_usage_file_path = os.path.join(base_dir, parent_file_rel_path) if base_dir else parent_file_rel_path

        # Get content of the usage file
        usage_file_content_str = get_file_content_safe(abs_usage_file_path)

        if usage_file_content_str:
            usage_file_lines_all = usage_file_content_str.splitlines()
            num_total_lines_in_file = len(usage_file_lines_all)

            if num_total_lines_in_file == 0:
                # File is empty, so no lines to include.
                pass
            else:
                lines_to_include_flags = [False] * num_total_lines_in_file

                for line_num_1_based in usage_line_numbers:
                    if line_num_1_based <= 0: # Defensive check for 1-based line numbers
                        continue
                    
                    line_num_0_based = line_num_1_based - 1
                    
                    # Ensure the usage line itself is valid before creating a window
                    if not (0 <= line_num_0_based < num_total_lines_in_file):
                        # Invalid usage line number provided, skip this one.
                        # Consider logging this if it's unexpected.
                        continue

                    # Determine the window of lines to mark for inclusion
                    # diff_window = 8
                    start_mark_0_based = max(0, line_num_0_based - diff_window)
                    end_mark_0_based = min(num_total_lines_in_file - 1, line_num_0_based + diff_window)
                    
                    for i in range(start_mark_0_based, end_mark_0_based + 1):
                        lines_to_include_flags[i] = True
                
                # Construct the consolidated snippet if any lines were marked
                if any(lines_to_include_flags):
                    current_block_idx = 0
                    while current_block_idx < num_total_lines_in_file:
                        if lines_to_include_flags[current_block_idx]:
                            # Start of a new contiguous block of lines to include
                            block_start_0_based = current_block_idx
                            
                            # Find the end of this contiguous block
                            block_end_0_based = current_block_idx
                            while (block_end_0_based + 1 < num_total_lines_in_file and
                                   lines_to_include_flags[block_end_0_based + 1]):
                                block_end_0_based += 1
                            
                            # This block runs from block_start_0_based to block_end_0_based (inclusive)
                            
                            # Identify which of the original usage_line_numbers fall into this block
                            # usage_line_numbers is a list of 1-based integers from dep_usage.dep.usage_lines
                            original_usages_in_this_block = set()
                            for original_ul_1_based in usage_line_numbers:
                                original_ul_0_based = original_ul_1_based - 1
                                if block_start_0_based <= original_ul_0_based <= block_end_0_based:
                                    original_usages_in_this_block.add(original_ul_1_based)
                            
                            if original_usages_in_this_block:
                                sorted_usages = sorted(list(original_usages_in_this_block))
                                filecontext += f"  source of `{parent_class_structure.full_classname}` in line(s): {', '.join(map(str, sorted_usages))}\n"
                            
                            filecontext += "  ```\n"
                            for line_idx_0_based in range(block_start_0_based, block_end_0_based + 1):
                                filecontext += f"  {line_idx_0_based + 1:4d} | {usage_file_lines_all[line_idx_0_based]}\n"
                            filecontext += "  ```\n"
                            
                            # Move current_block_idx to the position after this processed block
                            current_block_idx = block_end_0_based + 1
                        else:
                            # This line is not included, move to the next line
                            current_block_idx += 1
        else:
            print(f"  [INFO] Could not read content of '{parent_file_rel_path}' or it is a binary file.\n")
        filecontext += "\n" # Add a newline for separation between different usage entries in filecontext
    return filecontext

def final_process(
        file_infos: List[FileInfo], 
        prompt_params: Dict, 
        prompt_templates: Dict, 
        base_dir: str = None,
        is_filtering_enabled: bool = False, 
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
    file_info_map = {f.filepath: f for f in file_infos}
    dependency_files_with_priority = {}
    
    include_dependencies_first = True
    # First, collect all dependencies for each file
    for file_info in file_infos:
        if file_info.is_allowed_by_filter:
            rel_path = file_info.filepath
            dependency_files_with_priority[rel_path] = 0
            if include_dependencies_first:
                get_dependency_files(rel_path, dependency_files_with_priority)

    # Convert to list of tuples and sort by priority (highest first)
    files_to_process_pre = [(file, priority) for file, priority in dependency_files_with_priority.items()]
    files_to_process_pre.sort(key=lambda x: x[1], reverse=True)
    
    # List to collect all class summaries
    all_class_summaries = []
    files_to_process: List[FileInfo] = []

    only_missing = not is_force
    log_message_files = ""
    for file in files_to_process_pre:
        file_path = file[0]
        file_priority = file[1]
        file_info = file_info_map[file_path]

        old_classes = knowledge_store.get_file_description(file_path)

        file_to_process = None
        if is_filtering_enabled:
            if only_missing:
                if not old_classes and file_info.is_allowed_by_filter:
                    file_to_process = file_info
                    
            else:
                if file_info.is_allowed_by_filter:
                    file_to_process = file_info

        elif not old_classes or not only_missing:
            file_to_process = file_info

        # Check update only (only files that were modified)
        if is_update_only and file_to_process:
            allowed = False
            classes_in_file = knowledge_store.get_file_structure(rel_path)
            if not classes_in_file or not old_classes:
                allowed = True

            for cl in classes_in_file:
                old_description = knowledge_store.get_class_description_extended(cl.full_classname)
                if old_description and old_description.timestamp != cl.timestamp:
                    allowed = True

            if not allowed:
                file_to_process = None

        if file_to_process:
            files_to_process.append(file_to_process)
            log_message_files += f"\n- {file_path}, Priority: {file_priority}"
            

    logger.info(f"Files after limiting:\n {log_message_files}")
    
    processed_counter = 0

    total_files = len(files_to_process)
    logger.info(f"Processing {total_files} files for final summary (including dependencies)")
    logger.info("Generating summaries with context...")

    for i, file_info in enumerate(files_to_process):
        rel_path = file_info.filepath
        
        dependency_files_with_priority_single = get_dependency_files(rel_path, {})
        dependency_files = set(df for df, _ in dependency_files_with_priority_single)
        
        llm_final_process_file(rel_path,
                               dependency_files,
                               prompt_params, prompt_templates, base_dir,
                               processed_counter, total_files, all_class_summaries)
        
        processed_counter += 1

    logger.info(f"All files processed in dependency order")
    
    # Return the collected summaries
    return all_class_summaries

def process_embeddings(file_infos: List[FileInfo], base_dir: str = None, project_context: str = None) -> None:
    """
    Process embeddings for all classes in storage.final_process
    
    Args:
        file_infos: List of FileInfo objects (containing filepath, file_size, and modified_timestamp)
        base_dir: Base directory (needed for file operations)
        project_context: The project context string
    """
    embeddings.initialize(base_dir, create=True)

    total_classes = len(knowledge_store.descriptions_dict)
    logger.info(f"Processing {total_classes} classes for embeddings")
    logger.info("Generating embeddings...")
    
    processed = 0

    # Iterate through each class in storage.final_process
    for classname, class_storage in knowledge_store.descriptions_dict.items():
        processed += 1
        rel_path = class_storage.file
        
        logger.info(f"[{processed}/{total_classes}] Processing class: {classname}")
        
        # Get dependencies directly from class_storage.class_summary.dependencies
        # filecontext: str = f"{project_context}\n" if project_context else ""
        filecontext: str = ""

        # Get dependencies from the class summary
        class_summary = class_storage.class_summary
        class_preprocess = knowledge_store.get_class_structure(class_summary.full_classname)

        only_class = False

        if class_summary.features:
            filecontext += "Used in features:\n"
            for feature in class_summary.features:
                filecontext += f"{feature}\n"

        if class_summary.questions:
            filecontext += "Example questions:\n"
            for question in class_summary.questions:
                filecontext += f"{question}\n"
        
        # Generate embeddings and store with embeddings.store_embeddings
        logger.info(f"  Storing embeddings for class: {class_summary.full_classname}")
        embedd_summary = f"{class_summary.summary}"
        
        embeddings.store_class_description_embeddings('class', class_summary.full_classname, '', embedd_summary, filecontext, rel_path, class_storage.timestamp)

        if not only_class:
            for method in class_summary.methods:
                filecontext: str = ""
                if class_summary.features:
                    filecontext += "Used in features:\n"
                    for feature in class_summary.features:
                        filecontext += f"{feature}\n"

                # Generate embeddings and store with embeddings.store_embeddings
                logger.info(f"  Storing embeddings for class: {class_summary.full_classname}")
                embedd_summary = f"Method {method.method_name}: {method.method_summary}"
                
                embeddings.store_class_description_embeddings('method', class_summary.full_classname, method.method_name, embedd_summary, filecontext, rel_path, class_storage.timestamp)

            for property in class_summary.properties:
                filecontext: str = ""
                if class_summary.features:
                    filecontext += "Used in features:\n"
                    for feature in class_summary.features:
                        filecontext += f"{feature}\n"

                # Generate embeddings and store with embeddings.store_embeddings
                logger.info(f"  Storing embeddings for class: {class_summary.full_classname}")
                embedd_summary = f"Property {property.property_name}: {property.property_summary}"
                
                embeddings.store_class_description_embeddings('property', class_summary.full_classname, property.property_name, embedd_summary, filecontext, rel_path, class_storage.timestamp)
        
        # Print progress
        if processed % 10 == 0:
            logger.info(f"Progress: {processed}/{total_classes} classes processed")

    # Save to file
    embeddings.store_all_classes()

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
    input_dir = os.path.abspath(args.input_dir)
    
    # Load configuration
    global config
    config_file_path = os.path.join(input_dir, ".ai-agent", f"config.json")
    config = load_config(config_file_path)
    
    # Initialize LLM execution
    global llm_execution, embeddings
    if config.llm.mode == 'mlx':
        logger.info("Initializing connection to MLX...")
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature, logger=llm_logger)
    elif config.llm.mode == 'ollama':
        logger.info("Initializing connection to Ollama...")
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url, logger=llm_logger)
    elif config.llm.mode == 'anthropic':
        logger.info("Initializing connection to Anthropic...")
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key, logger=llm_logger)
    else:
        raise ValueError(f"Unsupported LLM mode: {config.llm.mode}")
    
    # Initialize embeddings with config
    embeddings = Embeddings(config, logger)
    
    # Load the prompt templates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    mode = args.mode  # Can be None if not specified
    
    # Get file information with optional filter
    filtered_file_infos = get_filtered_files(input_dir, config.source_dirs, extensions=('.kt', '.java'), name_filter=args.filter)
    
    is_filtering_enabled = args.filter != None

    # Print out the filtered files with sizes
    logger.info("Filtered files found:")
    for i, file_info in enumerate(filtered_file_infos, 1):
        formatted_size = format_file_size(file_info.file_size)
        status = "[WILL PROCESS]" if file_info.is_allowed_by_filter else "[FILTERED OUT]"
        logger.info(f"{i}. {file_info.filepath} ({formatted_size}) {status}")
    
    allowed_count = sum(1 for f in filtered_file_infos if f.is_allowed_by_filter)
    logger.info(f"Total: {len(filtered_file_infos)} files found, {allowed_count} match filter")

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
    prompt_params = params_add_list_of_files(prompt_params, filtered_file_infos)
    prompt_params = params_add_project_context(prompt_params, input_dir)
    
    # Determine which processes to run based on mode
    run_pre_process = mode is None or mode == 'Pre'
    run_final_process = mode is None or mode == 'Final'
    run_embeddings = mode is None or mode == 'Embedd'

    if run_final_process:
        llm_execution.on_load()
        
    # Pre-processing step with TreeSitter
    if run_pre_process:
        logger.error("=== Error, pre-process not implemented in this version ===")

    if not run_pre_process and (run_final_process or run_embeddings):
        # If we're not running pre-process but need the data for later steps
        logger.info("=== Loading pre-processed data from storage ===")
        knowledge_store.read_storage_pre_process(f"{input_dir}/.ai-agent/db_preprocess.json")

    # Final processing step
    if run_final_process:
        logger.info("=== Running Final processing ===")
        knowledge_store.read_storage_final(f"{input_dir}/.ai-agent/db_final.json")

        is_force = args.force 
        is_update_only = args.update

        # Process with memory and get all class summaries
        class_summaries_final = final_process(
            filtered_file_infos, 
            prompt_params, 
            prompt_templates, 
            base_dir=input_dir, 
            is_filtering_enabled=is_filtering_enabled, 
            is_force=is_force, 
            is_update_only=is_update_only)
        
        # Save all class summaries to storage at once
        logger.info(f"Saving {len(class_summaries_final)} class summaries with memory to storage...")
        for classs in class_summaries_final:
            knowledge_store.save_class_description(classs)

        knowledge_store.dump_write_storage_final(f"{input_dir}/.ai-agent/db_final.json")

    if run_embeddings and not run_final_process:
        # If we're only running embeddings and not final process, we need to load final data
        logger.info("=== Loading final processed data from storage ===")
        knowledge_store.read_storage_final(f"{input_dir}/.ai-agent/db_final.json")

    # Embeddings processing step
    if run_embeddings:
        logger.info("=== Running Embeddings processing ===")
        process_embeddings(filtered_file_infos, base_dir=input_dir, project_context=prompt_params["projectcontext"])

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