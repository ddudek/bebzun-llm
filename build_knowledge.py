import argparse
import os
import sys
from typing import List, Dict, Set
import asyncio

from datetime import datetime, timezone
from core.config.config import load_config
from core.utils.file_utils import get_file_content, is_binary_file, get_file_content_safe, format_file_size
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from knowledge.knowledge_store import KnowledgeStore
from knowledge.model import FileDescription, ClassDescription, ClassDescriptionExtended, FileInfo
from knowledge.embeddings import Embeddings
from static_analysis.model.model import ClassStructure, ClassStructureDependency
from core.llm.prepare_prompts import params_add_list_of_files, params_add_field_descriptions, params_add_ignored_packaged, params_add_project_context

# Initialize Storage as a global instance
knowledge_store = KnowledgeStore()
embeddings = None  # Will be initialized after config is loaded

workers_num: int = 1

async def final_process_file(worker_num: int, file_path: str, content: str, filecontext: str, prompt_params: Dict, prompt_templates: Dict, base_dir: str = None) -> FileDescription:
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

        response = await llm_execution.llm_invoke(worker_num, system_prompt, prompt=prompt, schema=FileDescription.model_json_schema())
        return FileDescription(**response)
    except Exception as e:
        error_msg = f"[Error generating summary: {str(e)}]"
        print(error_msg)
        return FileDescription(
            summary=error_msg,
            dependencies=[]
        )
    

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
            print(f"Warning: The source directory '{abs_src_path}' does not exist.")
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
    file_memory_classes = knowledge_store.get_file_structure(rel_path)

    print(f"Found {len(file_memory_classes)} classes for {rel_path}")
    
    # Dictionary to track dependency files and their usage count
    
    # Collect all distinct dependency files with their usage count
    for class_info in file_memory_classes if file_memory_classes else []:
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

async def final_worker(queue: asyncio.Queue, worker_num: int, prompt_params: Dict, prompt_templates: Dict,
                base_dir: str, lock: asyncio.Lock,
                processed_counter: List[int], total_files: int, all_class_summaries: List[ClassDescriptionExtended],
                processed_files: set) -> None:
    """
    Worker that processes files from the queue for final processing
    """
    while True:
        try:
            # Get item from queue with priority
            item = await queue.get()
            
            # Check if we received a sentinel value indicating we should exit
            if item is None:
                break
                
            # Unpack the item (priority, counter, file_info)
            file_info = item
            rel_path = file_info.filepath
            
            # Skip if already processed
            if rel_path in processed_files:
                continue
                
            # Mark as processed
            processed_files.add(rel_path)
            
            # Get dependency files with priority
            dependency_files_with_priority = get_dependency_files(rel_path, {})
            dependency_files = set(df for df, _ in dependency_files_with_priority)
            
            # Process the file using the extracted method
            await llm_final_process_file(worker_num, rel_path,
                                        knowledge_store.get_file_structure(rel_path),
                                        dependency_files,
                                        prompt_params, prompt_templates, base_dir, lock,
                                        processed_counter, total_files, all_class_summaries)
        
        except Exception as e:
            print(f"Error processing file {rel_path if 'rel_path' in locals() else 'unknown'}: {str(e)}")
        finally:
            queue.task_done()

async def llm_final_process_file(worker_num: int, rel_path: str, file_memory: List[ClassStructure], dependency_files: set,
                                prompt_params: Dict, prompt_templates: Dict, base_dir: str, lock: asyncio.Lock,
                                processed_counter: List[int], total_files: int, all_class_summaries: List[ClassDescriptionExtended]) -> None:
    """
    Process a file for final LLM processing and update shared data
    
    Args:
        worker_num: The worker number for logging
        rel_path: Relative path of the file
        file_memory: Memory information for the file from cache
        dependency_files: Set of dependency files
        prompt_params: Parameters for the prompts
        prompt_templates: The prompt templates to use
        base_dir: Base directory (needed for file operations)
        lock: Lock for thread-safe operations
        processed_counter: Counter for tracking progress
        total_files: Total number of files for progress reporting
        all_class_summaries: List to collect class summaries
    """

    add_usages_summaries = True
    add_usages_methods = False
    add_dependency_summaries = True
    add_dependency_methods = True
    add_dependency_variables = True

    # Create filecontext string with class information
    dependencies: Dict[str, ClassStructure] = {}
    if file_memory: # Check if file_memory is not None or empty
        for class_info in file_memory:
            for dependency in class_info.dependencies:
                dependency_info = knowledge_store.get_class_structure(dependency.full_classname)
                if dependency_info is not None:
                    dependencies[dependency.full_classname] = dependency_info


    class DepUsage:
        dep: ClassStructureDependency
        reference: ClassStructure


    # Convert relative path to absolute for file operations
    abs_file_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
    
    # Get file content
    content = get_file_content_safe(abs_file_path)

    usages: Dict[str, DepUsage] = {}
    if file_memory: # Check if file_memory is not None or empty
        for class_info in file_memory:
            target_classname_to_find_usages_for = class_info.full_classname
            
            for potential_user_class_storage in knowledge_store.class_structure_dict.values():
                # Check if the class represented by potential_user_class_storage uses target_classname_to_find_usages_for
                for dep in potential_user_class_storage.dependencies:
                    if dep.full_classname == target_classname_to_find_usages_for:
                        # The class from potential_user_class_storage uses target_classname_to_find_usages_for.
                        # Store the summary of the *using* class.

                        dep_usage = DepUsage()
                        dep_usage.dep = dep
                        dep_usage.reference = potential_user_class_storage
                        usages[potential_user_class_storage.full_classname] = dep_usage
                        # Found usage, no need to check further dependencies of this potential_user_class_storage
                        # for the current target_classname_to_find_usages_for.
                        break
    
    filecontext: str = ""
    for key, dependency in dependencies.items():
        filecontext += f"\n# Dependency class: {dependency.full_classname}\n"
        dependency_final_info = knowledge_store.get_class_description(dependency.full_classname)
        if dependency_final_info:
            if add_dependency_summaries:
                filecontext += f"{dependency_final_info.summary}\n"
            if add_dependency_methods:
                for method in dependency_final_info.methods:
                    if method.method_name in content:
                        filecontext += f"- Method {method.method_name}: {method.method_summary}\n"
            if add_dependency_variables:
                for variable in dependency_final_info.variables:
                    if variable.variable_name in content:
                        filecontext += f"- Variable {variable.variable_name}: {variable.variable_summary}\n"
            continue
    

    for key, dep_usage in usages.items():
        # dep_usage is of type DepUsage
        # dep_usage.reference is ClassSummaryOutput (contains file path and summary of the class *using* the dependency)
        # dep_usage.dep is DependencyStruct (contains full_classname of the *used* dependency and usage_lines)
        filecontext += "\n"
        using_class_summary = dep_usage.reference
        used_class_name = dep_usage.dep.full_classname
        usage_file_rel_path = dep_usage.reference.source_file # Relative path of the file where the usage occurs
        usage_line_numbers = dep_usage.dep.usage_lines # List of line numbers (1-based) where usage occurs

        filecontext += f"# Usage: {using_class_summary.full_classname} is using {used_class_name} in file '{usage_file_rel_path}'\n"
        
        final_context = knowledge_store.get_class_description(using_class_summary.full_classname)
        if final_context:
            if add_usages_summaries:
                filecontext += f"{final_context.simple_classname} explanation: {final_context.summary}\n"
            if add_usages_methods:
                for method in final_context.methods:
                    if "get" not in method.method_name:
                        filecontext += f"- Method {method.method_name}: {method.method_summary}\n"
        
        # Construct absolute path for reading the file where usage occurs
        # base_dir is a parameter of llm_final_process_file
        abs_usage_file_path = os.path.join(base_dir, usage_file_rel_path) if base_dir else usage_file_rel_path
        
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
                    diff_window = 4
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
                                filecontext += f"  Context for usage line(s): {', '.join(map(str, sorted_usages))}\n"
                            
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
            filecontext += f"  [INFO] Could not read content of '{usage_file_rel_path}' or it is a binary file.\n"
        filecontext += "\n" # Add a newline for separation between different usage entries in filecontext

    # Print dependency files for debugging
    print(f"Dependency files for {rel_path}: {list(dependency_files)}")
    print(f"Processing order: Dependencies processed before this file")
    
    if content is None:
        print(f"  [Binary file, skipping] {rel_path}")
        return
    
    # Generate summary
    print(f"\nProcessing #{worker_num}: {rel_path}")
    summary = await final_process_file(worker_num, rel_path, content, filecontext, prompt_params, prompt_templates, base_dir)
    print(f"Done #{worker_num}")
    
    # Process results and update shared data with lock
    async with lock:
        # Update processed counter
        processed_counter[0] += 1
        current_processed = processed_counter[0]
        
        # Print progress
        print(f"\n[{current_processed}/{total_files}] {rel_path}")
        
        # Process and save results
        for classs in summary.classes:
            print(f"\n  Class: {classs.full_classname}")
            # print(f"  Simple: {classs.simple_classname}")
            print(f"  Summary: {classs.summary}")
            print(f"  Category: {classs.category}")
            
            # Create ClassFinalSummaryStorage object
            storage_obj = ClassDescriptionExtended(
                class_summary=classs,
                file=rel_path
            )
            
            # Add to collection
            all_class_summaries.append(storage_obj)
            knowledge_store.save_class_description(storage_obj)
            knowledge_store.dump_write_storage_final(f"{base_dir}/.ai-agent/final.json")
        
        # Periodically report progress
        if current_processed % 10 == 0:
            print(f"Progress: {current_processed}/{total_files} files processed")

def final_process(file_infos: List[FileInfo], prompt_params: Dict, prompt_templates: Dict, base_dir: str = None) -> List[ClassDescriptionExtended]:
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
    
    # First, collect all dependencies for each file
    for file_info in file_infos:
        if file_info.is_allowed_by_filter:
            rel_path = file_info.filepath
            dependency_files_with_priority[rel_path] = 0
            get_dependency_files(rel_path, dependency_files_with_priority)

    # Convert to list of tuples and sort by priority (highest first)
    files_to_process_pre = [(file, priority) for file, priority in dependency_files_with_priority.items()]
    files_to_process_pre.sort(key=lambda x: x[1], reverse=True)
    
    total_files = len(files_to_process_pre)
    print(f"Processing {total_files} files for final summary (including dependencies)")
    print("Generating summaries with context...")
    
    # List to collect all class summaries
    all_class_summaries = []
    files_to_process = []

    for file in files_to_process_pre:
        print(file)

    for file in files_to_process_pre:
        file_path = file[0]
        file_priority = file[1]
        file_info = file_info_map[file_path]
        already_processed = knowledge_store.get_file_description(file_path)
        if file_info.is_allowed_by_filter and not already_processed:
            files_to_process.append((file_info, file_priority))
        elif not already_processed:
            files_to_process.append((file_info, file_priority))
    
    # Run the async processing
    async def process_files_async():
        # Create a priority queue
        queue = asyncio.Queue()
        
        # Set to track processed files
        processed_files = set()

        # Print queue order before processing
        print("\nQueue processing order (lower priority number = higher actual priority):")

        for i, (file_info, priority) in enumerate(files_to_process):
            print(f"{i+1}. Priority: {priority} - {file_info.filepath}")
            await queue.put(file_info)

        # Shared counter for tracking progress
        processed_counter = [0]
        
        # Create a lock for thread-safe operations
        lock = asyncio.Lock()
        
        # Create worker tasks
        tasks = []
        for i in range(workers_num):  # Create worker tasks
            task = asyncio.create_task(
                final_worker(queue, i, prompt_params, prompt_templates, base_dir,
                    lock, processed_counter, total_files, all_class_summaries, processed_files)
            )
            tasks.append(task)
        
        # Wait for all files to be processed
        await queue.join()
        
        # Send sentinel values to stop workers
        for _ in range(workers_num):
            await queue.put(None)
            
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        print(f"All files processed in dependency order")
    
    # Run the async event loop
    asyncio.run(process_files_async())
    
    # Return the collected summaries
    return all_class_summaries

def process_embeddings(file_infos: List[FileInfo], base_dir: str = None, project_context: str = None) -> None:
    embeddings.initialize(base_dir)
    """
    Process embeddings for all classes in storage.final_process
    
    Args:
        file_infos: List of FileInfo objects (containing filepath, file_size, and modified_timestamp)
        base_dir: Base directory (needed for file operations)
        project_context: The project context string
    """
    total_classes = len(knowledge_store.descriptions_dict)
    print(f"Processing {total_classes} classes for embeddings")
    print("Generating embeddings...")
    
    processed = 0

    # Iterate through each class in storage.final_process
    for classname, class_storage in knowledge_store.descriptions_dict.items():
        processed += 1
        rel_path = class_storage.file
        
        print(f"\n[{processed}/{total_classes}] Processing class: {classname}")
        
        # Get dependencies directly from class_storage.class_summary.dependencies
        # filecontext: str = f"{project_context}\n" if project_context else ""
        filecontext: str = ""
        dependencies: Dict[str, ClassStructure] = {}
        
        # Get dependencies from the class summary
        class_summary = class_storage.class_summary
        class_preprocess = knowledge_store.get_class_structure(class_summary.full_classname)
        print(f"  Getting dependencies for class: {class_summary.full_classname}")
        
        if class_summary.features:
            filecontext += "Used in features:\n"
            for feature in class_summary.features:
                filecontext += f"{feature}\n"

        if class_summary.questions:
            filecontext += "Example questions:\n"
            for question in class_summary.questions:
                filecontext += f"{question}\n"
        
        # Get file's last modification time from the FileInfo object if available
        timestamp = None
        # Try to find the matching FileInfo object for this file
        matching_file_info = next((fi for fi in file_infos if fi.filepath == rel_path), None)
        
        if matching_file_info:
            # Use the timestamp from the FileInfo object
            timestamp = matching_file_info.modified_timestamp.isoformat()
        else:
            # Fall back to checking the file system if no matching FileInfo is found
            abs_file_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
            if os.path.exists(abs_file_path):
                mtime = os.path.getmtime(abs_file_path)
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            else:
                # Use current time if file doesn't exist
                timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate embeddings and store with embeddings.store_embeddings
        print(f"  Storing embeddings for class: {class_summary.full_classname}")
        embedd_summary = f"{class_summary.summary}"
        if class_summary.methods:
            embedd_summary += "\nMethods:"
            for method in class_summary.methods:
                embedd_summary+=(f"\n Method `{method.method_name}`: {method.method_summary}")

        if class_summary.variables:
            embedd_summary += "\nProperties:"
            for variable_entry in class_summary.variables:
                embedd_summary+=(f"\n Property `{variable_entry.variable_name}`: {variable_entry.variable_summary}")

        embeddings.store_class_description_embeddings(class_summary.full_classname, embedd_summary, filecontext, rel_path, timestamp)
        
        # Print progress
        if processed % 10 == 0:
            print(f"Progress: {processed}/{total_classes} classes processed")

    # Save to file
    embeddings.store_all_classes()

llm_execution = None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate summaries for files in a directory.')
    parser.add_argument('-i', '--input-dir', help='Directory containing files to summarize. Overrides config file.')
    
    parser.add_argument('-m', '--mode', choices=['Pre', 'Final', 'Embedd'],
                        help='Processing mode: Pre (TreeSitter preprocessing), Final (final process only), Embedd (embeddings only)')
    parser.add_argument('-filter', help='Filter files by name (only process files containing this string in filename). '
                                        'Prefix with "!" to invert (exclude files containing the string)')
    args = parser.parse_args()
    

    # Determine input directory    
    input_dir = os.path.abspath(args.input_dir)
    
    # Load configuration
    config_file_path = os.path.join(input_dir, ".ai-agent", f"config.json")
    config = load_config(config_file_path)
    
    # Initialize LLM execution
    global llm_execution, embeddings
    if config.llm.mode == 'mlx':
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature)
    elif config.llm.mode == 'ollama':
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url)
    elif config.llm.mode == 'anthropic':
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key)
    else:
        raise ValueError(f"Unsupported LLM mode: {config.llm.mode}")
    
    # Initialize embeddings with config
    embeddings = Embeddings(config)

    
    # Load the prompt templates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    mode = args.mode  # Can be None if not specified
    
    # Get file information with optional filter
    filtered_file_infos = get_filtered_files(input_dir, config.source_dirs, extensions=('.kt', '.java'), name_filter=args.filter)
    
    # Print out the filtered files with sizes
    print("\nFiltered files found:")
    for i, file_info in enumerate(filtered_file_infos, 1):
        formatted_size = format_file_size(file_info.file_size)
        status = "[WILL PROCESS]" if file_info.is_allowed_by_filter else "[FILTERED OUT]"
        print(f"{i}. {file_info.filepath} ({formatted_size}) {status}")
    
    allowed_count = sum(1 for f in filtered_file_infos if f.is_allowed_by_filter)
    print(f"Total: {len(filtered_file_infos)} files found, {allowed_count} will be processed\n")

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
        print("Initializing connection to Ollama...")
        llm_execution.on_load()
        
    # Pre-processing step with TreeSitter
    if run_pre_process:
        print("\n=== Error, pre-process not implemented in this version ===")

    if not run_pre_process and (run_final_process or run_embeddings):
        # If we're not running pre-process but need the data for later steps
        print("\n=== Loading pre-processed data from storage ===")
        knowledge_store.read_storage_pre_process(f"{input_dir}/.ai-agent/preprocess.json")

    # Final processing step
    if run_final_process:
        print("\n=== Running Final processing ===")
        knowledge_store.read_storage_final(f"{input_dir}/.ai-agent/final.json")
        
        # Process with memory and get all class summaries
        class_summaries_final = final_process(filtered_file_infos, prompt_params, prompt_templates, base_dir=input_dir)
        
        # Save all class summaries to storage at once
        print(f"\nSaving {len(class_summaries_final)} class summaries with memory to storage...")
        for classs in class_summaries_final:
            knowledge_store.save_class_description(classs)

        knowledge_store.dump_write_storage_final(f"{input_dir}/.ai-agent/final.json")

    if run_embeddings and not run_final_process:
        # If we're only running embeddings and not final process, we need to load final data
        print("\n=== Loading final processed data from storage ===")
        knowledge_store.read_storage_final(f"{input_dir}/.ai-agent/final.json")

    # Embeddings processing step
    if run_embeddings:
        print("\n=== Running Embeddings processing ===")
        process_embeddings(filtered_file_infos, base_dir=input_dir, project_context=prompt_params["projectcontext"])

    print("\nSummary generation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)