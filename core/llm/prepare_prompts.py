import os
from typing import List, Dict

from knowledge.model import ClassDescription, FileInfo
from knowledge.knowledge_store import KnowledgeStore
from static_analysis.model.model import ClassStructure

def params_add_list_of_files(prompt_params: Dict, file_infos: List[FileInfo]) -> Dict:
    """
    Format the list_of_files string and add it to prompt_params
    
    Args:
        file_infos: List of FileInfo objects
        prompt_params: The parameters dictionary to update
        
    Returns:
        Updated prompt_params dictionary with list_of_files_str
    """
    # Format the list of files string
    list_of_files_str = "\n".join([f"{file_info.filepath}" for file_info in file_infos])
    
    # Add to prompt_params
    prompt_params["list_of_files"] = list_of_files_str
    
    return prompt_params

def params_add_field_descriptions(prompt_params: Dict) -> Dict:
    """
    Generate field descriptions for both models and add them to prompt_params
    
    Args:
        prompt_params: The parameters dictionary to update
        
    Returns:
        Updated prompt_params dictionary with field descriptions
    """
    # Generate field descriptions for summary model
    param_pre_process_field_descriptions = ""
    for name, field_info in ClassStructure.model_fields.items():
        param_pre_process_field_descriptions += (f"- {name}: {field_info.description}\n")
    
    # Generate field descriptions for final model
    param_final_process_field_descriptions = ""
    for name, field_info in ClassDescription.model_fields.items():
        param_final_process_field_descriptions += (f"- {name}: {field_info.description}\n")
    
    # Add to prompt_params
    prompt_params["field_descriptions_summary"] = param_pre_process_field_descriptions
    prompt_params["field_descriptions_final"] = param_final_process_field_descriptions
    
    return prompt_params

def params_add_ignored_packaged(prompt_params: Dict) -> Dict:
    """
    Add list of packages to ignore to prompt_params
    
    Args:
        prompt_params: The parameters dictionary to update
        
    Returns:
        Updated prompt_params dictionary with ignore_packages
    """
    # Define packages to ignore
    param_ignore_packages = ["androidx.*", "java.*", "android.os.*", "kotlinx.coroutines.*"]
    
    # Add to prompt_params
    prompt_params["ignore_packages"] = param_ignore_packages
    
    return prompt_params


def params_add_project_context(prompt_params: Dict, input_dir: str) -> Dict:
    """
    Add project context to prompt_params
    
    Args:
        prompt_params: The parameters dictionary to update
        input_dir: Input directory path
        
    Returns:
        Updated prompt_params dictionary with project context
    """
    # Define the path to the project context file
    project_context_path = os.path.join(input_dir, ".ai-agent", "project_context.txt")
    
    # Check if the file exists
    if os.path.exists(project_context_path):
        # Read the project context from file
        with open(project_context_path, "r", encoding="utf-8") as f:
            projectcontext = f.read()
    else:
        # Fallback to empty string if file doesn't exist
        projectcontext = ""
    
    # Add to prompt_params
    prompt_params["projectcontext"] = projectcontext
    
    return prompt_params

def params_add_containing_classes(prompt_params: Dict, rel_path: str, knowledge_store: KnowledgeStore) -> Dict:
    result = ""

    file_struct = knowledge_store.get_file_structure(rel_path)
    for cls in file_struct:
        result += f"- full_classname: `{cls.full_classname}`\n"
    
    # Add to prompt_params
    prompt_params["contain_classes"] = result
    
    return prompt_params