from typing import Dict, Optional, List
import json
import os
from pathlib import Path
from knowledge.model import ClassDescription, ClassDescriptionExtended
from static_analysis.model.model import ClassStructure, ClassStructureDependency
from core.utils.file_utils import get_file_content_safe

class DepUsage:
        parent_structure: ClassStructure
        parent_description: Optional[ClassDescriptionExtended]
        dep: ClassStructureDependency
        dependency_structure: ClassStructure
        dependency_description: Optional[ClassDescriptionExtended]

class KnowledgeStore:
    def __init__(self):
        # Initialize dictionaries to store class information
        self.class_structure_dict: Dict[str, ClassStructure] = {}  # Index by classname
        self.descriptions_dict: Dict[str, ClassDescriptionExtended] = {}   # Index by classname: stores class final summary with file path
    
    def save_class_structure(self, storage_obj: ClassStructure):
        # Extract classnames from dependencies and store by classname
        classname = storage_obj.full_classname
        if classname and classname.strip():  # Only store non-empty class names
            self.class_structure_dict[classname] = storage_obj
    

    def save_class_description(self, storage_obj: ClassDescriptionExtended):
        # Extract classname from storage object and store by classname
        classname = storage_obj.class_summary.full_classname
        if classname and classname.strip():  # Only store non-empty class names
            self.descriptions_dict[classname] = storage_obj
    
    def get_class_structure(self, classname: str) -> Optional[ClassStructure]:
        """
        Retrieve stored information by classname
        Returns None if the classname is not found
        
        Args:
            classname: The name of the class to look up
        """
        # Look up by classname and return the class_summary from the storage object
        storage_obj = self.class_structure_dict.get(classname)
        return storage_obj if storage_obj else None
        
    def get_file_structure(self, filepath: str) -> List[ClassStructure]:
        """
        Retrieve all class summaries for a given filepath
        Returns an empty list if no classes are found for the filepath
        
        Args:
            filepath: The path to the file
        
        Returns:
            A list of ClassSummaryOutput objects for all classes in the file
        """
        # Filter class_storage to find all entries with matching file path
        result = []
        for storage_obj in self.class_structure_dict.values():
            if storage_obj.source_file == filepath:
                result.append(storage_obj)
        return result
    
    def get_class_description(self, classname: str) -> Optional[ClassDescription]:
        """
        Retrieve stored final class information by classname
        Returns None if the classname is not found
        
        Args:
            classname: The name of the class to look up
        """
        # Look up by classname and return the class_summary from the storage object
        storage_obj = self.descriptions_dict.get(classname)
        return storage_obj.class_summary if storage_obj else None
    
    def get_class_description_extended(self, classname: str) -> Optional[ClassDescriptionExtended]:
        """
        Retrieve stored final class information by classname
        Returns None if the classname is not found
        
        Args:
            classname: The name of the class to look up
        """
        # Look up by classname and return the class_summary from the storage object
        storage_obj = self.descriptions_dict.get(classname)
        return storage_obj if storage_obj else None
    
    def find_class_description_extended(self, query: str) -> Optional[ClassDescriptionExtended]:
        """
        Retrieve stored final class information by classname
        Returns None if the classname is not found
        
        Args:
            classname: The name of the class to look up
        """
        # Look up by classname and return the class_summary from the storage object
        for storage_obj in self.descriptions_dict.values():
            if query.lower() in storage_obj.class_summary.full_classname.lower():
                return storage_obj
    
    def get_file_description(self, filepath: str) -> List[ClassDescription]:
        """
        Retrieve all final class summaries for a given filepath
        Returns an empty list if no classes are found for the filepath
        
        Args:
            filepath: The path to the file
        
        Returns:
            A list of ClassFinalSummaryOutput objects for all classes in the file
        """
        # Filter final_file_storage to find all entries with matching file path
        result = []
        for storage_obj in self.descriptions_dict.values():
            if storage_obj.file == filepath:
                result.append(storage_obj.class_summary)
        return result
    
    def list_all_dependencies(self):
        """
        Print statistics about stored dependencies:
        - Number of unique files in class_storage and final_file_storage
        - List of all classnames in class_storage and final_file_storage
        """
        # Get unique file paths and statistics for class_storage
        unique_files_preprocess = set(storage_obj.source_file for storage_obj in self.class_structure_dict.values())
        total_files_preprocess = len(unique_files_preprocess)
        total_classes_preprocess = len(self.class_structure_dict)
        
        # Get unique file paths and statistics for final_file_storage
        unique_files_final = set(storage_obj.file for storage_obj in self.descriptions_dict.values())
        total_files_final = len(unique_files_final)
        total_classes_final = len(self.descriptions_dict)
        
        print(f"\nPreprocess storage statistics:")
        print(f"Total unique files: {total_files_preprocess}")
        print(f"Total classes across all files: {total_classes_preprocess}")
        
        print(f"\nFinal storage statistics:")
        print(f"Total unique files: {total_files_final}")
        print(f"Total classes across all files: {total_classes_final}")
        
        # Print all classnames from class_storage
        print(f"\nPreprocess class storage contents ({total_classes_preprocess} classes):")
        for classname in sorted(self.class_structure_dict.keys()):
            print(f"  {classname}")
            
        # Print all classnames from final_file_storage
        print(f"\nFinal class storage contents ({total_classes_final} classes):")
        for classname in sorted(self.descriptions_dict.keys()):
            print(f"  {classname}")
    
    def dump_write_storage_preprocess(self, filepath: str, metadata: dict):
        """
        Save the pre_process and final_process dictionaries to a JSON file
        
        Args:
            filepath: Path where the JSON file will be saved (default: storage/data.json)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data structure for serialization
        storage_data = {
            "metadata": metadata,
            "pre_process": {},
        }
        
        # Convert pre_process to serializable format
        # storage_obj is ClassSummaryOutput
        for classname, storage_obj in self.class_structure_dict.items():
            storage_data["pre_process"][classname] = storage_obj.dict()
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        print(f"Storage saved to {filepath}")

    
    def dump_write_storage_final(self, filepath: str):
        """
        Save the pre_process and final_process dictionaries to a JSON file
        
        Args:
            filepath: Path where the JSON file will be saved (default: storage/data.json)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data structure for serialization
        storage_data = {
            "final_process": {},
        }

        # Convert final_process to serializable format
        for classname, storage_obj in self.descriptions_dict.items():
            storage_data["final_process"][classname] = storage_obj.dict()
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        print(f"Storage saved to {filepath}")
    
    def read_storage_pre_process(self, filepath: str):
        """
        Read a JSON file and fill class_storage and file_storage dictionaries
        
        Args:
            filepath: Path to the JSON file to read (default: storage/data.json)
        """
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Storage file not found: {filepath}")
            return
        
        try:
            # Read from file
            with open(filepath, 'r') as f:
                storage_data = json.load(f)
            
            # Clear existing storage
            self.class_structure_dict.clear()
            
            # Populate class_storage
            for classname, storage_data_entry in storage_data.get("pre_process", {}).items():
                self.class_structure_dict[classname] = ClassStructure(**storage_data_entry)
            
            print(f"Storage loaded from {filepath}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error loading storage from {filepath}: {str(e)}")

    def read_storage_final(self, filepath: str, input_dir: str):
        """
        Read a JSON file and fill class_storage and file_storage dictionaries
        
        Args:
            filepath: Path to the JSON file to read (default: storage/data.json)
        """
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Storage file not found: {filepath}")
            return
        
        try:
            # Read from file
            with open(filepath, 'r') as f:
                storage_data = json.load(f)
            
            # Clear existing storage
            self.descriptions_dict.clear()
            
            # Populate final_file_storage
            for classname, storage_data in storage_data.get("final_process", {}).items():
                item = ClassDescriptionExtended(**storage_data)

                abs_file_path = os.path.join(input_dir, item.file)
                try:
                    file_size = os.path.getsize(abs_file_path)
                except:
                    print(f"Warning, cannot read file size: {abs_file_path}")
                    continue

                item.file_size = file_size
                self.descriptions_dict[classname] = item

            
            print(f"Storage loaded from {filepath}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error loading storage from {filepath}: {str(e)}")

    
    def get_class_dependencies(self, classname: str) -> List[ClassStructureDependency]:
        # Look up by classname and return the class_summary from the storage object
        storage_obj = self.class_structure_dict.get(classname)
        return storage_obj.dependencies if storage_obj else []
    
    def get_file_dependencies(self, rel_path: str) -> Dict[str, DepUsage]:
        # Look up by classname and return the class_summary from the storage object
        dependencies: Dict[str, DepUsage] = {}

        for parent_structure in self.get_file_structure(rel_path):
            for dependency in parent_structure.dependencies:
                dependency_structure = self.get_class_structure(dependency.full_classname)
                dependency_description = self.get_class_description_extended(dependency.full_classname)
                if dependency_structure is not None:
                    dep_usage = DepUsage()
                    dep_usage.parent_structure = parent_structure
                    dep_usage.dep = dependency
                    dep_usage.dependency_structure = dependency_structure
                    dep_usage.dependency_description = dependency_description
                    dependencies[dependency.full_classname] = dep_usage
        
        return dependencies
    
    def get_file_usages(self, rel_path: str) -> Dict[str, DepUsage]:
        usages: Dict[str, DepUsage] = {}
        
        for class_structure in self.get_file_structure(rel_path):
            target_classname = class_structure.full_classname
            
            for potential_use_parent in self.class_structure_dict.values():
                # Check if the class represented by potential_user_class_storage uses target_classname_to_find_usages_for
                for dep in potential_use_parent.dependencies:
                    if dep.full_classname == target_classname:
                        # The class from potential_user_class_storage uses target_classname_to_find_usages_for.
                        # Store the summary of the *using* class.

                        usage_description = self.get_class_description_extended(potential_use_parent.full_classname)

                        dep_usage = DepUsage()
                        dep_usage.parent_structure = potential_use_parent
                        dep_usage.parent_description = usage_description
                        dep_usage.dep = dep
                        dep_usage.dependency_structure = class_structure
                        dep_usage.dependency_description = None
                        usages[potential_use_parent.full_classname] = dep_usage
                        break
        return usages
    
    def get_class_usages(self, target_classname: str) -> Dict[str, DepUsage]:
        usages: Dict[str, DepUsage] = {}
        class_structure = self.get_class_structure(target_classname)
        
        for potential_use_parent in self.class_structure_dict.values():
                # Check if the class represented by potential_user_class_storage uses target_classname_to_find_usages_for
                for dep in potential_use_parent.dependencies:
                    if dep.full_classname == target_classname:
                        # The class from potential_user_class_storage uses target_classname_to_find_usages_for.
                        # Store the summary of the *using* class.

                        usage_description = self.get_class_description_extended(potential_use_parent.full_classname)

                        dep_usage = DepUsage()
                        dep_usage.parent_structure = potential_use_parent
                        dep_usage.parent_description = usage_description
                        dep_usage.dep = dep
                        dep_usage.dependency_structure = class_structure
                        dep_usage.dependency_description = None
                        usages[potential_use_parent.full_classname] = dep_usage
                        break
        return usages
    
    def prepare_file_context(
                         self,
                         base_dir,
                         rel_path, 
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