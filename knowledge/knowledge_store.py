from typing import Dict, Optional, List
import json
import os
from pathlib import Path
from knowledge.model import ClassDescription, ClassDescriptionExtended
from static_analysis.model.model import ClassStructure, ClassStructureDependency

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

    def read_storage_final(self, filepath: str):
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
                self.descriptions_dict[classname] = ClassDescriptionExtended(**storage_data)
            
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
        
        for potential_use_class_structure in self.class_structure_dict.values():
            # Check if the class represented by potential_user_class_storage uses target_classname_to_find_usages_for
            for dep in potential_use_class_structure.dependencies:
                if dep.full_classname == target_classname:
                    # The class from potential_user_class_storage uses target_classname_to_find_usages_for.
                    # Store the summary of the *using* class.

                    usage_description = self.get_class_description_extended(potential_use_class_structure.full_classname)

                    dep_usage = DepUsage()
                    dep_usage.dep = dep
                    dep_usage.structure = potential_use_class_structure
                    dep_usage.dependency_description = usage_description
                    usages[potential_use_class_structure.full_classname] = dep_usage
                    break
        return usages