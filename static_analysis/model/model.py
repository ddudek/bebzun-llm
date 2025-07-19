"""
Pydantic models for representing code analysis results.

These models define the structure of the analysis output, including classes,
their dependencies, and methods.
"""

from typing import List
from pydantic import BaseModel, Field

class ClassStructureDependency(BaseModel):
    """A class or interface used by the class being summarized"""

    full_classname: str = Field(description="The fully-qualified name of a class or interface being used including package (e.g., `com.example.MyClass`)")
    usage_lines: List[int] = Field(description="Line numbers where this dependency is used", default_factory=list)

class ClassStructureMethod(BaseModel):
    """Method description"""
    name: str = Field(description="name of the function")
    definition_start: int = Field(description="Line number where the method definition starts")
    definition_end: int = Field(description="Line number where the method definition ends")

class ClassStructure(BaseModel):
    """Summary of a class or interface"""

    simple_classname: str = Field(description="Simple name of a class or interface")
    full_classname: str = Field(description="The fully-qualified name")
    dependencies: List[ClassStructureDependency] = Field(default_factory=list)
    public_methods: List[ClassStructureMethod] = Field(default_factory=list)
    source_file: str = Field(description="Path to the source file containing this class")
    timestamp: int = Field(description="Modification timestamp of the file when the analysis was performed", default=0)

class FileStructure(BaseModel):
    """Container for the entire analysis output"""
    
    classes: List[ClassStructure] = Field(description="List of all classes analyzed", default_factory=list)