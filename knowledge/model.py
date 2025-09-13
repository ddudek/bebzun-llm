from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class MethodDescription(BaseModel):
    method_name: str = Field(description="Name of the method")
    method_summary: str = Field(description="Explanation of what this method does, at least 2 sentences.")
    class Config:
        @staticmethod
        def json_schema_extra(schema: dict[str, any], model: type['MethodDescription']) -> None:
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)

    def getName(self) -> str:
        return self.method_name
    
    def getSummary(self) -> str:
        return self.method_summary
class VariableDescription(BaseModel):
    property_name: str = Field(description="Name of the variable")
    property_summary: str = Field(description="Explanation of what this variable is and how it behaves. If the variable is modified inside the class, please provide at least 2 sentences how it's being changed.")
    class Config:
        @staticmethod
        def json_schema_extra(schema: dict[str, any], model: type['VariableDescription']) -> None:
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)

    def getName(self) -> str:
        return self.property_name
    
    def getSummary(self) -> str:
        return self.property_summary

class ClassDescription(BaseModel):
    """Summary of the class"""

    simple_classname: str = Field(description="Simple class name without package (important!)")
    full_classname: str = Field(description="Full class name including package (important!)")
    summary: str = Field(description="An explanation of what this class does or contains. Begin with simple class name(!), then continue with explanation. The purpose of this summary will be to explain the class as a context to understand how the project works. Please explain how this class behaves, which and how other classes use it. For Enums and sealed data classes like UI states please explain what particular state causes. Include details that are not related to the filename.")
    category: Literal['UI', 'Logic', 'Data', 'Testing', 'Other'] = Field(description="Category of this class. UI: presentation layer of the app. Logic: classes containg important logic implemented. Data: External layer of the app, using database, 3rd party library wrappers, platform APIs. Testing: Test helpers and test classes. Other: Other usages.")
    questions: List[str] = Field(description="3 general questions about the project that this code would be a part of explanation. Avoid mentioning this class explicitly, focus on what is it used for, or which bigger feature uses it.")
    features: List[str] = Field(description="List of at least 3 features that this code relates to")
    methods: List[MethodDescription] = Field(description="List of all public and private methods")
    properties: List[VariableDescription] = Field(description="List of all public variables and properties, and private static properties.")
    class Config:
        @staticmethod
        def json_schema_extra(schema: dict[str, any], model: type['ClassDescription']) -> None:
            for prop in schema.get('properties', {}).values():
                prop.pop('title', None)
                prop.pop('description', None)

    def describe(self, bullet: str = " ", check_in_content: Optional[str] = None) -> str:
        summary = f"{self.summary}"
        if self.methods:
            for method in self.methods:
                if not check_in_content or (check_in_content and method.method_name in check_in_content):
                    summary+=(f"\n{bullet}Method `{method.method_name}`: {method.method_summary}")

        if self.properties:
            for property_entry in self.properties:
                if not check_in_content or (check_in_content and property_entry.property_name in check_in_content):
                    summary+=(f"\n{bullet}Property `{property_entry.property_name}`: {property_entry.property_summary}")
        return summary
    
    def find_method(self, name: str) -> Optional[MethodDescription]:
        for item in self.methods:
            if item.method_name == name:
                return item
    
    def find_property(self, name: str) -> Optional[VariableDescription]:
        for item in self.properties:
            if item.property_name == name:
                return item

class ClassDescriptionExtended(BaseModel):
    """Storage model for ClassFinalSummaryOutput with file path"""
    
    class_summary: ClassDescription = Field(description="The class summary output")
    file: str = Field(description="Relative path of the source file")
    file_size: int = Field(description="File size", default=-1)
    version: int = Field(description="Modification timestamp of the file when the analysis was performed", default=0)


class FileDescription(BaseModel):
    """Summary of the file"""

    classes: List[ClassDescription] = Field(description="Classes that are contained in this file")


class FileInfo(BaseModel):
    """Information about a file in the project"""
    
    filepath: str = Field(description="Relative path of the file")
    file_size: int = Field(description="Size of the file in bytes")
    version: datetime = Field(description="Last modification timestamp of the file")
    is_allowed_by_filter: bool = Field(description="Whether the file is allowed by the filter", default=True)
