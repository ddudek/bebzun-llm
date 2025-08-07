from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class BuildContextQueriesLLMResult(BaseModel):
    """Search queries to find relevant knowledge about files in the project"""

    user_task_refined: str = Field(description="Refined user task (english only)")
    similarity_search_queries: List[str] = Field(description="List of vector similarity search queries (english only)")
    bm25_search_queries: List[str] = Field(description="List of bm25 queries")

class BuildContextGetFilesLLMResult(BaseModel):
    """Files to open that are relevant for the user task"""
    
    files_to_open: List[str] = Field(description="List of file paths to open")
    additional_search_queries: List[str] = Field(description="List of additional queries to find more classes")
    classes_not_related: List[str] = Field(description="List of full classname that are not related at all to the user task")
    explanation_why_classes_not_related: str = Field(description="Explain your response from \"classes_not_related\"")


class BuildContextStep3(BaseModel):
    """Check the final output"""
    
    files_to_open: List[str] = Field(description="List of file paths to open")
    additional_search_queries: List[str] = Field(description="List of additional queries to find more classes")
    classes_not_related: List[str] = Field(description="List of full classname that are not related at all to the user task")
    finish_summary: str = Field(description="Summary of the finished result")