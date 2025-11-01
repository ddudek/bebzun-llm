from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class AffectedArea(BaseModel):
    """Area to search to find relevant knowledge about files in the project"""
    affected_area_summary: str = Field(description="Summary of the affected area or feature related to the given task")
    similarity_search_queries: List[str] = Field(description="One or more search query for vector similarity search (english only) for this affected area")
    bm25_search_queries: List[str] = Field(description="One or more bm25 queries for this affected area")

class BuildContextQueriesLLMResult(BaseModel):
    """Search queries to find relevant knowledge about files in the project"""

    user_task_refined: str = Field(description="Refined user task (english only)")
    #similarity_search_queries: List[str] = Field(description="List of vector similarity search queries (english only)")
    #bm25_search_queries: List[str] = Field(description="List of bm25 queries")
    affected_areas: List[AffectedArea] = Field(description="List of areas of features that are affected by the given task and search queries to find the related code")

class BuildContextGetFilesLLMResult(BaseModel):
    """Files to open that are relevant for the user task"""
    
    files_to_open: List[str] = Field(description="List of file paths to open")
    additional_search_area: Optional[AffectedArea] = Field(description="Additional search area to find more knowledge or classes")
    classes_not_related: List[str] = Field(description="List of full classname that are not related at all to the user task (do not remove classes that add some context to the related code)")
    explanation_why_classes_not_related: str = Field(description="Explain your response from \"classes_not_related\"")


class BuildContextStep3(BaseModel):
    """Check the final output"""
    
    files_to_open: List[str] = Field(description="List of file paths to open")
    additional_search_area: Optional[AffectedArea] = Field(description="Additional search area to find more knowledge or classes (optional)")
    finish_summary: str = Field(description="Summary of the finished result")
    classes_not_related: List[str] = Field(description="List of full classname that are not related at all to the user task (do not remove classes that add some context to the related code)")