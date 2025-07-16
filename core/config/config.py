from pydantic import BaseModel, Field
from typing import List, Literal, Union
import json
import os

class MlxConfig(BaseModel):
    model: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    temperature: float = 0.7

class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.7

class AnthropicConfig(BaseModel):
    key: str = "YOURYOUR_ANTHROPIC_API_KEY"
    model: str = "claude-3-opus-20240229"

class EmbeddingsConfig(BaseModel):
    model: str = "nomic-embed-text"
    vector_dimension: int = 768
    execution: Literal['mlx', 'ollama'] = 'ollama'

class LlmConfig(BaseModel):
    mode: Literal['mlx', 'ollama', 'anthropic']
    mlx: MlxConfig = Field(default_factory=MlxConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    max_context: int = Field(default=128000)
    warn_context: int = Field(default=90000)

class Config(BaseModel):
    source_dirs: List[str]
    llm: LlmConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

def load_config(path: str = "config.json") -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return Config(**data)