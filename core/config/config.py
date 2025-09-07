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

class OpenAIConfig(BaseModel):
    key: str = "YOUR_OPENAI_API_KEY"
    model: str = "gpt-4-turbo"
    temperature: float = 0.7
    url: str = "https://localhost:8080/v1"

class EmbeddingsConfig(BaseModel):
    model: str = "nomic-embed-text"
    vector_dimension: int = 768
    execution: Literal['mlx', 'ollama', 'openai'] = 'ollama'
    url: str = "https://localhost:8080/v1"

class LlmConfig(BaseModel):
    mode: Literal['mlx', 'ollama', 'anthropic', 'openai']
    mlx: MlxConfig = Field(default_factory=MlxConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    max_context: int = Field(default=128000)
    warn_context: int = Field(default=75000)
    min_context: int = Field(default=0)

class RerankerConfig(BaseModel):
    mode: Literal["none", "experimental-transfrormers-qwen3", "llama_rerank"] = "none"
    model: str = "bge-reranker-v2-m3"
    url: str = "http://127.0.0.1:8012/v1/rerank"

class Config(BaseModel):
    source_dirs: List[str]
    file_extensions: tuple[str] = Field(default=('.kt', '.java'))
    exclude: List[str] = Field(default=['.DS_Store', '.ai-agent', '.ai-agent-bak', '.git', '__pycache__', '.idea', 'build', 'gradle'])
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LlmConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    bench_rerank: List[dict] = Field(default=[])
    bench_embedd: List[dict] = Field(default=[])

def load_config(path: str = "config.json") -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return Config(**data)