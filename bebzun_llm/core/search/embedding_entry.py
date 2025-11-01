from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass
class EmbeddingEntry:
    type: str
    detail: str
    full_classname: str
    rel_path: str
    embedding: List[float]
    version: int

    def to_dict(self) -> Dict:
        return asdict(self)