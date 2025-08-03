from typing import Optional
from interact.memory.memory import Memory

class ChatState:
    def __init__(self, knowledge_store):
        self.search_used_count: int = 0
        self.get_file_used_count: int = 0

        self.memory: Memory = Memory(knowledge_store)