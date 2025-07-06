from typing import Optional
from interact.memory.memory import Memory

class ChatState:
    def __init__(self):
        self.current_step: int = 0
        self.new_step: int = 1 
        
        self.search_used_count: int = 0
        self.get_file_used_count: int = 0

        self.finish_unlocked: bool = False

        self.step_change: bool = False
        self.final_answer_provided: bool = False
        self.memory: Memory = Memory()