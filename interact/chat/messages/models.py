class BaseMessage:
    """Represents a base message."""
    def __init__(self, content: str):
        self.content = content

class UserMessage(BaseMessage):
    """Represents a user message."""
    pass

class AssistantMessage(BaseMessage):
    """Represents an assistant message."""
    pass

class ToolObservationMessage(BaseMessage):
    """Represents a tool observation message."""
    def __init__(self, content: str, tool_name: str, is_error: bool = False):
        super().__init__(content)
        self.tool_name = tool_name
        self.is_error = is_error

class MemoryMessage(BaseMessage):
    """Represents a follow-up message."""
    pass