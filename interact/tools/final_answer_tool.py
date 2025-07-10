import logging
from interact.chat_state import ChatState

class FinalAnswerTool:
    name: str = "final_answer"
    description: str = (
        "Provides the final answer to the user's task based on the gathered context.\n"
        "Use this tool when you have sufficient information to answer the user's request.\n"
        "Parameters:\n"
        "- answer (required): The comprehensive final answer.\n"
        "Example: `<final_answer><answer>The final answer is...</answer></final_answer>`"
    )

    def run(self, chat_state: ChatState, answer: str) -> str:
        """
        Returns the final answer.
        """
        chat_state.final_answer_provided = True
        return ""