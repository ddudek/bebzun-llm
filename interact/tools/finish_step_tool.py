import logging
from interact.chat_state import ChatState

class FinishStepTool:
    name: str = "finish_step"
    description: str = (
        "Finishes the current step of the process.\n"
        "Parameters:\n"
        "- step_number (required): The number of the step to finish.\n"
        "Example: `<finish_step><step_number>1</step_number></finish_step>`")

    def run(self, chat_state: ChatState, step_number: str) -> str:
        """
        This tool doesn't perform an action in the traditional sense but signals
        a state change in the chat loop. The return value can be used for logging.
        """
        invalid_step_message = f"Error: Invalid step number provided: {step_number}"

        try:
            step = int(step_number)
            if step == chat_state.current_step:
                chat_state.new_step = chat_state.current_step + 1

                if step == 1:
                    return "Step 1 (gathering initial knowledge) is complete."
                if step == 2:
                    return "Step 2 (implementation gathering) is complete."
                else:
                    return f"Signal to finish step {step} received."
            else:
                return invalid_step_message
        except (ValueError, TypeError):
            return invalid_step_message
