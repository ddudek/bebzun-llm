import argparse
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

from core.config.config import load_config
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from knowledge.embeddings import Embeddings
from knowledge.knowledge_store import KnowledgeStore
from knowledge.model import ClassDescription
from core.llm.prepare_prompts import params_add_project_context
from interact.memory.memory import Memory
from interact.tools.list_files_tool import ListFilesTool
from interact.tools.get_file_content_tool import GetFileContentTool
from interact.tools.search_knowledge_tool import SearchKnowledgeTool
from interact.tools.finish_step_tool import FinishStepTool
from interact.tools.final_answer_tool import FinalAnswerTool
from interact.chat_state import ChatState
from interact.chat.messages.models import AssistantMessage, BaseMessage, FollowUpMessage, ToolObservationMessage, UserMessage, UnlockToolMessage, MemoryMessage
from core.utils.logging_utils import setup_logging, setup_llm_logger


def get_system_prompt(tools: List[Any], input_dir: str) -> str:
    tools_description = "\n".join([f"## {t.name}\nDescription: {t.description}\n" for t in tools])
    prompt_params = params_add_project_context(prompt_params={}, input_dir=input_dir)
    
    prompt_path = os.path.join("interact", "chat", "system_prompt.txt")
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
        
    project_context_val = prompt_params.get('projectcontext', '')

    return prompt_template.format(
        project_context=project_context_val,
        tools_description=tools_description
    )

def get_folloup_prompt(chat_state: ChatState, tools: List[Any]) -> str:
    if chat_state.current_step == 1:
        if chat_state.search_used_count > 1:
            return "\nDo you want to search again for more knowledge or finish this step? You must use one of these tools:\n" + ("\n".join([f"- {t.name}" for t in tools]))
        else:
            return "\nPlease use search_knowledge_tool again with different queries."
    if chat_state.current_step == 2:
        return "\nDo you want to find more code or finish this step? You must use one of these tools:\n" + ("\n".join([f"- {t.name}" for t in tools]))
    
    return ""

def messages_to_llm_input(messages: List[BaseMessage], chat_state: ChatState, STEP_TOOLS, input_dir: str) -> List[Dict[str, Any]]:
    """Concatenates a list of agent message objects to a short list with a single memory state."""
    output = []
    system_prompt = get_system_prompt(STEP_TOOLS[chat_state.current_step], input_dir)
    output.append({"role": "system", "content": system_prompt})

    memory_output = f"\n{chat_state.memory.get_formatted_memory(chat_state.current_step)}\n"
    
    # Find the index of the last relevant tool message
    last_tool_idx = -1
    for i, msg in reversed(list(enumerate(messages))):
        if isinstance(msg, ToolObservationMessage) and msg.tool_name in ["read_file", "search_knowledge_tool"]:
            last_tool_idx = i
            break

    # 1. Create a list of messages with memory output injected
    processed_messages = []
    if last_tool_idx != -1:
        for i, msg in enumerate(messages):
            processed_messages.append(msg)
            if i == last_tool_idx:
                processed_messages.append(MemoryMessage(content=memory_output))
    else:
        # Find last user message to insert memory before it
        last_user_idx = -1
        for i, msg in reversed(list(enumerate(messages))):
            if isinstance(msg, UserMessage):
                last_user_idx = i
                break
        if last_user_idx != -1:
            messages.insert(last_user_idx, MemoryMessage(content=memory_output))
        else:
            messages.append(MemoryMessage(content=memory_output))
        processed_messages = messages

    # 2. Concatenate messages
    current_content = ""
    current_role = None

    for msg in processed_messages:
        role = None
        if isinstance(msg, AssistantMessage):
            role = "assistant"
        else:
            role = "user"

        if current_role and role != current_role:
            output.append({"role": current_role, "content": current_content.strip()})
            current_content = ""
        
        current_role = role
        current_content += f"\n{msg.content}"

    if current_role and current_content:
        output.append({"role": current_role, "content": current_content.strip()})

    return output

def main():
    parser = argparse.ArgumentParser(description='Chat with Ollama model with file listing and embeddings search capabilities.')
    parser.add_argument('-i', '--input-dir', required=True, help='Directory to list files from. Overrides config file.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level.')
    parser.add_argument('--log-file', help='Path to the log file.')
    parser.add_argument('--llm-log-file', help='Path to the LLM log file.')
    parser.add_argument('-p', '--prompt', help='Initial user prompt.')

    args = parser.parse_args()
    
    global logger
    global llm_logger
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    llm_logger = setup_llm_logger(log_level=args.log_level, log_file=args.llm_log_file)

    # Determine input directory    
    input_dir = os.path.abspath(args.input_dir)
    
    # Load configuration
    config_file_path = os.path.join(input_dir, ".ai-agent", f"config.json")
    config = load_config(config_file_path)

    
    embeddings_instance = None
    try:
        embeddings_instance = Embeddings(config, logger)
        if not embeddings_instance.initialize(input_dir):
            logger.warning("Embeddings storage is not available. Context search will be disabled.")
            embeddings_instance = None
    except Exception:
        logger.exception("Error initializing embeddings:")
        logger.warning("Context search will be disabled.")
        embeddings_instance = None

    knowledge = KnowledgeStore()
    knowledge_file_path = os.path.join(input_dir, ".ai-agent", f"db_final.json")
    if os.path.exists(knowledge_file_path):
        knowledge.read_storage_final(knowledge_file_path)
        logger.info(f"Successfully loaded knowledge db from {knowledge_file_path}")
    else:
        logger.warning(f"Warning: knowledge db file not found at {knowledge_file_path}. Context might be incomplete.")

         # Initialize LLM execution
    if config.llm.mode == 'mlx':
        print(f"Loading MLX model: {config.llm.mlx.model}")
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature, logger=llm_logger)
    elif config.llm.mode == 'ollama':
        print(f"Loading Ollama model: {config.llm.ollama.model}")
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url, logger=llm_logger)
    elif config.llm.mode == 'anthropic':
        print(f"Connection to Anthropic model: {config.llm.anthropic.model}")
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key, logger=llm_logger)
    else:
        raise ValueError(f"Unsupported LLM mode: {config.llm.mode}")

    final_answer_tool = FinalAnswerTool()
    step1_tools = [FinishStepTool()]
    step2_tools = [FinishStepTool()]
    step3_tools = [final_answer_tool]

    search_tool_name = "search_knowledge_tool"
    if embeddings_instance and embeddings_instance.is_loaded():
        search_knowledge_tool = SearchKnowledgeTool(embeddings=embeddings_instance, knowledge_store=knowledge, logger=logger)
        search_tool_name = search_knowledge_tool.name
        step1_tools.append(search_knowledge_tool)
        step2_tools.append(search_knowledge_tool)
        step3_tools.append(search_knowledge_tool)

    if config.source_dirs:
        list_files_tool = ListFilesTool(base_path=input_dir, source_dirs=config.source_dirs, logger=logger)
        get_file_content_tool = GetFileContentTool(base_path=input_dir, source_dirs=config.source_dirs, logger=logger)
        step2_tools.append(list_files_tool)
        step2_tools.append(get_file_content_tool)
        step3_tools.append(list_files_tool)
        step3_tools.append(get_file_content_tool)
    
    llm_initialized = False
    
    logger.info(f"Chat initialized with {llm_execution.model_desc()} model.")
    if args.input_dir:
        logger.info(f"File tools enabled for directory: {args.input_dir}")
    if embeddings_instance and embeddings_instance.is_loaded():
        logger.info("Embeddings context search enabled.")
    logger.info("Type 'exit' to end the conversation.")

    STEP_PROMPTS = {
        1: f"""Current Step: 1. Please use {search_tool_name} to find knowledge for this user task:
<user_task>
{{user_input}}
</user_task>""",
        2: f"""Current Step: 2.\nNow, proceed with step 2: Get implementation details by reading the relevant files using the `read_file` tool""",
        3: f"""Current Step: 3.\nNow, proceed with step 3: Answer the user's original task based on the knowledge and code you have gathered. When you are ready to provide the final answer, use the `final_answer` tool."""
    }

    STEP_TOOLS = {
        1: step1_tools,
        2: step2_tools,
        3: step3_tools
    }

    messages: List[BaseMessage] = []

    user_initial_input = args.prompt.strip() if args.prompt else input("\nYou: ").strip()
    user_input = user_initial_input
    chat_state = ChatState()

    # User input loop
    if not llm_initialized:
        llm_initialized = True
        llm_execution.on_load()
    
    # LLM steps loop
    try:
        while True:
            if chat_state.current_step != chat_state.new_step:
                # new step is setup
                chat_state.current_step = chat_state.new_step
                chat_state.step_change = True
            
            if not chat_state.final_answer_provided:

                if chat_state.step_change:
                    
                    # add new step message
                    step_change_message_compact = STEP_PROMPTS[chat_state.current_step].format(user_input=user_initial_input)
                    chat_state.step_change = False
                    messages.append(FollowUpMessage(content=step_change_message_compact))

            if chat_state.final_answer_provided:
                # user input
                if chat_state.current_step > 1:
                    user_input = input("\nYou: ").strip()
                    messages.append(UserMessage(content=user_input))

                if user_input.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue

            # LLM invoke
            messages_formatted = messages_to_llm_input(messages, chat_state, STEP_TOOLS, args.input_dir or "")
            print(f"\nAI:")
            response_content = llm_execution.llm_chat(messages_formatted)
            print(f"{response_content}")
            response_content_cleaned = clean_thinking_tag(response_content)

            messages.append(AssistantMessage(content=response_content_cleaned))

            matches = find_tool_invocations(response_content_cleaned)

            tool_found_flag = False
            for match in matches:
                tool_name = match.group(1).strip()
                tool_content = match.group(2).strip()

                tools_for_step = STEP_TOOLS[chat_state.current_step]
                tool_found = next((t for t in tools_for_step if t.name == tool_name), None)

                tool_found_flag = True
                if tool_found:
                    param_pattern = r"<(\w+)>(.*?)</\1>"
                    param_matches = re.findall(param_pattern, tool_content, re.DOTALL)
                    tool_kwargs = {key.strip(): value.strip() for key, value in param_matches}
                    
                    tool_output = tool_found.run(chat_state, **tool_kwargs)
                    print(f"\nAgent:\n{tool_output}")
                    logger.info(tool_output)
                    
                    tool_message = ToolObservationMessage(content=tool_output, tool_name=tool_name)
                    messages.append(tool_message)
                else:
                    error_message = f"Error: Tool {tool_name} not found for current step {chat_state.current_step}."
                    tool_message = ToolObservationMessage(content=error_message, tool_name=tool_name, is_error=True)
                    messages.append(tool_message)

                    logger.error(error_message)


            # Prepare result and follow up as a last message
            
            # tools messages                  
            if not tool_found_flag and not chat_state.final_answer_provided:
                message = FollowUpMessage(f"Error: no tool used! You must use one of the tools available.")
                messages.append(message)

            # unlocking new tools
            if chat_state.get_file_used_count >= 1 and not chat_state.finish_unlocked:
                chat_state.finish_unlocked = True
                step2_tools.append(final_answer_tool)
                log_message = "Unlocked finish tool."
                logger.info(log_message)
                print(f"\nAgent: \n{log_message}")
                unlock_message = "Unlocked new tool!\n" + f"# New tool:\n## {final_answer_tool.name}\nDescription: {final_answer_tool.description}\n" 
                message = UnlockToolMessage(unlock_message)
                messages.append(message)

            # follow up message if step doesn't change
            if chat_state.current_step == chat_state.new_step and not chat_state.final_answer_provided:
                follow_up_content = get_folloup_prompt(chat_state, STEP_TOOLS[chat_state.current_step])
                if follow_up_content:
                    message = FollowUpMessage(follow_up_content)
                    messages.append(message)
                    logger.info(follow_up_content)
                    print(f"\nAgent:{follow_up_content}")
        
    except Exception:
        logger.exception("Error occurred:")


def find_tool_invocations(response_content_cleaned):
    tool_pattern = r"<(\w+)>(.*?)</\1>"
    matches = list(re.finditer(tool_pattern, response_content_cleaned, re.DOTALL))
    return matches

def clean_thinking_tag(response_content):
    return re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception:
        logging.getLogger().exception("An unexpected error occurred:")