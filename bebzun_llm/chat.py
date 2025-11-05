import argparse
import os
import re
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from bebzun_llm.core.config.config import load_config
from bebzun_llm.core.utils.file_manager import FileManager
from bebzun_llm.core.llm.llm_execution_openai import OpenAILlmExecution
from bebzun_llm.core.llm.llm_execution_anthropic import AnthropicLlmExecution
from bebzun_llm.core.llm.llm_execution_ollama import OllamaLlmExecution
from bebzun_llm.core.llm.llm_execution_mlx import MlxLlmExecution
from bebzun_llm.knowledge.embeddings_store import Embeddings
from bebzun_llm.knowledge.knowledge_store import KnowledgeStore
from bebzun_llm.core.search.search import KnowledgeSearch, SearchResult
from bebzun_llm.knowledge.model import ClassDescription
from bebzun_llm.core.llm.prepare_prompts import params_get_project_context
from bebzun_llm.interact.memory.memory import Memory
from bebzun_llm.interact.tools.list_files_tool import ListFilesTool
from bebzun_llm.interact.tools.get_file_content_tool import GetFileContentTool
from bebzun_llm.interact.tools.search_knowledge_tool import SearchKnowledgeTool
from bebzun_llm.interact.tools.finish_step_tool import FinishStepTool
from bebzun_llm.interact.tools.final_answer_tool import FinalAnswerTool
from bebzun_llm.interact.chat_state import ChatState
from bebzun_llm.interact.chat.messages.models import AssistantMessage, BaseMessage, ToolObservationMessage, UserMessage, MemoryMessage
from bebzun_llm.core.utils.logging_utils import setup_logging, setup_llm_logger
from bebzun_llm.core.context.build_context import BuildContext

def main():
    parser = argparse.ArgumentParser(
        description='Chat with LLM model with file listing and embeddings search capabilities.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start chat in current directory
  %(prog)s

  # Start chat in specific directory
  %(prog)s -i /path/to/project

  # Single-shot question with output to file
  %(prog)s -ss "Explain the architecture" -o output.txt

  # Start with initial prompt
  %(prog)s -p "Help me understand the authentication flow"
        """
    )
    parser.add_argument('-i', '--input-dir',
                        help='Directory to list files from. Defaults to current directory.')
    parser.add_argument('-p', '--prompt',
                        help='User task to generate context.')
    parser.add_argument('-ss', '--single-shot',
                        help='User prompt for single shot answer.')
    parser.add_argument('-o', '--output',
                        help='Output file for single-shot answer')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    parser.add_argument('--log-file', help='Path to the log file')
    parser.add_argument('--llm-log-file', help='Path to the LLM log file')

    args = parser.parse_args()
    
    global logger
    global llm_logger
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    llm_logger = setup_llm_logger(log_level=args.log_level, log_file=args.llm_log_file)

    # Determine input directory - default to current directory if not specified
    input_dir = os.path.abspath(args.input_dir if args.input_dir else os.getcwd())
    
    # Validate directory exists
    if not os.path.exists(input_dir):
        print(f"\nError: Directory '{input_dir}' does not exist.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"\nError: '{input_dir}' is not a directory.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Check for config file
    config_file_path = os.path.join(input_dir, ".ai-agent", "config.json")
    
    if not os.path.exists(config_file_path):
        print(f"\nError: Configuration file not found at '{config_file_path}'.", file=sys.stderr)
        print(f"Please ensure you are running this command from a project directory", file=sys.stderr)
        print(f"or use the -i parameter to specify the project directory.\n", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_file_path)

    # Initialize LLM execution
    if config.llm.mode == 'mlx':
        print(f"Loading MLX model: {config.llm.mlx.model}")
        llm_execution = MlxLlmExecution(model=config.llm.mlx.model, temperature=config.llm.mlx.temperature, logger=llm_logger)
    elif config.llm.mode == 'ollama':
        print(f"Loading Ollama model: {config.llm.ollama.model}")
        llm_execution = OllamaLlmExecution(model=config.llm.ollama.model, temperature=config.llm.ollama.temperature, url=config.llm.ollama.url, logger=llm_logger, max_context=config.llm.max_context)
    elif config.llm.mode == 'anthropic':
        print(f"Connection to Anthropic model: {config.llm.anthropic.model}")
        llm_execution = AnthropicLlmExecution(model=config.llm.anthropic.model, key=config.llm.anthropic.key, logger=llm_logger)
    elif config.llm.mode == 'openai':
        logger.info(f"Initializing connection to {config.llm.openai.url}...")
        llm_execution = OpenAILlmExecution(
            model=config.llm.openai.model,
            temperature=config.llm.openai.temperature,
            key=config.llm.openai.key,
            base_url=config.llm.openai.url,
            logger=llm_logger
        )
    else:
        raise ValueError(f"Unsupported LLM mode: {config.llm.mode}")
    
    llm_execution.on_load()

    file_manager = FileManager()
    file_manager.load(logger=logger, base_dir=input_dir, config=config, name_filter=None)
    
    embeddings_instance = None
    try:
        embeddings_instance = Embeddings(config, logger)
        if not embeddings_instance.initialize(input_dir):
            logger.error("Embeddings storage is not available. Context search will be disabled.")
            embeddings_instance = None
    except Exception:
        logger.exception("Error initializing embeddings:")
        sys.exit(1)

    knowledge = KnowledgeStore()
    knowledge_file_path = os.path.join(input_dir, ".ai-agent", f"db_final.json")
    structure_file_path = os.path.join(input_dir, ".ai-agent", f"db_preprocess.json")
    if os.path.exists(knowledge_file_path):
        knowledge.read_storage_final(knowledge_file_path, input_dir)
        logger.info(f"Successfully loaded knowledge db from {knowledge_file_path}")
    else:
        logger.error(f"Warning: knowledge db file not found at {knowledge_file_path}. Context might be incomplete.")
        sys.exit(1)

    if os.path.exists(structure_file_path):
        knowledge.read_storage_pre_process(structure_file_path)
        logger.info(f"Successfully loaded preprocess db from {structure_file_path}")
    else:
        logger.error(f"Warning: preprocess db file not found at {structure_file_path}. Context might be incomplete.")
        sys.exit(1)

    knowledge_search = KnowledgeSearch(embeddings_instance, knowledge, config, logger)

    chat_state = ChatState(knowledge)

    context_builder = BuildContext(
        input_dir=input_dir,
        config=config,
        llm_execution=llm_execution,
        knowledge_store=knowledge,
        knowledge_search=knowledge_search,
        memory=chat_state.memory,
        logger=logger
    )
    
    step3_tools = []

    if embeddings_instance and embeddings_instance.is_loaded():
        search_knowledge_tool = SearchKnowledgeTool(input_dir, knowledge_store=knowledge, knowledge_search=knowledge_search, logger=logger, config=config)
        step3_tools.append(search_knowledge_tool)

    if config.source_dirs:
        list_files_tool = ListFilesTool(base_path=input_dir, source_dirs=config.source_dirs, logger=logger)
        get_file_content_tool = GetFileContentTool(base_path=input_dir, source_dirs=config.source_dirs, logger=logger, file_manager=file_manager)
        step3_tools.append(list_files_tool)
        step3_tools.append(get_file_content_tool)
    
    llm_initialized = False
    
    logger.info(f"Chat initialized with {llm_execution.model_desc()} model.")
    if args.input_dir:
        logger.info(f"File tools enabled for directory: {args.input_dir}")
    if embeddings_instance and embeddings_instance.is_loaded():
        logger.info("Embeddings context search enabled.")
    logger.info("Type 'exit' to end the conversation.")

    messages: List[BaseMessage] = []

    user_initial_input = args.prompt.strip() if args.prompt else input("\nYou: ").strip()
    user_single_shot_question = get_followup_prompt(args.single_shot) if args.single_shot else None
    file_output = Path(args.output) if args.output else None

    user_input = user_initial_input

    messages.append(UserMessage(content=f"# User task:\n<user_task>\n{user_input}\n</user_task>"))

    context_builder.build(user_input)

    # User input loop
    if not llm_initialized:
        llm_initialized = True
        llm_execution.on_load()
    
    tool_observation_flag = False
    response_content_cleaned = ""

    # LLM steps loop
    try:
        while True:
            skip_ai = False

            # user input
            if tool_observation_flag == False or tool_used_counter > 3:
                if user_single_shot_question:
                    user_input = user_single_shot_question
                    user_single_shot_question = None
                else:
                    user_input = input("\nYou: ").strip()

                if user_input.lower() == 'exit':
                    print("\nGoodbye!")
                    exit()

                if user_input.startswith('\\'):
                    user_input = user_input.replace('\\','/')
                
                if user_input.startswith('/'):
                    skip_ai = True
                
                if user_input.lower().startswith('/add '):
                    query = user_input[5:]
                    if query:
                        add_class_memory(chat_state, query, input_dir, knowledge)
                    skip_ai = True

                if user_input.lower().startswith('/file '):
                    query = user_input[6:]
                    if query:
                        add_file_memory(chat_state, query, input_dir, file_manager)
                    skip_ai = True

                if user_input.lower().startswith('/remove '):
                    query = user_input[9:]
                    if query:
                        remove_class_memory(chat_state, query, input_dir, knowledge)
                    skip_ai = True

                if user_input.lower().startswith('/del '):
                    count = int(user_input[5:])
                    if count > 0:
                        del messages[-count:]
                        print(messages)
                    skip_ai = True

                if user_input.lower().startswith('/list'):
                    print("\n".join(f"- {type(message).__name__}:" + message.content[:60].replace('\n', '') + "..." for message in messages))
                    skip_ai = True

                if user_input.lower().startswith('/memory'):
                    print(f"Memory:\n{chat_state.memory.get_formatted_memory_compact()}")
                    skip_ai = True

                if user_input.lower().startswith('/memory-full'):
                    print(f"Memory:\n{chat_state.memory.get_formatted_memory()}")
                    skip_ai = True

                if user_input.lower().startswith('/save '):
                    file = user_input[6:]
                    if file:
                        file_path = Path(file)
                        save_last_message(chat_state, file_path, response_content_cleaned)
                    skip_ai = True
                
                if not user_input:
                    skip_ai = True

                if not skip_ai:
                    messages.append(UserMessage(content=user_input))
                    tool_used_counter = 0

            tool_observation_flag = False

            if skip_ai:
                continue

            # LLM invoke
            messages_formatted = messages_to_llm_input(messages, chat_state, step3_tools, input_dir)
            print(f"\nAI:")
            response_content = llm_execution.llm_chat(messages_formatted, verbose=True)
            
            response_content_cleaned = clean_thinking_tag(response_content)

            messages.append(AssistantMessage(content=response_content_cleaned))

            matches = find_tool_invocations(response_content_cleaned)

            if len(matches) > 0:
                tool_used_counter += 1
                tool_observation_flag = True

            for match in matches:
                tool_name = match.group(1).strip()
                tool_content = match.group(2).strip()

                tools_for_step = step3_tools
                tool_found = next((t for t in tools_for_step if t.name == tool_name), None)

                if tool_found:
                    param_pattern = r"<(\w+)>(.*?)</\1>"
                    param_matches = re.findall(param_pattern, tool_content, re.DOTALL)
                    tool_kwargs = {key.strip(): value.strip() for key, value in param_matches}
                    tool_kwargs['original_query'] = user_initial_input
                    
                    tool_output = tool_found.run(chat_state, **tool_kwargs)
                    print(f"\nAgent:\n{tool_output}")
                    logger.debug(tool_output)
                    
                    tool_message = ToolObservationMessage(content=tool_output, tool_name=tool_name)
                    messages.append(tool_message)
                else:
                    error_message = f"Error: Tool {tool_name} not found."
                    tool_message = ToolObservationMessage(content=error_message, tool_name=tool_name, is_error=True)
                    messages.append(tool_message)

                    logger.error(error_message)

            if file_output is not None and (tool_observation_flag == False or tool_used_counter > 3):
                # use response_content_cleaned and save to file_output
                save_last_message(chat_state, file_output, response_content_cleaned)

                sys.exit(0)

    except Exception:
        logger.exception("Error occurred:")

def save_last_message(chat_state: ChatState, file_output: Path, response_content_cleaned):
    file_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_output, 'w', encoding='utf-8') as f:
            memory = chat_state.memory.get_formatted_memory()
            memory_compact = chat_state.memory.get_formatted_memory_compact()
            f.write(f"{response_content_cleaned}\n\n*** Response generated with this context: ***\n{memory_compact}\n\n\n*** Response generated with this context: ***\n{memory}")
    except Exception:
        sys.exit(1)


def get_system_prompt(tools: List[Any], input_dir: str) -> str:
    tools_description = "\n".join([f"## {t.name}\nDescription: {t.description}\n" for t in tools])
    
    prompt_path = os.path.join("interact", "chat", "system_prompt.txt")
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
        
    project_context_val = params_get_project_context(input_dir)

    return prompt_template.format(
        project_context=project_context_val,
        tools_description=tools_description
    )


def get_followup_prompt(arg: str) -> str:
    is_file = False
    try: 
        is_file = os.path.exists(Path(arg))
    except:
        is_file = False

    if is_file:
        prompt_path = os.path.join("interact", "chat", "follow_up_tickets.txt")
        with open(prompt_path, "r") as f:
            prompt = f.read()
        return prompt
    
    return arg

def messages_to_llm_input(messages: List[BaseMessage], chat_state: ChatState, tools, input_dir: str) -> List[Dict[str, Any]]:
    """Concatenates a list of agent message objects to a short list with a single memory state."""
    output = []
    system_prompt = get_system_prompt(tools, input_dir)
    output.append({"role": "system", "content": system_prompt})

    memory_output = f"\n{chat_state.memory.get_formatted_memory()}\n"
    
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
        processed_messages = messages.copy()
        # Find last user message to insert memory before it
        last_user_idx = -1
        for i, msg in reversed(list(enumerate(processed_messages))):
            if isinstance(msg, UserMessage):
                last_user_idx = i
                break
        if last_user_idx != -1:
            processed_messages.insert(last_user_idx, MemoryMessage(content=memory_output))
        else:
            processed_messages.append(MemoryMessage(content=memory_output))

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

def add_class_memory(chat_state: ChatState, query: str, input_dir: str, knowledge: KnowledgeStore):
    class_info = knowledge.find_class_description_extended(query)
    if class_info:
        print(f"Added: `{class_info.class_summary.full_classname}`")
        abs_file_path = os.path.join(input_dir, class_info.file)
        file_size = os.path.getsize(abs_file_path)

        result = SearchResult(
                    full_classname = class_info.class_summary.full_classname,
                    file = class_info.file,
                    details = [],
                    class_description = class_info.class_summary
                )
        
        chat_state.memory.add_search_result(result, class_info.file, file_size)

def add_file_memory(chat_state: ChatState, query: str, input_dir: str, file_manager: FileManager):
    file_infos = [name for name in file_manager.get_all_supported_files(only_filtered=False) if query in name]

    for file_info in file_infos:
        if file_info:

            # raw_input returns the empty string for "enter"
            yes = {'yes','y', 'ye', ''}
            no = {'no','n'}

            choice = input(f"Add: `{file_info}?` [Y/n]").lower()
            if choice in yes:
            
                abs_file_path = os.path.join(input_dir, file_info)
                file_size = os.path.getsize(abs_file_path)

                with open(abs_file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                
                chat_state.memory.add_file(abs_file_path, content, input_dir)
                print(f"Added file to memory: {file_info}")

def remove_class_memory(chat_state: ChatState, query: str, input_dir: str, knowledge: KnowledgeStore):
    result = chat_state.memory.remove_class_memory(query)
    for i in result:
        print(f"Removed: `{i}`")
    

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