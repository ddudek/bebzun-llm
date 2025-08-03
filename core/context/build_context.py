import logging
import os
import re
from typing import Union, List
from pathlib import Path

from knowledge.model import ClassDescriptionExtended
from core.config.config import Config
from core.llm.llm_execution_anthropic import AnthropicLlmExecution
from core.llm.llm_execution_mlx import MlxLlmExecution
from core.llm.llm_execution_ollama import OllamaLlmExecution
from core.llm.llm_execution_openai import OpenAILlmExecution
from core.context.build_context_result import BuildContextQueriesLLMResult, BuildContextGetFilesLLMResult, BuildContextStep3
from core.search.search import KnowledgeSearch
from core.search.search_result import SearchResult
from interact.memory.memory import Memory, MemoryItemClassSummary
from knowledge.knowledge_store import KnowledgeStore
from core.search.embedding_entry import EmbeddingEntry

LlmExecution = Union[AnthropicLlmExecution, MlxLlmExecution, OllamaLlmExecution, OpenAILlmExecution]

class BuildContext:
    """
    Builds context for a given user task by interacting with the LLM and knowledge base.
    """
    def __init__(self, 
                 input_dir: str,
                 config: Config, 
                 llm_execution: LlmExecution, 
                 knowledge_store: KnowledgeStore, 
                 knowledge_search: KnowledgeSearch,
                 memory: Memory, 
                 logger: logging.Logger):
        """
        Initializes the BuildContext class.

        Args:
            config: The application configuration.
            llm_execution: The LLM execution instance.
            knowledge_store: The knowledge store instance.
            knowledge_search: The knowledge search instance.
            memory: The interaction memory instance.
            logger: The logger instance.
        """
        self.input_dir = input_dir
        self.config = config
        self.llm_execution = llm_execution
        self.knowledge_store = knowledge_store
        self.knowledge_search = knowledge_search
        self.memory = memory
        self.logger = logger

    def build(self, user_task: str) -> str:
        """
        Builds the context for the given user task.

        Args:
            user_task: The user's task description.
        """
        self.logger.info(f"Building context for user task:\n\n'{user_task}'\n")

        project_context = ""
        if self.input_dir:
            project_context_path = Path(self.input_dir) / ".ai-agent" / "project_context.txt"
            if project_context_path.exists():
                with open(project_context_path, "r", encoding="utf-8") as f:
                    project_context = f.read()
            else:
                self.logger.error(f"Project context file not found at: {project_context_path}")

        system_prompt = ""
        system_prompt_path = Path(__file__).parent / "prompts" / "build_context_prompt_system.txt"
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        # Step 1: Generate search queries from user task
        step1_result = self._step1_generate_search_queries(user_task, project_context, system_prompt)
        self._execute_searches(step1_result.similarity_search_queries, step1_result.bm25_search_queries, step1_result.user_task_refined)

        # Step 2: Use search results, review and get files with real implementation
        self._step2_get_and_review_files(user_task, step1_result.user_task_refined, project_context, system_prompt)

        # Step 3: Review files and remove unwanted memory items
        result = self._step3_summarize_and_final(user_task, step1_result.user_task_refined, project_context, system_prompt)
        
        print("--- Build context finished. ---")
        return result

    def _step1_generate_search_queries(self, user_task: str, project_context: str, system_prompt: str) -> BuildContextQueriesLLMResult:
        """
        Step 1: Generate search queries using the LLM.
        """
        self.logger.info("Step 1: Generating search queries...")
        
        prompt_template_path = Path(__file__).parent / "prompts" / "build_context_prompt_step1.txt"
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(project_context=project_context, user_task=user_task)

        schema = BuildContextQueriesLLMResult.model_json_schema()

        self.logger.debug(f"LLM prompt:\n\n{system_prompt}\n\n{prompt}")

        try:
            llm_response = self.llm_execution.llm_invoke(
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema
            )
            result = BuildContextQueriesLLMResult.model_validate(llm_response)
            self.logger.info(f"Generated search queries:\n {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error during LLM invocation in Step 1: {e}", exc_info=True)
            raise

    def _execute_searches(self, similarity_search_queries: List[str], bm25_search_queries: List[str], user_task: str):
        """
        Step 1: Execute the generated search queries and rerank the results.
        """
        self.logger.info("Executing searches...")
        
        # Execute hybrid search with the generated queries
        sorted_results = self.knowledge_search.hybrid_search(
            query_embeddings=similarity_search_queries,
            query_bm25=bm25_search_queries,
            limit=30
        )

        # Rerank the results based on the original user task
        reranked_results = self.knowledge_search.rerank_results(
            sorted_results,
            user_task,
            rerank_limit=50
        )

        search_results = reranked_results

        for result in search_results:
            classname = result.class_description.full_classname
            
            class_info = self.knowledge_store.get_class_description_extended(classname)

            result.class_description = class_info.class_summary

            abs_file_path = os.path.join(self.input_dir, class_info.file)
            file_size = os.path.getsize(abs_file_path)

            self.memory.add_search_result(result, class_info.file, file_size)

        if not reranked_results:
            self.logger.info("No results found after reranking.")
            return

        self.logger.info(f"Found {len(reranked_results)} relevant results after reranking.")

    def _step2_get_and_review_files(self, user_task: str, user_task_refined: str, project_context: str, system_prompt: str):
        self.logger.info("Step 2: Reviewing search results and getting files...")

        prompt_template_path = Path(__file__).parent / "prompts" / "build_context_prompt_step2.txt"
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        formatted_search_results = self.memory.get_formatted_memory()

        prompt = prompt_template.format(user_task=user_task, search_results=formatted_search_results, project_context=project_context)
        schema = BuildContextGetFilesLLMResult.model_json_schema()

        self.logger.debug(f"LLM prompt:\n\n{system_prompt}\n\n{prompt}")

        try:
            llm_response = self.llm_execution.llm_invoke(
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema
            )
            get_files_result = BuildContextGetFilesLLMResult.model_validate(llm_response)
            self.logger.info(f"LLM decided to open files: {get_files_result.files_to_open}")

            for file_path in get_files_result.files_to_open:
                self._add_file_to_memory(file_path)

            similarity_search_queries = get_files_result.additional_search_queries
            bm25_search_queries = []
            if similarity_search_queries:
                self._execute_searches(similarity_search_queries, bm25_search_queries, user_task_refined)

            for file_path in get_files_result.files_to_open:
                self.logger.info(f"Adding classes related to `{file_path}`")
                to_add: List[ClassDescriptionExtended] = []
                to_add.extend([cls.parent_description for cls in self.knowledge_store.get_file_usages(file_path).values()])
                # to_add.extend([cls.dependency_description for cls in self.knowledge_store.get_file_dependencies(file_path).values()])

                for class_info in to_add:
                    if class_info and class_info.class_summary:
                        classname = class_info.class_summary.full_classname
                        self.logger.info(f"Adding class summary: {classname}")

                        abs_file_path = os.path.join(self.input_dir, class_info.file)
                        file_size = os.path.getsize(abs_file_path)
                    
                        search_result = SearchResult(
                            full_classname=classname,
                            file=class_info.file,
                            details=[],
                            class_description=class_info.class_summary,
                        )

                        self.memory.add_search_result(search_result, class_info.file, file_size)

            
            for cls_name in get_files_result.classes_not_related:
                self.memory.remove_class_memory(cls_name)

        except Exception as e:
            self.logger.error(f"Error during LLM invocation in Step 2: {e}", exc_info=True)
            raise

    def _step3_summarize_and_final(self, user_task: str, user_task_refined: str, project_context: str, system_prompt: str) -> str:
        self.logger.info("Step 3: Finalize...")

        prompt_template_path = Path(__file__).parent / "prompts" / "build_context_prompt_step3.txt"
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        formatted_search_results = self.memory.get_formatted_memory()

        prompt = prompt_template.format(user_task=user_task, search_results=formatted_search_results, project_context=project_context)
        schema = BuildContextStep3.model_json_schema()

        self.logger.debug(f"LLM prompt:\n\n{system_prompt}\n\n{prompt}")

        try:
            llm_response = self.llm_execution.llm_invoke(
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema
            )
            llm_result = BuildContextStep3.model_validate(llm_response)

            if llm_result.final_answer and (llm_result.additional_search_queries or llm_result.files_to_open):
                self.logger.warning("Error: Both final answer and more information is provided")

            for file_path in llm_result.files_to_open:
                self._add_file_to_memory(file_path)

            similarity_search_queries = llm_result.additional_search_queries
            bm25_search_queries = []
            if similarity_search_queries:
                self._execute_searches(similarity_search_queries, bm25_search_queries, user_task_refined)

            for cls_name in llm_result.classes_not_related:
                print(f"Removing: `{cls_name}`")
                self.memory.remove_class_memory(cls_name)
                cls_desc = self.knowledge_store.find_class_description_extended(cls_name)
                if cls_desc and cls_desc.file:
                    self.memory.remove_file_memory(cls_desc.file)

            if llm_result.final_answer:
                print("--- LLM Final Answer: ---")
                return llm_result.final_answer

        except Exception as e:
            self.logger.error(f"Error during LLM invocation in Step 2: {e}", exc_info=True)
            raise
            
    def _add_file_to_memory(self, path: str):
        """
        Reads a file and adds its content to the memory.
        """
        source_dirs = [os.path.normpath(d) for d in self.config.source_dirs]

        if not source_dirs:
            self.logger.error("No source directories configured. Cannot read file.")
            return

        path = self._clean_path_input(path)
        
        try:
            if not path:
                self.logger.error(f"Empty path provided")
                return

            full_path = os.path.normpath(os.path.join(self.input_dir, path))
            
            # Security check: ensure the requested path is within one of the allowed source directories
            abs_path = os.path.abspath(full_path)
            is_allowed = False
            for src_dir in source_dirs:
                abs_src_dir = os.path.abspath(os.path.join(self.input_dir, src_dir))
                if abs_path.startswith(abs_src_dir):
                    is_allowed = True
                    break
            
            if not is_allowed:
                self.logger.error(f"Safety check failed: Path is not within allowed source directories: {path}")
                return
            
            if not os.path.exists(full_path):
                self.logger.error(f"Path does not exist: '{full_path}'")
                return
                
            if not os.path.isfile(full_path):
                self.logger.error(f"Path is not a file: '{full_path}'")
                return
                
            with open(full_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                
            self.memory.add_file(path, content, self.input_dir)
            self.logger.info(f"Added file to memory: {path}")
            
        except Exception as e:
            self.logger.error(f"Error reading file '{path}': {e}", exc_info=True)

    def _clean_path_input(self, path: str) -> str:
        path = re.sub(r'[\n`]+$', '', path)
        path = re.sub(r'\n```$', '', path)
        path = path.strip()
        return path