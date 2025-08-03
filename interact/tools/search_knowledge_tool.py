import os
import logging
from knowledge.knowledge_store import KnowledgeStore
from interact.chat_state import ChatState
from core.config.config import Config
from core.search.search import KnowledgeSearch

class SearchKnowledgeTool:
    name: str = "search_knowledge_tool"
    description: str = ("Search for relevant knowledge about the project's code database based on your query.\n"
                        "Parameters:\n"
                        "- query_embeddings (required): The search query to find relevant context for vector search.\n"
                        "- query_bm25 (required): The search query for BM25 keyword search.\n"
                        "Example: \n"
                        "```\n"
                        "<search_knowledge_tool>\n"
                        "<query_embeddings>How frequently is data updated?</query_embeddings>\n"
                        "<query_bm25>data update frequency interval</query_bm25>\n"
                        "</search_knowledge_tool>\n"
                        "```")
    
    def __init__(self, input_dir: str, knowledge_store: KnowledgeStore, knowledge_search: KnowledgeSearch, logger: logging.Logger, config: Config):
        self.input_dir = input_dir
        self.logger = logger
        self.knowledge_store = knowledge_store
        self.knowledge_search = knowledge_search

    def run(self, chat_state: ChatState, query_embeddings: str = "", query_bm25: str = "", original_query: str = "", **kwargs) -> str:
        self.logger.debug(f"Tool invoked:  ({self.name}), embeddings Search query: '{query_embeddings}', BM25 Search query: '{query_bm25}', original query:\n'{original_query}'\n")
        
        query_embeddings = query_embeddings.strip()
        query_bm25 = query_bm25.strip()
        
        if not query_bm25:
            self.logger.error(f"Empty query_bm25 provided")
            return "Error: Please provide both 'query_embeddings' and 'query_bm25'."
        
        if not query_embeddings:
            self.logger.error(f"Empty query_embeddings provided")
            return "Error: Please provide both 'query_embeddings' and 'query_bm25'."
        
        sorted_results = self.knowledge_search.hybrid_search([query_embeddings], [query_bm25], limit=20)

        sorted_results = self.knowledge_search.rerank_results(sorted_results, original_query, rerank_limit = 20)

        if not sorted_results:
            print(f"No results found for queries.")
            return "No relevant context found for your query."

        classes_added_to_memory = []

        for i, result in enumerate(sorted_results, 1):
            classname = result.class_description.full_classname
            
            class_info = result.class_description
            class_structure = self.knowledge_store.get_class_structure(classname)
            
            if not class_info:
                self.logger.warning(f"Could not find class info for '{classname}' in cache.")
                continue

            vector_score = result.vector_score
            bm25_score = result.bm25_score
            rerank_score = result.rerank_score
            total_score = result.total_score

            score_details = ""
            if rerank_score > 0:
                score_details = f"Score: {rerank_score:.2f} (reranked)"
            elif vector_score > 0 and bm25_score > 0:
                score_details = f"Score: {total_score:.2f}, Vec: {vector_score:.2f} + BM25: {bm25_score:.2f}"
            elif vector_score > 0:
                score_details = f"Score: {total_score:.2f}, Vec only"
            elif bm25_score > 0:
                score_details = f"Score: {total_score:.2f}, BM25 only"
            else:
                score_details = f"Score: {total_score:.2f}"

            classes_added_to_memory.append(f"- {classname} ({score_details})")

            abs_file_path = os.path.join(self.input_dir, result.file)
            file_size = os.path.getsize(abs_file_path)
            
            chat_state.memory.add_search_result(result, result.file, file_size)

        chat_state.search_used_count += 1
        observation = f"Observation from {self.name}:\nClasses found and added to the memory:\n" + "\n".join(classes_added_to_memory)
        
        self.logger.debug(f"Tool result ({self.name}):\n{observation}")

        return observation
            