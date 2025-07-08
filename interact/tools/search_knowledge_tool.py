from knowledge.embeddings import Embeddings
from knowledge.knowledge_store import KnowledgeStore
from interact.chat_state import ChatState
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
    
    def __init__(self, embeddings: Embeddings, knowledge_store: KnowledgeStore):
        self.knowledge_store = knowledge_store
        self.knowledge_search = KnowledgeSearch(embeddings, knowledge_store)

    def run(self, chat_state: ChatState, query_embeddings: str = "", query_bm25: str = "") -> str:
        print(f"\n---- EmbeddingsContextTool Debug Info ----")
        print(f"Embeddings Search query: '{query_embeddings}'")
        print(f"BM25 Search query: '{query_bm25}'")
        
        try:
            query_embeddings = query_embeddings.strip()
            query_bm25 = query_bm25.strip()
            
            if not query_embeddings or not query_bm25:
                print(f"Empty query provided")
                return "Error: Please provide both 'query_embeddings' and 'query_bm25'."
            
            sorted_results = self.knowledge_search.hybrid_search(query_embeddings, query_bm25)

            if not sorted_results:
                print(f"No results found for queries.")
                return "No relevant context found for your query."

            classes_added_to_memory = []

            print("--- Combined & Re-ranked Results ---")
            for i, result in enumerate(sorted_results, 1):
                item = result.item
                classname = item.full_classname
                
                class_info = self.knowledge_store.get_class_description_extended(classname)
                class_structure = self.knowledge_store.get_class_structure(classname)
                
                if not class_info:
                    print(f"Could not find class info for '{classname}' in cache.")
                    continue

                vector_score = result.vector_score
                bm25_score = result.bm25_score
                total_score = result.total_score

                score_details = ""
                if vector_score > 0 and bm25_score > 0:
                    score_details = f"Score: {total_score:.2f}, Vec: {vector_score:.2f} + BM25: {bm25_score:.2f}"
                elif vector_score > 0:
                    score_details = f"Score: {total_score:.2f}, Vec only"
                elif bm25_score > 0:
                    score_details = f"Score: {total_score:.2f}, BM25 only"
                else:
                    score_details = f"Score: {total_score:.2f}"

                classes_added_to_memory.append(f"- {classname} ({score_details})")
                
                chat_state.memory.add_class(classname, class_info)

            chat_state.search_used_count += 1
            observation = f"Observation from {self.name}:\nClasses added to the memory:\n" + "\n".join(classes_added_to_memory)
            
            print(f"Found {len(classes_added_to_memory)} results")
            print(observation)
            print(f"---- End Debug Info ----\n")

            return observation
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"---- End Debug Info ----\n")
            return f"Error searching context: {str(e)}"