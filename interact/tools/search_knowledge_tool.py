from rank_bm25 import BM25Okapi
from knowledge.embeddings import Embeddings
from knowledge.knowledge_store import KnowledgeStore
from knowledge.model import ClassDescription
from typing import Dict, Any, Tuple
from interact.memory.memory import Memory
from interact.chat_state import ChatState
from knowledge.embeddings import EmbeddingEntry

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
        self.embeddings = embeddings
        self.knowledge_store = knowledge_store
        self.documents = list(self.embeddings.get_all_documents().values())
        
        corpus_for_bm25 = []
        for doc in self.documents:
            classname = doc.full_classname
            summary = ""
            if classname:
                class_info = self.knowledge_store.get_class_description(classname)
                if class_info:
                    summary = class_info.summary
                    if class_info.methods:
                        for method in class_info.methods:
                            summary += f"- Method `{method.method_name}`: {method.method_summary}\n"
                    
                    if class_info.variables:
                        for var in class_info.variables:
                            summary += f"- Variable `{var.variable_name}`: {var.variable_summary}\n"
            
            # use lowercase for bm25
            corpus_for_bm25.append(summary.lower())

        self.tokenized_corpus = [text.split() for text in corpus_for_bm25]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def run(self, chat_state: ChatState, query_embeddings: str = "", query_bm25: str = "") -> str:
        print(f"\n---- EmbeddingsContextTool Debug Info ----")
        print(f"Embeddings Search query: '{query_embeddings}'")
        print(f"BM25 Search query: '{query_bm25}'")
        
        try:
            query_embeddings = query_embeddings.strip()
            query_bm25 = query_bm25.strip().lower() # use lowercase for bm25 for case insensitive search
            
            if not query_embeddings or not query_bm25:
                print(f"Empty query provided")
                return "Error: Please provide both 'query_embeddings' and 'query_bm25'."
            
            # Vector search
            vector_results = self.embeddings.search_similar(query_embeddings, limit=5)

            # BM25 search
            tokenized_query = query_bm25.split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to 0.0-1.0 range
            max_bm25_score = max(doc_scores) if any(doc_scores) else 0.0
            
            if max_bm25_score > 0:
                normalized_scores = [score / max_bm25_score for score in doc_scores]
            else:
                normalized_scores = [0.0] * len(doc_scores)
            
            bm25_results = []
            for i, score in enumerate(normalized_scores):
                if score > 0 and i < len(self.documents):
                    item = self.documents[i]
                    bm25_results.append({
                        "item": item,
                        "score": score
                    })
            
            # Sort BM25 results by score
            bm25_results = sorted(bm25_results, key=lambda x: x['score'], reverse=True)[:5]

            # Combine and re-rank results
            combined_results = {}

            # Process vector results
            for result in vector_results:
                item: EmbeddingEntry = result["item"]
                classname = item.full_classname
                if classname not in combined_results:
                    combined_results[classname] = {
                        "item": item,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                    }
                combined_results[classname]["vector_score"] = result["score"]

            # Process BM25 results
            for result in bm25_results:
                item: EmbeddingEntry = result["item"]
                classname = item.full_classname
                if classname not in combined_results:
                    combined_results[classname] = {
                        "item": item,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                    }
                combined_results[classname]["bm25_score"] = result["score"]

            # Calculate total score and create a list of results
            final_results_list = []
            for classname, data in combined_results.items():
                total_score = data["vector_score"] + data["bm25_score"]
                data['total_score'] = total_score
                final_results_list.append(data)

            # Sort by total score
            sorted_results = sorted(final_results_list, key=lambda x: x['total_score'], reverse=True)

            if not sorted_results:
                print(f"No results found for queries.")
                return "No relevant context found for your query."

            classes_added_to_memory = []

            print("--- Combined & Re-ranked Results ---")
            for i, result in enumerate(sorted_results, 1):
                item = result["item"]
                classname = item.full_classname
                
                class_info = self.knowledge_store.get_class_description_extended(classname)
                
                if not class_info:
                    print(f"Could not find class info for '{classname}' in cache.")
                    continue

                vector_score = result.get("vector_score", 0.0)
                bm25_score = result.get("bm25_score", 0.0)
                total_score = result["total_score"]

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