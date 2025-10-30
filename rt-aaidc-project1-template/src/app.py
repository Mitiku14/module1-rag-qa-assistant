import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_documents() -> List[Dict[str, Any]]:
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "..", "data")
    data_path = os.path.abspath(data_path)

    results = []

    if not os.path.exists(data_path):
        print(f"Error: Data folder not found at '{data_path}'. Please create it.")
        return results

    if not os.path.isdir(data_path):
        print(f"Error: '{data_path}' is not a directory.")
        return results

    print(f"Attempting to load documents from: {data_path}")
    found_files = False
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        results.append({"content": text,
                                        "metadata": {"source": filename}})
                        found_files = True
                    else:
                        print(f"Warning: Skipping empty file '{filename}'.")
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")

    if found_files:
        print(f"Successfully loaded {len(results)} text files from '{data_path}'")
    else:
        print(f"No text files found or loaded from '{data_path}'.")

    return results


class RAGAssistant:
    
    def __init__(self):
        """Initialize the RAG assistant."""
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        self.vector_db = VectorDB()

        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful AI assistant. Use ONLY the following context to answer the question.\n"
            "If the question cannot be answered from the context provided, respond with 'I cannot answer based on the provided information.'\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):

        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
        else:
            return None


    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        
        try:
            search_results = self.vector_db.search(input, top_k=n_results)
            
            context_chunks = search_results['documents'][0] if search_results and 'documents' in search_results and search_results['documents'] and search_results['documents'][0] else []
            
            if not context_chunks:
                context = "No relevant context found in vector database."
            else:
                context = "\n\n".join(context_chunks)
            
            response = self.chain.invoke({
                "context": context,
                "question": input
            })
            
            print("\nRetrieved Context:")
            print(context[:400] + "..." if len(context) > 400 else context)
            print("\nModel Response:")
            return response

        except Exception as e:
            print(f"Error in invoke(): {e}")
            return "Sorry, something went wrong while processing your query."


def main():

    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        print("\nLoading documents...")
        sample_docs = load_documents()
        
        if sample_docs:
            assistant.add_documents(sample_docs)
        else:
            print("Warning: No documents loaded. RAG functionality will be limited to LLM's general knowledge unless context is added manually.")


        done = False

        while not done:
            question = input("\nEnter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(result)

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (for OpenAI GPT models)")
        print("- GROQ_API_KEY (for Groq Llama models)")
        print("- GOOGLE_API_KEY (for Google Gemini models)")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()