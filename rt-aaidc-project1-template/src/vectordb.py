import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        script_dir = os.path.dirname(__file__)
        chroma_db_path = os.path.join(script_dir, "chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        if not text.strip():
            print("Empty text provided for chunking")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",   # Double newline (paragraph break)
                "\n",     # Single newline (line break)
                " ",      # Space
                ".",      # Period (sentence end)
                ",",      # Comma
                ";",      # Semicolon
                "!",      # Exclamation mark
                "?",      # Question mark
                ""        # Fallback character
            ],
            length_function=len,
            is_separator_regex=False, 
        )
        chunks = text_splitter.split_text(text)
        print(f"✅ Split text into {len(chunks)} chunks")

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            print("No documents provided to add")
            return

        print(f"Processing {len(documents)} documents...")
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content.strip():
                print(f"Skipping empty document at index {doc_idx}")
                continue

            chunks = self.chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

                if len(all_chunks) >= 1000: 
                    self._add_to_vector_db_batch(all_chunks, all_metadatas, all_ids)
                    all_chunks, all_metadatas, all_ids = [], [], []
        if all_chunks:
            self._add_to_vector_db_batch(all_chunks, all_metadatas, all_ids)
        print("All documents added to the vector database.")

    def _add_to_vector_db_batch(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Helper to add a batch of chunks to the vector database."""
        if not chunks:
            return
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist() 
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        print(f"Added {len(chunks)} chunks to the vector database in a batch.")


    def search(self, query: str, top_k: int = 5) -> Dict[str, List[Any]]:
        
        print(f"Searching for: {query}")
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist() # Added show_progress_bar
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        if not results.get("documents") or not results["documents"][0]:
            print("No relevant documents found.")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
        else:
            print(f"Found {len(results['documents'][0])} relevant documents.")
        return results