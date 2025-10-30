
# Module 1 - RAG Q&A Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** based **Question & Answer Assistant** that retrieves relevant information from documents and generates accurate, context-aware responses using large language models (LLMs).  

---

## Overview

This project demonstrates the integration of **vector databases**, **embeddings**, and **language models** to build a context-aware assistant capable of answering user queries from custom document sources.

It was developed as part of **Module 1 Course Project**, focusing on **AI-driven information retrieval** and **intelligent response generation**.

---

## Features

- 🗂️ Loads and embeds documents for efficient retrieval  
- 🧠 Uses RAG (Retrieval-Augmented Generation) to combine search + generation  
- 💬 Supports OpenAI, Groq (LLaMA), and Google Gemini models  
- 🧾 Provides accurate, context-based answers  
- ⚡ Interactive CLI interface  
- 🧰 Modular, extensible, and beginner-friendly design  

---

## Tech Stack

| Category | Technology |
|-----------|-------------|
| **Programming Language** | Python 3.8+ |
| **AI Framework** | LangChain |
| **Embeddings Model** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | Chroma / FAISS (via `langchain_community.vectorstores`) |
| **LLMs** | Groq LLaMA 3.1, OpenAI GPT, or Google Gemini |
| **Environment Management** | dotenv |
| **Development** | Visual Studio Code / Terminal |

---

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mitiku14/module1-rag-qa-assistant.git
cd module1-rag-qa-assistant
2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
🔑 Environment Setup

Create a .env file in the project root with your API keys:
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_api_key_here
📚 Adding Documents

Create a data/ folder in the root directory and place your text or PDF files there.
📁 src/
 ├── app.py
 ├── vectordb.py
 ├── rag_assistant.py
 ├── data/
 │    ├── ai_intro.txt
 │    └── quantum_computing.pdf
▶️ Running the Application

Run the assistant from your terminal:
python app.py
🧠 How It Works

Document Loading → Reads text/PDF files from the data/ folder.

Embedding Generation → Converts document chunks into vector embeddings.

Vector Storage → Stores embeddings in a local vector database (Chroma).

Retrieval → When a query is entered, the assistant searches for relevant vectors.

Augmented Generation → The model generates a response using retrieved context.
🏷️ Tags

AI RAG LLM LangChain Python Vector Database Q&A Assistant Machine Learning
Acknowledgements

LangChain

Sentence Transformers

ChromaDB

Groq

OpenAI

Google Generative AI