# 🧠 AI Workplace Burnout Assistant

> A Retrieval-Augmented Generation (RAG) chatbot that provides real-time, grounded workplace wellness support using large language models and semantic search.

---

## 📌 Overview

Workplace burnout is one of the most pressing challenges in modern organizations. This project tackles that problem with a production-style AI assistant that retrieves relevant information from curated psychological and HR wellness documents, then uses a large language model to generate empathetic, accurate, and grounded responses.

Unlike a standard chatbot that relies purely on an LLM's pre-trained knowledge, this system is **grounded in real documents** — meaning every response is backed by retrieved context, reducing hallucinations and improving reliability.

---

## 🎯 Key Features

- **RAG Pipeline** — Combines semantic search with LLM generation for accurate, document-grounded responses
- **Semantic Search** — FAISS vector database enables fast similarity search across embedded document chunks
- **Multi-Source Ingestion** — Ingests both PDF documents and live web pages as knowledge sources
- **Prompt Engineering** — Carefully designed system prompt ensures safe, empathetic, and contextually appropriate outputs
- **Interactive Chat UI** — Streamlit-powered chatbot interface with persistent chat history and sample questions
- **Model Agnostic** — Uses OpenRouter, allowing easy swapping between GPT-4, Claude, Mistral, and others

---

## 🏗️ System Architecture

```
User Query (Streamlit UI)
        │
        ▼
  FAISS Retriever  ◄──── HuggingFace Embeddings ◄──── Document Chunks
        │                                                     ▲
        │                                              PDF + Web Docs
        ▼
  Prompt Builder
  (Context Injection)
        │
        ▼
  LLM via OpenRouter
  (GPT-3.5-Turbo)
        │
        ▼
  Response → Streamlit Chat Interface
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|

| LLM Framework | LangChain (LCEL) |
| Vector Database | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM Provider | OpenRouter (GPT-3.5-Turbo) |
| Document Loaders | LangChain PyPDFLoader, WebBaseLoader |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
WorkBurnout_AI_assistant/
│
├── app.py                   # Streamlit chat interface (entry point)
├── .env                     # API keys (not committed)
├── requirements.txt         # Project dependencies
├── README.md
│
├── data/
│   └── documents/           # Source PDF documents
│       └── workburnout.pdf
│
├── faiss_index/             # Persisted FAISS vector index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
│
└── src/
    ├── __init__.py          # Package interface
    ├── ingestion.py         # Document loading, chunking, embeddings
    ├── vector_store.py      # FAISS index: build, save, load
    ├── retriever.py         # Semantic search logic
    └── llm_chain.py         # RAG chain with prompt engineering
```

---

## ⚙️ How It Works

### 1. Document Ingestion (`ingestion.py`)
Loads PDF documents and live web pages using LangChain document loaders. Text is split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50) to preserve context across boundaries.

### 2. Vector Embeddings & Storage (`vector_store.py`)
Each chunk is converted into a dense vector using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model. Vectors are stored in a FAISS index and persisted to disk — the index is built once and reloaded on subsequent runs for efficiency.

### 3. Semantic Retrieval (`retriever.py`)
When a user submits a query, it is embedded using the same model and compared against all document vectors via cosine similarity. The top-k most relevant chunks are returned as context.

### 4. LLM Response Generation (`llm_chain.py`)
Retrieved chunks are injected into a carefully engineered prompt template alongside the user query. The prompt instructs the LLM to be empathetic, grounded in the provided context, and to recommend professional help when appropriate. The chain is built using LangChain's modern LCEL (pipe `|`) syntax.

### 5. Chat Interface (`app.py`)
A Streamlit app provides the user-facing interface. The RAG chain is built once and cached in `session_state` to avoid re-loading embeddings on every message. Chat history is persisted across the session.

---












## 📄 License

MIT License — feel free to use this project as a reference or starting point.
