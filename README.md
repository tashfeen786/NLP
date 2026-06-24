# 🧠 NLP & RAG Engineering — A Practical Learning Path

<div align="center">

A hands-on, end-to-end curriculum covering **Retrieval-Augmented Generation (RAG)** from fundamentals to production deployment and agentic systems.

14 modules · Jupyter Notebooks · LangChain · LangGraph · FastAPI · Qdrant · Pinecone

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 About

This repository is a structured, practical guide to building production-grade NLP and RAG systems. Each module builds on the previous — starting from raw document ingestion and ending with deployed agentic AI systems that can retrieve, reason, and produce reports.

Every concept is implemented from scratch in Jupyter Notebooks with working code, not just theory.

---

## 🗺️ Learning Path

```
Documents → Chunks → Embeddings → Vector DB → Retrieval → RAG Pipeline
    ↓
Evaluation → Debugging → Hybrid Search → Agentic RAG → Deployment → Capstones
```

---

## 📚 Modules

| # | Module | Key Concepts |
|---|---|---|
| 01 | **Document Loaders — Ingesting Data for RAG** | PDF, web, CSV, Notion loaders; LangChain document loaders |
| 02 | **Chunking Strategies — Text Splitters** | Fixed-size, recursive, semantic, markdown splitters; overlap strategies |
| 03 | **Embeddings Explained — Dense vs Sparse** | OpenAI embeddings, sentence-transformers, TF-IDF, BM25 |
| 04 | **Vector Databases — Qdrant & Pinecone** | Indexing, similarity search, namespaces, metadata filtering |
| 05 | **Semantic Search & Retrievers** | MMR, similarity threshold, contextual compression, multi-query |
| 06 | **Building Vanilla RAG with LCEL** | End-to-end RAG pipeline, prompt engineering, LangChain Expression Language |
| 07 | **RAG Evaluation with RAGAS** | Faithfulness, answer relevancy, context recall, precision metrics |
| 08 | **Common RAG Issues & Solutions** | Hallucination debugging, retrieval failures, chunk quality fixes |
| 09 | **Hybrid Search — Dense + BM25 + RRF** | Combining semantic and keyword search, Reciprocal Rank Fusion |
| 10 | **Agentic RAG with LangGraph** | Reflect & improve loops, self-correcting RAG, graph-based agents |
| 11 | **CAG — Cache Augmented Generation** | KV-cache injection, latency optimization, long-context strategies |
| 12 | **Deploying RAG with FastAPI** | Production API, async endpoints, Docker-ready RAG service |
| 13 | **Capstone 1 — Enterprise Document Q&A** | Production-grade RAG assistant, end-to-end build |
| 14 | **Capstone 2 — Agentic Research Assistant** | Retrieve, analyze & produce structured reports with LangGraph |

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Notebooks** | Jupyter Notebook / JupyterLab |
| **LLM Framework** | LangChain, LangGraph, LCEL |
| **Embeddings** | OpenAI `text-embedding-3-small`, Sentence-Transformers |
| **Vector Databases** | Qdrant, Pinecone, ChromaDB |
| **Evaluation** | RAGAS |
| **Deployment** | FastAPI, Uvicorn |
| **Search** | BM25, Dense retrieval, RRF (Reciprocal Rank Fusion) |
| **Agent Framework** | LangGraph (StateGraph, ToolNode, conditional edges) |

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab
- API keys for OpenAI / Gemini (used in notebooks)

### 1. Clone

```bash
git clone https://github.com/tashfeen786/NLP.git
cd NLP
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install per-module (each folder has its own requirements where needed):

```bash
pip install langchain langchain-community langgraph openai sentence-transformers \
            chromadb qdrant-client pinecone-client ragas fastapi uvicorn jupyter
```

### 3. Set API Keys

Create a `.env` file in the root:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
QDRANT_API_KEY=your_qdrant_key     # if using cloud
QDRANT_URL=your_qdrant_url         # if using cloud
```

### 4. Run Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Open any module folder and run the notebook step by step.

---

## 🎯 What You'll Build

By the end of this curriculum you will have built:

**Module 13 — Enterprise Document Q&A System**
- Upload any document (PDF, DOCX, TXT)
- Chunking + embedding pipeline
- Retrieval-augmented answer generation
- Deployed as a production FastAPI service

**Module 14 — Agentic Research Assistant**
- Multi-step research agent using LangGraph
- Retrieves from multiple sources
- Self-reflects and improves answers
- Produces structured analytical reports

---

## 📂 Repository Structure

```
NLP/
├── 1 Document Loaders (Ingesting Data for RAG)/
│   └── notebook.ipynb
├── 2 Chunking Strategies for RAG Text Splitters/
│   └── notebook.ipynb
├── 3 Embeddings Explained Dense vs Sparse/
│   └── notebook.ipynb
├── 4 Vector Databases Qdrant & Pinecone/
│   └── notebook.ipynb
├── 5 Semantic Search & Retrievers/
│   └── notebook.ipynb
├── 6 Building Vanilla RAG with LCEL/
│   └── notebook.ipynb
├── 7 RAG Evaluation with RAGAS/
│   └── notebook.ipynb
├── 8 Common RAG Issues & Solutions/
│   └── notebook.ipynb
├── 9 Hybrid Search Dense + BM25 + RRF/
│   └── notebook.ipynb
├── 10 Agentic RAG with LangGraph/
│   └── notebook.ipynb
├── 11 CAG (Cache Augmented Generation)/
│   └── notebook.ipynb
├── 12 Deploying RAG with FastAPI/
│   └── notebook.ipynb + app.py
├── 13 Capstone 1 — Enterprise Document Q&A/
│   └── notebook.ipynb + full project
├── 14 Capstone 2 — Agentic Research Assistant/
│   └── notebook.ipynb + full project
└── README.md
```

---

## 💡 Key Concepts Covered

**RAG Fundamentals**
- Document ingestion from multiple sources
- Chunking with overlap for context preservation
- Dense vs sparse embeddings and when to use each
- Vector similarity search (cosine, dot product, euclidean)

**Advanced Retrieval**
- Hybrid search combining semantic + keyword (BM25 + RRF)
- Multi-query retrieval for comprehensive coverage
- Contextual compression to reduce noise
- Metadata filtering for scoped retrieval

**Quality & Debugging**
- RAGAS evaluation framework (faithfulness, relevancy, recall)
- Common failure modes and systematic fixes
- Hallucination detection and mitigation

**Agentic Systems**
- LangGraph state machines for multi-step agents
- Self-reflection and answer improvement loops
- Tool-calling agents with RAG as a tool

**Production**
- FastAPI deployment with async endpoints
- Cache Augmented Generation for low latency
- End-to-end capstone projects

---

## 🚀 Recommended Learning Order

Follow the numbered modules in sequence — each one builds on the previous:

```
01 → 02 → 03 → 04 → 05 → 06   (Core RAG pipeline)
                              ↓
                    07 → 08 → 09   (Quality & advanced retrieval)
                                  ↓
                         10 → 11 → 12   (Agentic & deployment)
                                        ↓
                               13 → 14   (Capstone projects)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b module/your-addition`
3. Commit: `git commit -m "Add: your module description"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Tashfeen Aziz**  
GitHub: [@tashfeen786](https://github.com/tashfeen786)

---

<div align="center">

From raw documents to production-grade agentic AI — built step by step.

🧠 LangChain · LangGraph · RAG · FastAPI · Qdrant · Pinecone · RAGAS

</div>
