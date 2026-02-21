# ⚙️ OrchestratorX

> **Your AI Command Centre** — A unified multi-agent platform combining a conversational AI chatbot with a deep-research blog writing agent, all built on LangGraph and Streamlit.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/streamlit-1.54+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/langgraph-1.0.8+-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/langchain-1.2.10+-1C3C3C?style=flat-square&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/openai-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/chromadb-1.1.0+-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square"/>
</p>

---

## 🌟 Overview

OrchestratorX is a full-stack AI application that brings together two powerful tools under a single, clean interface:

| Tool | Description |
|------|-------------|
| 💬 **Multi-Utility Chatbot** | Conversational AI with persistent memory, web search, stock prices, calculator, and per-thread PDF Q&A via RAG |
| ✍️ **Blog Writing Agent** | Multi-agent LangGraph pipeline that autonomously researches, plans, writes, and illustrates long-form technical blog posts |

---

## 🏗️ Architecture

```
OrchestratorX/
├── 🎨 Frontend          orchestratorx_app.py  — Streamlit UI
│
├── 🤖 Chatbot Stack
│   ├── LangGraph Graph  chatBot — ReAct agent with tool-calling loop
│   ├── Tools            Web Search · Stock Price · Calculator · RAG
│   ├── Memory           SQLite-persisted conversation checkpoints
│   └── RAG Pipeline     PDF → Chunks → ChromaDB → MMR Retrieval
│
├── ✍️ Blog Agent Stack
│   ├── Router Node      Classifies topic → closed_book / hybrid / open_book / rag_grounded
│   ├── Research Node    Web search with recency filtering
│   ├── RAG Node         Optional grounding from uploaded user documents
│   ├── Orchestrator     GPT-4.1-mini plans sections & fans out to workers
│   ├── Worker Nodes     Parallel section writers (LangGraph Send API)
│   ├── Reducer Node     Merges sections → decides image placements
│   └── Image Node       Generates & embeds images via Google Gemini
│
└── 🗄️ Data Layer
    ├── SQLite           Conversation checkpoints + thread metadata
    └── ChromaDB         Per-thread vector stores (persisted to disk)
```

### 🤖 Chatbot — LangGraph Workflow

<p align="center">
  <img src="utils_img/Chatbot Workflow.png" alt="Chatbot LangGraph workflow" width="240"/>
</p>

The chatbot is a **ReAct loop**: `chat_node` calls GPT-4o-mini with bound tools. If the model decides to use a tool, control passes to the `tools` node and loops back — otherwise it routes to `__end__`.

### ✍️ Blog Writing Agent — LangGraph Workflow

<p align="center">
  <img src="utils_img/agent workflow.png" alt="Blog Writing Agent LangGraph workflow" width="240"/>
</p>

The blog agent is a **multi-stage pipeline**: the `router` classifies the topic and conditionally branches to `research` (web search) and/or the `orchestrator`. The orchestrator fans out to parallel `worker` nodes (one per section), which converge at the `reducer` for merging, image generation, and final assembly.

---

## ✨ Features

### 💬 Multi-Utility Chatbot
- **Persistent multi-thread conversations** — switch between chats, names auto-generated after 3 messages
- **Real-time streaming responses** with tool-use status indicators
- **Web search** — always up-to-date answers
- **Live stock prices** via Alpha Vantage API
- **Built-in calculator** for arithmetic operations
- **Per-thread PDF Q&A (RAG)** — upload a PDF to any conversation and ask questions about it; stored in ChromaDB and persists across sessions
- **Conversation history** fully restored on page reload

### ✍️ Blog Writing Agent
- **Intelligent routing** — automatically decides whether to use web research, RAG, or closed-book generation based on the topic
- **Parallel section writing** — uses LangGraph's `Send` API to write blog sections concurrently
- **Deep research** — runs 5–8 targeted search queries and collects evidence with source attribution
- **Structured planning** — generates a full `BlogPlan` with tasks, target word counts, tone, and audience
- **AI image generation** — plans image placements with captions, generates via Gemini, and embeds as base64
- **Auto-save** — every generated blog saved to disk with metadata
- **Load & Delete** saved blogs from the sidebar
- **Download** as Markdown or a full bundle (MD + images zip)
- **Live progress tracking** — real-time streaming with per-node status updates, evidence count, and section progress

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit 1.54+ |
| **Agent Framework** | LangGraph 1.0.8+ (StateGraph, Send API, SqliteSaver) |
| **LLMs** | OpenAI GPT-4o-mini (chat) · GPT-4.1-mini (blog) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Image Generation** | Google Gemini (`google-genai`) |
| **Vector Store** | ChromaDB (persisted per thread) |
| **Web Search** | DuckDuckGo (`ddgs`) |
| **Stock Data** | Alpha Vantage API |
| **Memory / Checkpointing** | SQLite via `langgraph-checkpoint-sqlite` |
| **PDF Ingestion** | LangChain PyPDFLoader + RecursiveCharacterTextSplitter |
| **Observability** | LangSmith (project: OrchestratorX) |

---

## 📂 Project Structure

```
OrchestratorX/
├── App/
│   ├── orchestratorx_app.py           # Main Streamlit UI — chatbot & blog agent views
│   ├── langgraph_backend.py           # LangGraph graphs: chatBot + blog_agent + storage helpers
│   ├── rag_utility.py                 # PDF ingestion, ChromaDB vector store, per-thread retriever
│   ├── utility_tools.py               # LangChain tools: RAG, web search, stock price, calculator
│   ├── sqlite_functions.py            # SQLite helpers for thread metadata (names, created_at)
│   └── streamlit_utility_functions.py # Streamlit session helpers: thread management, naming
│
├── utils_img/
│   ├── Chatbot Workflow.png           # LangGraph diagram — chatbot ReAct loop
│   └── agent workflow.png             # LangGraph diagram — blog writing agent pipeline
│
├── .env                               # API keys (not committed)
├── .gitignore
├── .python-version                    # Python 3.11
├── pyproject.toml                     # Project metadata & dependencies
├── requirements.txt                   # Pip-installable dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sema-phore/-OrchestratorX.git
cd OrchestratorX
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Using `uv`?** A `uv.lock` file is included — run `uv sync` instead.

### 4. Set up environment variables

Create a `.env` file inside the `App/` folder:
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key             # For Gemini image generation
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
LANGSMITH_API_KEY=your_langsmith_api_key       # Optional — for tracing
LANGSMITH_TRACING=true                         # Optional
```

### 5. Run the app
```bash
cd App
streamlit run orchestratorx_app.py
```

---

## 🔄 How the Blog Agent Works

```
Topic Input
    │
    ▼
┌──────────┐   closed_book   ─────────────────────────────────────┐
│  Router  │   hybrid        ──► Research Node (web search)        │
│   Node   │   open_book     ──► Research Node (web search)        │
│          │   rag_grounded  ──► RAG Retrieval Node                │
└──────────┘                                                       │
                                                                   ▼
                                                      ┌─────────────────────┐
                                                      │    Orchestrator     │
                                                      │  plans sections &   │
                                                      │  fans out via Send  │
                                                      └─────────────────────┘
                                                                   │
                                           ┌───────────────────────┼──────────────────────┐
                                           ▼                       ▼                      ▼
                                       Worker 1               Worker 2              Worker N
                                       (section)              (section)             (section)
                                           └───────────────────────┬──────────────────────┘
                                                                   ▼
                                                      ┌─────────────────────┐
                                                      │      Reducer        │
                                                      │  merge → images     │
                                                      │  → final blog MD    │
                                                      └─────────────────────┘
```

---

## 🔑 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Powers GPT-4o-mini (chat) and GPT-4.1-mini (blog) |
| `GOOGLE_API_KEY` | ✅ Yes | Gemini image generation in blog agent |
| `ALPHA_VANTAGE_API_KEY` | ⚠️ Optional | Live stock prices (free tier available) |
| `LANGSMITH_API_KEY` | ⚠️ Optional | LangSmith tracing and observability |
| `LANGSMITH_TRACING` | ⚠️ Optional | Set to `true` to enable tracing |

---

## 📝 Notes

- The SQLite database (`chatbot.db`) and ChromaDB folders are created automatically on first run inside `App/`.
- Generated blogs are saved to `generated_blogs/` with a metadata header containing title, timestamp, and thread ID.
- Each chat thread has its own isolated vector store — uploading a PDF in one thread does not affect other threads.
- The blog agent uses LangGraph's `Send` API for parallel section writing, meaning all sections are written concurrently.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.