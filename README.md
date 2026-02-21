# ⚙️ OrchestratorX

> **Your AI Command Centre** — A unified multi-agent platform combining a conversational AI chatbot with a deep-research blog writing agent, all built on LangGraph and Streamlit.

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
OrchestratorX
├── 🎨 Frontend          Streamlit UI (streamlit_frontend.py)
│
├── 🤖 Chatbot Stack
│   ├── LangGraph Graph  chatBot — ReAct agent with tool-calling loop
│   ├── Tools            Web Search · Stock Price · Calculator · RAG
│   ├── Memory           SQLite-persisted conversation checkpoints
│   └── RAG Pipeline     PDF → Chunks → ChromaDB → MMR Retrieval
│
├── ✍️ Blog Agent Stack
│   ├── Router Node      Classifies topic → closed_book / hybrid / open_book / rag_grounded
│   ├── Research Node    DuckDuckGo web search with recency filtering
│   ├── RAG Node         Optional grounding from uploaded user documents
│   ├── Orchestrator     GPT-4.1-mini plans sections & fan-outs to workers
│   ├── Worker Nodes     Parallel section writers (LangGraph Send API)
│   ├── Reducer Node     Merges sections → decides image placements
│   └── Image Node       Generates & embeds images via Gemini
│
└── 🗄️ Data Layer
    ├── SQLite           Conversation checkpoints + thread metadata
    └── ChromaDB         Per-thread vector stores (persisted to disk)
```

---

## ✨ Features

### 💬 Multi-Utility Chatbot
- **Persistent multi-thread conversations** — switch between chats, names auto-generated after 3 messages
- **Real-time streaming responses** with tool-use status indicators
- **Web search** via DuckDuckGo — always up-to-date answers
- **Live stock prices** via Alpha Vantage API
- **Built-in calculator** for arithmetic operations
- **Per-thread PDF Q&A (RAG)** — upload a PDF to any conversation and ask questions about it; context is stored in ChromaDB and persists across sessions
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
|-------|-----------|
| **Frontend** | Streamlit |
| **Agent Framework** | LangGraph (StateGraph, Send API, SqliteSaver) |
| **LLMs** | OpenAI GPT-4o-mini (chat) · GPT-4.1-mini (blog) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Image Generation** | Google Gemini |
| **Vector Store** | ChromaDB (persisted per thread) |
| **Web Search** | DuckDuckGo (via LangChain) |
| **Stock Data** | Alpha Vantage API |
| **Memory / Checkpointing** | SQLite via LangGraph SqliteSaver |
| **PDF Ingestion** | LangChain PyPDFLoader + RecursiveCharacterTextSplitter |
| **Observability** | LangSmith (project: OrchestratorX) |

---

## 📂 Project Structure

```
orchestratorx/
├── streamlit_frontend.py          # Main Streamlit UI — chatbot & blog agent views
├── langgraph_backend.py           # LangGraph graphs: chatBot + blog_agent + storage helpers
├── rag_utility.py                 # PDF ingestion, ChromaDB vector store, per-thread retriever
├── utility_tools.py               # LangChain tools: RAG, web search, stock price, calculator
├── sqlite_functions.py            # SQLite helpers for thread metadata (names, created_at)
├── streamlit_utility_functions.py # Streamlit session helpers: thread management, conversation naming
├── .env                           # API keys (not committed)
├── requirements.txt               # Python dependencies
├── chatbot.db                     # SQLite DB: conversation checkpoints + thread metadata
├── chatbot_chroma/                # ChromaDB vector stores, one sub-folder per thread_id
│   └── <thread_id>/
└── generated_blogs/               # Auto-saved blog markdown files with metadata headers
    └── <timestamp>_<title>.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/orchestratorx.git
cd orchestratorx
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

### 4. Set up environment variables

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key          # For Gemini image generation
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
LANGSMITH_API_KEY=your_langsmith_api_key    # Optional — for tracing
LANGSMITH_TRACING=true                      # Optional
```

### 5. Run the app
```bash
streamlit run streamlit_frontend.py
```

---

## 🔄 How the Blog Agent Works

```
Topic Input
    │
    ▼
┌─────────┐    closed_book ──────────────────────────────┐
│ Router  │    hybrid      ──► Research Node (web search) │
│  Node   │    open_book   ──► Research Node (web search) │
│         │    rag_grounded ─► RAG Retrieval Node         │
└─────────┘                                              │
                                                         ▼
                                               ┌──────────────────┐
                                               │  Orchestrator    │
                                               │  (plans sections │
                                               │   + fan-out)     │
                                               └──────────────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼          ▼          ▼
                                         Worker 1   Worker 2   Worker N
                                         (section)  (section)  (section)
                                              └──────────┬──────────┘
                                                         ▼
                                               ┌──────────────────┐
                                               │     Reducer      │
                                               │  merge → images  │
                                               │  → final blog    │
                                               └──────────────────┘
```

---

## 📸 Screenshots

> *Add screenshots of your chatbot and blog agent UI here.*

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

- The SQLite database (`chatbot.db`) and ChromaDB folders (`chatbot_chroma/`) are created automatically on first run.
- Generated blogs are saved to `generated_blogs/` with a metadata header containing title, timestamp, and thread ID.
- Each chat thread has its own isolated vector store — uploading a PDF in one thread does not affect other threads.
- The blog agent uses LangGraph's `Send` API for parallel section writing, meaning all sections are written concurrently.

---

## 📄 License

This project is licensed under the MIT License.
