# 🤖 Multi-Utility LangGraph PDF Chatbot

A Streamlit-based AI chatbot built with **LangGraph, LangChain, Groq, Hugging Face embeddings, FAISS, and SQLite**.

The application supports normal conversations, PDF-based RAG, web search, stock-price lookup, calculator operations, conversation persistence, and thread-based chat history.

---


🌐 **Live Demo:**  
https://chatai-fdk22pnw32hjndidupktjc.streamlit.app/


## ✨ Features

* 💬 General AI conversation
* 📄 PDF upload and question answering
* 🔎 Retrieval-Augmented Generation (RAG)
* 🧠 FAISS vector database for document retrieval
* 🔢 Calculator tool
* 🌐 DuckDuckGo web search
* 📈 Alpha Vantage stock-price tool
* 🧵 Thread-based conversations
* 💾 SQLite conversation persistence
* 🔄 Persistent FAISS vector stores
* ⚡ Streaming AI responses
* 🛠️ LangGraph tool-calling workflow
* 🤗 Hugging Face sentence-transformer embeddings
* 🎨 Streamlit frontend

---

# 🏗️ Project Architecture

```text
ChatBot/
│
├── backend/
│   └── chatbot.py
│       ├── LLM configuration
│       ├── Hugging Face embeddings
│       ├── PDF ingestion
│       ├── FAISS vector stores
│       ├── RAG tool
│       ├── Web search tool
│       ├── Calculator tool
│       ├── Stock price tool
│       ├── LangGraph state
│       ├── Chat node
│       ├── Tool node
│       ├── SQLite checkpointing
│       └── Thread helpers
│
├── frontend/
│   └── app.py
│       ├── Streamlit interface
│       ├── Chat interface
│       ├── PDF uploader
│       ├── Thread management
│       ├── Conversation history
│       └── Streaming responses
│
├── data/
│   └── vectorstores/
│       └── <thread_id>/
│           ├── index.faiss
│           ├── index.pkl
│           └── metadata.json
│
├── .env
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── chatbot.db
```

---

# 🔄 System Workflow

The chatbot uses LangGraph to control the conversation and tool-calling process.

```text
                         User
                          │
                          ▼
                  ┌───────────────┐
                  │   Streamlit   │
                  │    Frontend   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   LangGraph   │
                  │   Chat Node   │
                  └───────┬───────┘
                          │
                 Tool required?
                    ┌─────┴─────┐
                    │           │
                   NO          YES
                    │           │
                    ▼           ▼
                  Answer     ToolNode
                                │
             ┌──────────────────┼─────────────────┐
             │                  │                 │
             ▼                  ▼                 ▼
        Web Search         Calculator          Stocks
             │                  │                 │
             └──────────────────┼─────────────────┘
                                │
                                ▼
                           RAG Tool
                                │
                                ▼
                          FAISS Retriever
                                │
                                ▼
                           PDF Context
                                │
                                ▼
                           Chat Node
                                │
                                ▼
                           Final Answer
```

---

# 📄 PDF / RAG Pipeline

When a PDF is uploaded, the application performs the following steps:

```text
PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Document Pages
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
Text Chunks
 │
 ▼
Hugging Face Embeddings
 │
 ▼
FAISS Vector Store
 │
 ▼
Persistent Storage
```

When the user asks a question about the document:

```text
User Question
      │
      ▼
    LLM
      │
      ▼
 rag_tool
      │
      ▼
 FAISS Retriever
      │
      ▼
Relevant Chunks
      │
      ▼
   LLM Context
      │
      ▼
 Final Answer
```

The current chunking configuration is:

```text
Chunk size:      1000
Chunk overlap:   200
Retrieval K:     4
Embedding model: sentence-transformers/all-MiniLM-L6-v2
```

---

# 🧠 LLM

The application uses Groq through LangChain.

Current model configuration:

```text
Provider: Groq
Model: openai/gpt-oss-120b
Temperature: 0.1
```

The API key is loaded from the environment.

---

# 🤗 Embeddings

The application uses Hugging Face sentence-transformer embeddings:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized:

```python
encode_kwargs={
    "normalize_embeddings": True
}
```

These embeddings are used to convert PDF text chunks into vectors for FAISS retrieval.

---

# 🗃️ Vector Database

FAISS is used as the local vector database.

Each thread has its own vector-store directory:

```text
data/vectorstores/<thread_id>/
```

This allows document storage to be associated with individual conversations.

FAISS is persisted to disk so the vector store can be restored after restarting the application.

---

# 🧰 Available Tools

The chatbot currently provides four tools.

## 1. Calculator

Supports:

```text
add
sub
mul
div
```

Example:

```text
Calculate 125 * 48
```

The LLM can call the calculator automatically when appropriate.

---

## 2. Web Search

The chatbot uses DuckDuckGo through LangChain.

Example:

```text
Who is the current Prime Minister of India?
```

The model can use web search when current information is required.

---

## 3. Stock Price

The stock tool uses Alpha Vantage.

Example:

```text
What is the latest AAPL stock price?
```

Supported symbols depend on Alpha Vantage.

---

## 4. PDF RAG

The RAG tool retrieves relevant information from the PDF associated with the current chat thread.

Example:

```text
According to the uploaded PDF, what is the project budget?
```

---

# 🧵 Conversation Threads

Each conversation receives a unique UUID:

```text
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The thread ID is used for:

* Conversation persistence
* LangGraph checkpointing
* PDF/vector-store association
* Conversation history
* Document metadata

Example:

```text
Thread
  │
  ├── Chat history
  │
  └── PDF vector store
```

---

# 💾 Conversation Persistence

LangGraph uses SQLite checkpointing.

```text
LangGraph
    │
    ▼
SqliteSaver
    │
    ▼
SQLite
    │
    ▼
chatbot.db
```

This allows conversations to persist between Streamlit sessions.

---

# 🖥️ Frontend

The frontend is implemented using Streamlit.

Main frontend responsibilities:

* Display chatbot UI
* Accept user messages
* Upload PDFs
* Display conversation history
* Display previous threads
* Stream AI responses
* Display tool execution status
* Display document information

Run the application with:

```powershell
uv run streamlit run frontend/app.py
```

---

# ⚙️ Requirements

The project uses Python 3.10+.

Important dependencies include:

```text
streamlit
langchain
langchain-core
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
langgraph
langgraph-checkpoint-sqlite
faiss-cpu
sentence-transformers
pypdf
ddgs
requests
python-dotenv
```

---

# 🚀 Installation

## 1. Clone or create the project

```powershell
cd D:\ChatBot
```

---

## 2. Create a UV environment

If you haven't created one:

```powershell
uv venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

If using `pyproject.toml`:

```powershell
uv sync
```

Or install individual dependencies:

```powershell
uv add streamlit
uv add langchain
uv add langchain-community
uv add langchain-groq
uv add langchain-huggingface
uv add langchain-text-splitters
uv add langgraph
uv add langgraph-checkpoint-sqlite
uv add faiss-cpu
uv add sentence-transformers
uv add pypdf
uv add ddgs
uv add requests
uv add python-dotenv
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
SEARCH_REGION=wt-wt
```

Never commit `.env` to Git.

Add this to `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
chatbot.db
data/vectorstores/
```

---

# ▶️ Running the Application

From the project root:

```powershell
cd D:\ChatBot
```

Run:

```powershell
uv run streamlit run frontend/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 Testing the Application

After starting the application, test each major feature independently.

## Basic Chat

Ask:

```text
What is machine learning?
```

Expected:

A normal AI-generated response.

---

## Calculator

Ask:

```text
Calculate 125 × 48 + 350.
```

Expected:

```text
6350
```

---

## Web Search

Ask:

```text
Search the web and tell me the current Prime Minister of India.
```

Expected:

The chatbot should invoke the web-search tool.

---

## Stock Tool

Ask:

```text
What is the latest AAPL stock price?
```

Expected:

The stock tool should be invoked and return the available quote.

---

## PDF RAG

Upload a PDF and ask a question whose answer exists specifically inside the document.

For example:

```text
According to the uploaded PDF, what is the project budget?
```

The answer should be based on the document.

---

## RAG Hallucination Test

Ask about information that does not exist in the PDF:

```text
According to the uploaded PDF, what is the author's favorite programming language?
```

If that information isn't present, the chatbot should say that it cannot find the information rather than inventing it.

---

## Thread Isolation Test

### Thread A

Upload:

```text
company_a.pdf
```

Ask:

```text
What is the company name?
```

### Thread B

Create a new chat and upload:

```text
company_b.pdf
```

Ask:

```text
What is the company name?
```

The answers should come from the appropriate document.

---

# 🔍 Troubleshooting

## `ModuleNotFoundError: No module named 'backend'`

Run Streamlit from the project root:

```powershell
cd D:\ChatBot
uv run streamlit run frontend/app.py
```

The frontend also adds the project root to `sys.path`.

---

## `No module named 'langgraph.checkpoint.sqlite'`

Install the SQLite checkpoint integration:

```powershell
uv add langgraph-checkpoint-sqlite
```

Verify:

```powershell
uv run python -c "from langgraph.checkpoint.sqlite import SqliteSaver; print('SqliteSaver OK')"
```

---

## FAISS ImportError

Install FAISS:

```powershell
uv add faiss-cpu
```

Verify:

```powershell
uv run python -c "import faiss; print('FAISS OK')"
```

Then verify LangChain:

```powershell
uv run python -c "from langchain_community.vectorstores import FAISS; print('LangChain FAISS OK')"
```

---

## DuckDuckGo ImportError

Install:

```powershell
uv add ddgs
```

Verify:

```powershell
uv run python -c "from langchain_community.tools import DuckDuckGoSearchRun; print('DuckDuckGo OK')"
```

---

## PDF Loader Error

Install:

```powershell
uv add pypdf
```

Verify:

```powershell
uv run python -c "from langchain_community.document_loaders import PyPDFLoader; print('PDF loader OK')"
```

---

# 🔒 Security Considerations

The application currently uses:

```python
FAISS.load_local(
    ...,
    allow_dangerous_deserialization=True
)
```

This should only be used with FAISS files created and controlled by the application.

Do not load untrusted FAISS indexes without understanding the security implications.

API keys should always be stored in environment variables rather than hard-coded in Python files.

---

# ⚠️ Current Architectural Considerations

The current implementation associates FAISS storage with a thread:

```text
data/vectorstores/<thread_id>/
```

The frontend currently allows users to upload multiple PDFs to a thread, but the current backend ingestion implementation creates a new FAISS store for each ingestion.

Therefore, if multiple PDFs are uploaded to the same thread, the latest ingestion can replace the previous vector store.

For reliable multi-PDF-per-thread support, the vector store should eventually be updated by adding new documents to the existing FAISS index rather than recreating it.

Another architectural improvement planned for the RAG system is to avoid making the LLM responsible for supplying the internal `thread_id` to the RAG tool. The application should provide the thread context itself.

---

# 📁 Data Persistence

The application creates persistent data such as:

```text
chatbot.db
data/vectorstores/
```

These files should generally not be committed to Git.

Example `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
chatbot.db
data/vectorstores/
```

---

# 🧩 Technology Stack

| Component                | Technology                         |
| ------------------------ | ---------------------------------- |
| Frontend                 | Streamlit                          |
| LLM                      | Groq                               |
| Model                    | openai/gpt-oss-120b                |
| Agent Framework          | LangGraph                          |
| LLM Framework            | LangChain                          |
| Embeddings               | Hugging Face Sentence Transformers |
| Embedding Model          | all-MiniLM-L6-v2                   |
| Vector Database          | FAISS                              |
| PDF Loader               | PyPDFLoader                        |
| Text Splitter            | RecursiveCharacterTextSplitter     |
| Conversation Persistence | SQLite                             |
| Checkpointing            | LangGraph SqliteSaver              |
| Web Search               | DuckDuckGo                         |
| Stock Data               | Alpha Vantage                      |
| Environment Management   | UV                                 |
| Configuration            | python-dotenv                      |
| Frontend Language        | Python                             |

---

# 🧠 Core LangGraph Architecture

The graph consists of two main nodes:

```text
START
  │
  ▼
chat_node
  │
  ▼
tools_condition
  │
  ├─────────────── No tool ──────────────► END
  │
  └─────────────── Tool required
                          │
                          ▼
                       tools
                          │
                          ▼
                      chat_node
```

The chatbot can therefore decide dynamically whether it needs to use:

* Search
* Calculator
* Stock API
* PDF RAG

before producing the final answer.

---

# 📌 Example Use Cases

This project can be used for:

* Personal document assistants
* Research assistants
* PDF question-answering
* Financial information lookup
* General-purpose AI assistants
* AI agent demonstrations
* LangGraph learning projects
* RAG demonstrations
* Portfolio projects

---

# 🚧 Future Improvements

Potential improvements include:

* [ ] Support multiple PDFs in a single thread without overwriting FAISS
* [ ] Improve RAG source citations
* [ ] Inject thread context into RAG without exposing it to the LLM
* [ ] Add document deletion
* [ ] Add document replacement
* [ ] Add PDF page references to answers
* [ ] Add better RAG relevance filtering
* [ ] Add hybrid search
* [ ] Add MMR retrieval
* [ ] Improve conversation loading
* [ ] Improve tool error handling
* [ ] Normalize stock API responses
* [ ] Move configuration into a dedicated configuration module
* [ ] Separate tools into individual modules
* [ ] Separate RAG functionality into its own module
* [ ] Add automated tests
* [ ] Add logging
* [ ] Add authentication
* [ ] Add database-backed document metadata
* [ ] Add production deployment configuration

---

# 👨‍💻 Development

The project is designed as an existing modular AI application rather than a single chatbot script.

The main responsibilities are currently divided between:

```text
frontend/app.py
        │
        ▼
backend/chatbot.py
        │
        ├── LLM
        ├── LangGraph
        ├── Tools
        ├── RAG
        ├── FAISS
        └── SQLite
```

As the application grows, these responsibilities can be separated into dedicated modules.

---

# 📜 License

This project is intended as a personal/educational AI project.

Add an appropriate open-source license here if you decide to publish the project publicly.

---

# ⭐ Project Status

**Current status:** Functional development version.

Implemented:

* ✅ LangGraph agent workflow
* ✅ Groq LLM
* ✅ Tool calling
* ✅ Calculator
* ✅ Web search
* ✅ Stock price lookup
* ✅ PDF ingestion
* ✅ Hugging Face embeddings
* ✅ FAISS vector storage
* ✅ RAG retrieval
* ✅ SQLite checkpointing
* ✅ Conversation threads
* ✅ Streamlit interface
* ✅ Streaming responses

The RAG/document subsystem is still an area for further improvement, particularly around multi-document handling and thread-context management.
