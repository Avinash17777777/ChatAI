from __future__ import annotations

import os
import json
import tempfile
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# ----------------------------------------------
# Persistent Vector-store paths
# -----------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstores"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True,)

# -------------------
# 1. LLM + embeddings
# -------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

llm = ChatGroq(
        model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=GROQ_API_KEY,

)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={
        "normalize_embeddings": True
    },
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_vectorstore_path(thread_id: str) -> Path:
    return VECTORSTORE_DIR / str(thread_id)


def _get_metadata_path(thread_id: str) -> Path:
    return _get_vectorstore_path(thread_id) / "metadata.json"


def _get_retriever(thread_id: Optional[str]):
    """
    Get the retriever from memory.

    If it is not in memory, try to restore the FAISS
    vector store from disk.
    """

    if not thread_id:
        return None

    thread_id = str(thread_id)

    # Already available in memory
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    vectorstore_path = _get_vectorstore_path(thread_id)

    # No saved vector store exists
    if not vectorstore_path.exists():
        return None

    try:
        vector_store = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[thread_id] = retriever

        return retriever

    except Exception as e:
        print(
            f"Failed to load vector store for thread "
            f"{thread_id}: {e}"
        )

        return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF.

    The vector store and metadata are saved permanently
    using the thread ID.
    """

    if not file_bytes:
        raise ValueError(
            "No bytes received for ingestion."
        )

    thread_id = str(thread_id)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError(
                "No readable content was found in the PDF."
            )

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "The PDF did not produce any searchable text chunks."
            )

        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        # Get persistent storage path
        vectorstore_path = _get_vectorstore_path(
            thread_id
        )

        vectorstore_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save FAISS to disk
        vector_store.save_local(
            str(vectorstore_path)
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Save retriever in memory
        _THREAD_RETRIEVERS[thread_id] = retriever

        # Create metadata
        metadata = {
            "filename": (
                filename
                or os.path.basename(temp_path)
            ),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        # Save metadata in memory
        _THREAD_METADATA[thread_id] = metadata

        # Save metadata permanently
        with open(
            _get_metadata_path(thread_id),
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(metadata,file, indent=4,)
        return metadata

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
SEARCH_REGION = os.getenv(
    "SEARCH_REGION",
    "wt-wt",
)

search_tool = DuckDuckGoSearchRun(
    region=SEARCH_REGION
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price.

    Example symbols:
    AAPL, TSLA, MSFT
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": (
                "ALPHA_VANTAGE_API_KEY "
                "is not configured."
            )
        }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": api_key,
            },
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        return {
            "error": f"Stock API request failed: {str(e)}"
        }



@tool
def rag_tool(query: str, thread_id: Optional[str] = None,) -> dict:
    """
    Retrieve relevant information from the uploaded PDF
    for the current chat thread.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": (
                "No document indexed for this chat. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    try:
        results = retriever.invoke(query)
        context = [doc.page_content for doc in results]
        sources = [doc.metadata for doc in results]
        document_metadata = thread_document_metadata(str(thread_id))

        return {
            "query": query,
            "context": context,
            "metadata": sources,
            "source_file": document_metadata.get(
                "filename"
            ),
        }

    except Exception as e:
        return {
            "error": f"RAG retrieval failed: {str(e)}",
            "query": query,
        }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads() -> list[str]:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = (checkpoint.config.get("configurable", {}).get("thread_id"))

        if thread_id:
            all_threads.add(str(thread_id))
    return sorted(all_threads)


def thread_has_document(thread_id: str) -> bool:
    thread_id = str(thread_id)
    if thread_id in _THREAD_RETRIEVERS:
        return True
    vectorstore_path = _get_vectorstore_path(thread_id)
    return vectorstore_path.exists()


def thread_document_metadata(thread_id: str,) -> dict:

    thread_id = str(thread_id)

    # Already in memory
    if thread_id in _THREAD_METADATA:
        return _THREAD_METADATA[thread_id]

    metadata_path = _get_metadata_path(thread_id)

    # No persistent metadata
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path,"r", encoding="utf-8",) as file:
            metadata = json.load(file)
        _THREAD_METADATA[thread_id] = metadata
        return metadata

    except Exception as e:
        print(
            f"Failed to load metadata "
            f"for thread {thread_id}: {e}"
        )
        return {}
