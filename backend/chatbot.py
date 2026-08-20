import os
import re
import uuid
import shutil
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

CHROMA_DIR = BASE_DIR / "chroma_db"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to the .env file."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=GROQ_API_KEY,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, friendly and professional AI chatbot.

Rules:

1. Answer the user's actual question directly.
2. Keep answers clear and reasonably concise.
3. Use simple language when possible.
4. Use Markdown when it improves readability.
5. Do not repeatedly use the user's name.
6. Avoid unnecessary introductions.
7. Avoid unnecessary conclusions.
8. Do not say things like:
   "Feel free to ask anything."
   "Happy to help!"
   "Have a great day!"
   unless genuinely appropriate.
"""


# ============================================================
# RAG PROMPT
# ============================================================

RAG_PROMPT = """
You are answering a question using retrieved documents.

Rules:

1. Use the retrieved context when it is relevant.
2. Do not invent information that is not supported by the context.
3. If the context does not contain the answer, clearly say that.
4. You may use general knowledge for basic explanations,
   but do not pretend unsupported information came from the documents.
5. Keep the answer clear and reasonably concise.

Retrieved Context:

{context}
"""


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={
        "normalize_embeddings": True
    },
)


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents():

    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():

        if not file_path.is_file():
            continue

        try:

            extension = file_path.suffix.lower()

            if extension == ".pdf":

                loader = PyPDFLoader(
                    str(file_path)
                )

                docs = loader.load()

            elif extension in {".txt", ".md"}:

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                )

                docs = loader.load()

            elif extension == ".pptx":

                loader = UnstructuredPowerPointLoader(
                  str(file_path)
                )

                docs = loader.load()

            else:

                continue

            for doc in docs:

                doc.metadata["source"] = (
                    file_path.name
                )

            documents.extend(docs)

        except Exception as e:

            print(
                f"Could not load "
                f"{file_path.name}: {e}"
            )

    return documents


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    return splitter.split_documents(
        documents
    )


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_vector_store():

    vector_store = Chroma(
        collection_name="chatbot_documents",
        embedding_function=embeddings,
        persist_directory=str(
            CHROMA_DIR
        ),
    )

    existing_count = (
        vector_store._collection.count()
    )

    if existing_count == 0:

        documents = load_documents()

        if not documents:

            return vector_store

        chunks = split_documents(
            documents
        )

        vector_store.add_documents(
            chunks
        )

    return vector_store


vector_store = create_vector_store()


# ============================================================
# CHAT STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    retrieved_context: str

    use_rag: bool


# ============================================================
# RAG ROUTER NODE
# ============================================================

def router_node(
    state: ChatState
):

    messages = state["messages"]

    if not messages:

        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    question = messages[-1].content

    try:

        results = (
            vector_store
            .similarity_search_with_relevance_scores(
                question,
                k=4,
            )
        )

    except Exception:

        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    if not results:

        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    relevant_documents = []

    for document, score in results:

        if score >= 0.25:

            relevant_documents.append(
                document
            )

    if not relevant_documents:

        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    context_parts = []

    for document in relevant_documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        context_parts.append(
            f"[Source: {source}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    return {
        "retrieved_context": context,
        "use_rag": True,
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_question(
    state: ChatState
):

    if state.get("use_rag", False):

        return "rag"

    return "chat"


# ============================================================
# NORMAL CHAT NODE
# ============================================================

def chat_node(
    state: ChatState
):

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        *state["messages"],
    ]

    response = llm.invoke(
        messages
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# RAG NODE
# ============================================================

def rag_node(
    state: ChatState
):

    context = state.get(
        "retrieved_context",
        "",
    )

    prompt = RAG_PROMPT.format(
        context=context
    )

    messages = [
        SystemMessage(
            content=prompt
        ),
        *state["messages"],
    ]

    response = llm.invoke(
        messages
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "router",
    router_node,
)

graph.add_node(
    "chat",
    chat_node,
)

graph.add_node(
    "rag",
    rag_node,
)


graph.add_edge(
    START,
    "router",
)


graph.add_conditional_edges(
    "router",
    route_question,
    {
        "chat": "chat",
        "rag": "rag",
    },
)


graph.add_edge(
    "chat",
    END,
)

graph.add_edge(
    "rag",
    END,
)


# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = MemorySaver()


# ============================================================
# COMPILE
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# RESPONSE CLEANING
# ============================================================

def clean_response(
    text: str
):

    if not isinstance(
        text,
        str,
    ):

        text = str(text)

    text = text.replace(
        "\u202f",
        " ",
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = text.replace(
        r"\(",
        "",
    )

    text = text.replace(
        r"\)",
        "",
    )

    text = text.replace(
        r"\times",
        "×",
    )

    text = text.replace(
        r"\div",
        "÷",
    )

    text = text.replace(
        r"\cdot",
        "·",
    )

    text = re.sub(
        r"\{,\}",
        ",",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# THREAD
# ============================================================

def create_thread():

    return str(
        uuid.uuid4()
    )


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(
    thread_id: str,
    user_message: str,
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_message
                )
            ],
            "retrieved_context": "",
            "use_rag": False,
        },
        config=config,
    )

    messages = response.get(
        "messages",
        [],
    )

    if not messages:

        return (
            "I couldn't generate "
            "a response."
        )

    return clean_response(
        messages[-1].content
    )


# ============================================================
# HISTORY
# ============================================================

def get_history(
    thread_id: str,
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    snapshot = chatbot.get_state(
        config
    )

    messages = snapshot.values.get(
        "messages",
        [],
    )

    history = []

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            role = "user"

        else:

            role = "assistant"

        history.append(
            {
                "role": role,
                "content": clean_response(
                    message.content
                ),
            }
        )

    return history


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

def get_graph_mermaid():

    return (
        chatbot
        .get_graph()
        .draw_mermaid()
    )


# ============================================================
# REBUILD KNOWLEDGE BASE
# ============================================================

def rebuild_knowledge_base():

    global vector_store

    documents = load_documents()

    if not documents:
        return 0

    chunks = split_documents(
        documents
    )

    vector_store.add_documents(
        chunks
    )

    return (
        vector_store
        ._collection
        .count()
    )
