import os
import re
import uuid
import hashlib
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
    UnstructuredPowerPointLoader,
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

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True,)

CHROMA_DIR.mkdir(parents=True, exist_ok=True,)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY to your .env file."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=GROQ_API_KEY,
)


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
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ChatAI, a helpful and professional AI assistant.

Rules:

1. Answer the user's actual question directly.
2. Use simple and clear language.
3. Keep answers concise unless more detail is necessary.
4. Use Markdown when useful.
5. Never claim that a document contains something unless the
   retrieved document context actually supports it.
6. Never invent document information.
"""


# ============================================================
# RAG PROMPT
# ============================================================

RAG_PROMPT = """
You are a highly accurate document question-answering assistant.

The user has uploaded one or more documents.

You MUST answer the user's question using the retrieved document
context below.

IMPORTANT RULES:

1. Carefully read ALL retrieved document context.
2. Answer the exact question asked by the user.
3. If the answer is present in the context, answer directly.
4. Do NOT say "please upload the PDF" if document context is provided.
5. Do NOT invent information.
6. If the question asks for a list, extract ALL relevant items
   visible in the retrieved context.
7. For resume questions, extract information such as:
   - name
   - skills
   - technical skills
   - programming languages
   - frameworks
   - libraries
   - education
   - projects
   - experience
   - certifications
   - achievements
8. Preserve the terminology used in the document.
9. If page information is provided, mention the page when useful.
10. If the retrieved context genuinely does not contain the answer,
    say:

    "I couldn't find that information in the uploaded document."

11. Do not use your general knowledge to fabricate missing
    document information.

Retrieved Document Context:

{context}
"""


# ============================================================
# DOCUMENT LOADING
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".pptx",}


def load_documents():
    """
    Load all supported documents from the documents folder.
    """

    documents = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        try:

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            if extension == ".pdf":

                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            # ------------------------------------------------
            # TXT / Markdown
            # ------------------------------------------------

            elif extension in {".txt",".md",}:

                loader = TextLoader(str(file_path),encoding="utf-8",)
                docs = loader.load()

            # ------------------------------------------------
            # PowerPoint
            # ------------------------------------------------

            elif extension == ".pptx":

                loader = UnstructuredPowerPointLoader(str(file_path))
                docs = loader.load()
                
            else:
                
                continue

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            for doc in docs:
                doc.metadata["source"] = (file_path.name)

                # Normalize page number.
                if "page" in doc.metadata:
                    try:
                        doc.metadata["page"] = int(doc.metadata["page"])
                    except Exception:
                        pass

            documents.extend(docs)

            print(
                f"Loaded: {file_path.name} "
                f"({len(docs)} pages/sections)"
            )

        except Exception as e:

            print(
                f"[ERROR] Could not load "
                f"{file_path.name}: {e}"
            )

    return documents


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def split_documents(documents):
    """
    Split documents into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(
        documents
    )


# ============================================================
# CHROMA CONFIGURATION
# ============================================================

COLLECTION_NAME = "chatbot_documents"
CHROMA_SPACE = "cosine"


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def build_vector_store():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_configuration={
            "hnsw": {
                "space": CHROMA_SPACE
            }
        },
    )


vector_store = build_vector_store()


# ============================================================
# VECTOR STORE INFORMATION
# ============================================================

def chunk_count():
    try:
        return (vector_store._collection.count())

    except Exception as e:
        print(f"[ERROR] Could not get chunk count: {e}")
        return 0


def indexed_files():
    try:
        data = vector_store.get(include=["metadatas"])
        metadatas = (data.get("metadatas") or [])

        return {
            metadata.get("source")
            for metadata in metadatas
            if metadata
            and metadata.get("source")
        }

    except Exception as e:

        print(f"[ERROR] Could not inspect vector store: {e}")
        return set()


def files_on_disk():

    return {
        file_path.name
        for file_path in DOCUMENTS_DIR.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    }


# ============================================================
# REBUILD KNOWLEDGE BASE
# ============================================================

def rebuild_knowledge_base():

    global vector_store

    print(
        "\n========================================"
    )

    print(
        "REBUILDING KNOWLEDGE BASE"
    )

    print(
        "========================================"
    )

    documents = load_documents()

    if not documents:
        print("No supported documents found.")
        return 0
    
    chunks = split_documents(documents)

    print(f"Documents loaded : {len(documents)}")

    print(f"Chunks created   : {len(chunks)}")

    try:
        client = vector_store._client
        client.delete_collection(name=COLLECTION_NAME)

        print("Old Chroma collection deleted.")

    except Exception as e:
        print(f"Could not delete old collection: {e}")

    # --------------------------------------------------------
    # Create a completely new cosine collection.
    # --------------------------------------------------------

    vector_store = build_vector_store()

    # --------------------------------------------------------
    # Add chunks.
    # --------------------------------------------------------

    vector_store.add_documents(chunks)

    count = chunk_count()

    print(f"Chunks indexed   : {count}")

    print(
        f"Files indexed    : "
        f"{sorted(indexed_files())}"
    )

    print(
        "========================================\n"
    )
    return count


# ============================================================
# SYNC KNOWLEDGE BASE
# ============================================================

def sync_knowledge_base():

    disk_files = files_on_disk()

    # No documents.
    if not disk_files:
        return 0
    indexed = indexed_files()

    # Rebuild if:
    #
    # 1. index is empty
    # 2. a file was added
    # 3. a file was removed
    #

    if (chunk_count() == 0 or indexed != disk_files):
        return rebuild_knowledge_base()
    return chunk_count()


# ============================================================
# INITIAL SYNC
# ============================================================

sync_knowledge_base()


# ============================================================
# CHAT STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage],add_messages,]
    retrieved_context: str
    
    use_rag: bool


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

RETRIEVAL_K = 6


# ============================================================
# FOLLOW-UP QUESTION HANDLING
# ============================================================

FOLLOW_UP_PREFIXES = (
    "and ",
    "also ",
    "what about",
    "how about",
    "his ",
    "her ",
    "their ",
    "its ",
    "that ",
    "this ",
    "those ",
    "these ",
    "it ",
)


def build_retrieval_query(messages):

    if not messages:

        return ""

    question = (messages[-1].content.strip())
    if not question:
        return ""
    lowered = question.lower()

    # --------------------------------------------------------
    # Detect follow-up questions.
    # --------------------------------------------------------

    is_follow_up = (
        len(question.split()) <= 8
        or lowered.startswith(
            FOLLOW_UP_PREFIXES
        )
    )

    if not is_follow_up:
        return question

    # --------------------------------------------------------
    # Find previous user question.
    # --------------------------------------------------------

    previous_questions = [
        message.content.strip()
        for message in messages[:-1]
        if isinstance(
            message,
            HumanMessage,
        )
        and message.content.strip()
    ]

    if not previous_questions:
        return question
    previous_question = (previous_questions[-1])
    return (f"{previous_question} {question}")


# ============================================================
# RAG ROUTER
# ============================================================

def router_node(state: ChatState):

    messages = state["messages"]

    if not messages:

        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    question = (messages[-1].content.strip())
    if not question:
        return {
            "retrieved_context": "",
            "use_rag": False,
        }

    # --------------------------------------------------------
    # If no documents are indexed, use normal chat.
    # --------------------------------------------------------

    if chunk_count() == 0:

        print("No documents indexed.")

        return {"retrieved_context": "", "use_rag": False,}

    search_query = build_retrieval_query(messages)

    print(f"\n[QUERY] {search_query}")

    # --------------------------------------------------------
    # Retrieve documents.
    #
    # IMPORTANT:
    # We intentionally do NOT use a hard relevance-score
    # threshold here.
    #
    # If a document exists, the top retrieved chunks are passed
    # to the RAG model. The model is instructed to answer only
    # from those chunks.
    #
    # This prevents the previous problem where a valid PDF was
    # indexed but the router decided to use normal chat.
    # --------------------------------------------------------

    try:
        results = (vector_store.similarity_search(search_query, k=RETRIEVAL_K,))

    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")

        return {"retrieved_context": "","use_rag": False,}

    if not results:
        print("No results.")
        return {"retrieved_context": "", "use_rag": False,}

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []
    for index, document in enumerate(results, start=1,):
        source = document.metadata.get("source", "Unknown",)
        page = document.metadata.get("page")

        if page is not None:
            source_text = (f"{source}," f"page {page + 1}")

        else:
            source_text = source
        context_parts.append(
            f"[Retrieved Chunk {index}]\n"
            f"[Source: {source_text}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    print(f"Retrieved {len(results)} chunks.")

    for index, document in enumerate(results, start=1,):

        source = document.metadata.get("source", "Unknown",)

        page = document.metadata.get("page")

        preview = (document.page_content[:100].replace("\n", " "))

        print(
            f"[RAG {index}] "
            f"{source} "
            f"page={page} "
            f"| {preview}"
        )

    return {
        "retrieved_context": context,
        "use_rag": True,
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_question(state: ChatState):

    if state.get("use_rag", False,):
        return "rag"
    
    return "chat"


# ============================================================
# NORMAL CHAT
# ============================================================

def chat_node(state: ChatState):

    messages = [SystemMessage(content=SYSTEM_PROMPT),*state["messages"],]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================
# RAG NODE
# ============================================================

def rag_node(state: ChatState):

    context = state.get("retrieved_context","",)

    prompt = RAG_PROMPT.format(context=context)

    messages = [SystemMessage(content=prompt),*state["messages"],]

    response = llm.invoke(messages)

    return {"messages": [response]}


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)


# ==============================================================
# Add Nodes
# ==============================================================

graph.add_node("router", router_node,)
graph.add_node("chat",chat_node,)
graph.add_node("rag", rag_node,)


# =============================================================
# Add Edges
# =============================================================

graph.add_edge(START, "router",)
graph.add_conditional_edges("router",route_question,{
        "chat": "chat",
        "rag": "rag",
    },)

graph.add_edge("chat",END,)
graph.add_edge("rag",END,)


# ============================================================
# MEMORY
# ============================================================

checkpointer = MemorySaver()


# ============================================================
# COMPILE GRAPH
# ============================================================

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# RESPONSE CLEANING
# ============================================================

def clean_response(text: str):
    if not isinstance(text, str,):
        text = str(text)

    text = text.replace("\u202f"," ",)
    text = text.replace("\u00a0"," ",)
    text = text.replace(r"\(", "",)
    text = text.replace(r"\)", "",)
    text = text.replace(r"\times","×",)
    text = text.replace(r"\div","÷",)
    text = text.replace(r"\cdot","·",)
    text = re.sub(r"\{,\}",",",text,)
    text = re.sub(r"\n{3,}", "\n\n", text,)
    return text.strip()


# ============================================================
# THREAD
# ============================================================

def create_thread():

    return str(
        uuid.uuid4()
    )


# ============================================================
# CHAT
# ============================================================

def chat(
    thread_id: str,
    user_message: str,
):

    user_message = (
        user_message
        .strip()
    )

    if not user_message:

        return ("Please enter a question.")

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

    messages = snapshot.values.get("messages",[],)
    history = []

    for message in messages:
        if isinstance(message,HumanMessage,):
            role = "user"
            
        else:
            role = "assistant"
        history.append({"role": role, "content": clean_response(message.content),})
    return history


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnose(question: str,k: int = RETRIEVAL_K,):

    print("\n========================================")
    print("RAG DIAGNOSTICS")
    print("========================================")
    
    print(f"Chunks indexed : {chunk_count()}")

    print(
        f"Files indexed  : "
        f"{sorted(indexed_files())}"
    )

    print(
        f"Files on disk  : "
        f"{sorted(files_on_disk())}"
    )
    print(f"Chroma space   : {CHROMA_SPACE}")
    print(f"\nQuestion:\n{question}\n")

    if chunk_count() == 0:
        print("NO DOCUMENTS ARE INDEXED.")
        return
    
    try:
        results = (vector_store.similarity_search(question, k=k,))

    except Exception as e:
        print(f"Retrieval error: {e}")
        return

    if not results:
        print("No results returned.")
        return

    print(f"Retrieved {len(results)} chunks:\n")

    for index, document in enumerate(results, start=1,):
        source = document.metadata.get("source", "Unknown",)
        page = document.metadata.get("page")
        preview = (document.page_content[:150].replace("\n", " "))

        print(
            f"{index}. "
            f"{source} "
            f"page={page} "
            f"| {preview}"
        )

    print("\n========================================")


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys
    query = (" ".join(sys.argv[1:]) or "What are the skills mentioned in the PDF?")
    diagnose(query)
