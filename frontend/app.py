import sys
from pathlib import Path
 
import streamlit as st
 
# ============================================================
# BACKEND PATH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
 
from backend.chatbot import (
    chat,
    create_thread,
    get_history,
    rebuild_knowledge_base,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)
 
DOCUMENTS_DIR = ROOT_DIR / "documents"
FILE_TYPES = ["pdf", "txt", "md", "pptx"]
 
HELP_TEXT = """
### Available Commands
- `help` — Show commands
- `history` — Show conversation history
- `clear` — Start a new conversation
- `exit` — Close the conversation
 
You can also ask questions about your uploaded documents.
"""
 
# ============================================================
# SESSION STATE
# ============================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_thread()
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
 
if "last_sidebar_upload" not in st.session_state:
    st.session_state.last_sidebar_upload = None
 
 
# ============================================================
# HELPERS
# ============================================================
def save_uploads(files):
    """Persist uploaded files to the documents folder, return their names."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for uploaded_file in files:
        (DOCUMENTS_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
        names.append(uploaded_file.name)
    return names
 
 
def start_new_thread():
    """Reset the visible transcript and begin a fresh backend thread."""
    st.session_state.conversations.pop(st.session_state.thread_id, None)
    st.session_state.thread_id = create_thread()
    st.session_state.messages = []
 
 
def format_history(history):
    if not history:
        return "No conversation history yet."
    parts = []
    for item in history:
        role = "You" if item["role"] == "user" else "AI"
        parts.append(f"**{role}:** {item['content']}")
    return "\n\n".join(parts)
 
 
# ============================================================
# HEADER
# ============================================================
st.title("🤖 ChatAI")
st.caption("Ask questions. Get answers.")
 
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Chat Controls")
 
    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------
    if st.button("🆕 New Chat", use_container_width=True, key="new_chat_button"):
        st.session_state.thread_id = create_thread()
        st.session_state.messages = []
        st.rerun()
 
    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------
    if st.button("🧹 Clear Chat", use_container_width=True, key="clear_chat_button"):
        start_new_thread()
        st.rerun()
 
    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------
    st.subheader("📜 Conversation History")
 
    if not st.session_state.conversations:
        st.caption("No previous conversations")
    else:
        for thread_id, conversation in st.session_state.conversations.items():
            if st.button(
                conversation["title"],
                key=f"conversation_{thread_id}",
                use_container_width=True,
            ):
                st.session_state.thread_id = thread_id
                st.session_state.messages = get_history(thread_id)
                st.rerun()
 
    st.divider()
 
    # ========================================================
    # DOCUMENT SECTION
    # ========================================================
    st.header("📚 Knowledge Base")
 
    sidebar_file = st.file_uploader(
        "Upload a document",
        type=FILE_TYPES,
        key="sidebar_uploader",
    )
 
    # Guard against re-writing the same file on every rerun.
    if (
        sidebar_file is not None
        and sidebar_file.name != st.session_state.last_sidebar_upload
    ):
        save_uploads([sidebar_file])
        st.session_state.last_sidebar_upload = sidebar_file.name
        st.success(f"Uploaded: {sidebar_file.name} — press Re-index to make it searchable.")
 
    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------
    if st.button("🔄 Re-index Documents", use_container_width=True, key="reindex_button"):
        with st.spinner("Building knowledge base..."):
            count = rebuild_knowledge_base()
        st.success(f"Knowledge base updated. {count} chunks indexed.")
 
    st.divider()
 
    # ========================================================
    # COMMANDS
    # ========================================================
    st.header("⌨️ Commands")
    st.markdown(
        """
`help` → Show commands
 
`history` → Show conversation history
 
`clear` → Start a new conversation
 
`exit` → Close conversation
"""
    )
 
# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# ============================================================
# USER INPUT
# ============================================================
user_input = st.chat_input(
    "Ask something...",
    accept_file=True,
    file_type=FILE_TYPES,
)
 
if user_input:
    message_text = (user_input.text or "").strip()
 
    # ========================================================
    # FILE UPLOAD
    # ========================================================
    if user_input.files:
        names = save_uploads(user_input.files)
        with st.spinner("Indexing uploaded document..."):
            count = rebuild_knowledge_base()
        st.success(
            f"Indexed {', '.join(names)}. {count} chunks available."
        )
 
    # A file-only submission needs no answer — stop here.
    if message_text:
        # ----------------------------------------------------
        # RECORD USER MESSAGE + CONVERSATION TITLE
        # ----------------------------------------------------
        st.session_state.messages.append(
            {"role": "user", "content": message_text}
        )
 
        st.session_state.conversations.setdefault(
            st.session_state.thread_id,
            {"title": message_text[:40]},
        )
 
        command = " ".join(message_text.lower().split())
        answer = None
 
        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------
        if command in {"exit", "quit", "bye", "goodbye"}:
            answer = "Goodbye! 👋"
 
        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------
        elif command in {"help", "/help", "?"}:
            answer = HELP_TEXT
 
        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------
        elif command in {"clear", "/clear", "reset", "/reset"}:
            start_new_thread()
            st.rerun()
 
        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------
        elif command in {"history", "/history"}:
            answer = format_history(get_history(st.session_state.thread_id))
 
        # ----------------------------------------------------
        # NORMAL CHAT / RAG
        # ----------------------------------------------------
        else:
            with st.spinner("Thinking..."):
                answer = chat(st.session_state.thread_id, message_text)
 
        # ----------------------------------------------------
        # RECORD AI RESPONSE
        # ----------------------------------------------------
        if answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
 
        st.rerun()
