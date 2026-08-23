import sys
import hashlib
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = (Path(__file__).resolve().parent.parent)

sys.path.insert(0,str(ROOT_DIR),)


# ============================================================
# BACKEND IMPORT
# ============================================================

from backend.chatbot import (
    chat,
    create_thread,
    get_history,
    rebuild_knowledge_base,
    chunk_count,
    indexed_files,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChatAI",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        create_thread()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


if "uploaded_file_hash" not in st.session_state:

    st.session_state.uploaded_file_hash = None


# ============================================================
# HEADER
# ============================================================

st.title("ChatAI")

st.caption("Ask questions. Get answers.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Chat Controls")

    # --------------------------------------------------------
    # NEW CONVERSATION
    # --------------------------------------------------------

    if st.button("New Conversation",use_container_width=True,):
        st.session_state.thread_id = (create_thread())
        st.session_state.messages = []
        st.rerun()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button("Clear Chat",use_container_width=True,):
        st.session_state.thread_id = (create_thread())
        st.session_state.messages = []
        st.rerun()

    st.divider()


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.header("Knowledge Base")


    # --------------------------------------------------------
    # CURRENT INDEX STATUS
    # --------------------------------------------------------

    current_chunks = chunk_count()
    current_files = indexed_files()

    if current_files:

        st.success(
            f"{len(current_files)} "
            f"document(s) indexed"
        )

        st.caption(f"Chunks: {current_chunks}")

        for filename in sorted(current_files):
            st.write(f"• {filename}")
    else:
        st.info("No documents indexed.")


    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf","txt","md","pptx",],)

    if uploaded_file:
        documents_dir = (ROOT_DIR / "documents")
        documents_dir.mkdir(parents=True,exist_ok=True,)
        file_path = (documents_dir / uploaded_file.name)

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        file_bytes = (uploaded_file.getvalue())

        # ----------------------------------------------------
        # Stable hash
        # ----------------------------------------------------

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # ----------------------------------------------------
        # Only index when the file is new/changed.
        # ----------------------------------------------------

        if (st.session_state.uploaded_file_hash != file_hash):
            try:
                file_path.write_bytes(file_bytes)
                with st.spinner("Indexing document..."):
                    count = (rebuild_knowledge_base())

                st.session_state.uploaded_file_hash = (file_hash)
                st.success(f"{uploaded_file.name} "f"indexed successfully.")
                st.info(f"{count} chunks indexed.")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to index document:\n{e}")


    # --------------------------------------------------------
    # REINDEX BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔄 Re-index Documents",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Rebuilding knowledge base..."
            ):

                count = (
                    rebuild_knowledge_base()
                )

            st.session_state.uploaded_file_hash = None

            st.success(
                f"Knowledge base rebuilt. "
                f"{count} chunks indexed."
            )

            st.rerun()

        except Exception as e:
            st.error(f"Re-indexing failed:\n{e}")

    st.divider()


    # ========================================================
    # COMMANDS
    # ========================================================

    st.header("⌨Commands")

    st.markdown(
        """
`help` → Show commands

`history` → Show conversation history

`clear` → Start a new conversation

`exit` → End the conversation
"""
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Ask something...")


if user_input:
    message_text = (user_input.strip())
    if not message_text:
        st.stop()


    # ========================================================
    # NORMALIZE COMMAND
    # ========================================================

    command = (" ".join(message_text.lower().split()))

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    st.session_state.messages.append({
            "role": "user",
            "content": message_text,
        })


    # ========================================================
    # EXIT
    # ========================================================

    if command in {
        "exit",
        "quit",
        "bye",
        "goodbye",
        "close",
        "end",
        "stop",
        "see you",
        "see ya",
        "talk to you later",
    }:

        answer = ("Goodbye! 👋")


    # ========================================================
    # HELP
    # ========================================================

    elif command in {"help","/help","?",}:

        answer = """
### Available Commands

- `help` — Show commands
- `history` — Show conversation history
- `clear` — Start a new conversation
- `exit` — End the conversation

You can also ask questions about your uploaded documents.
"""


    # ========================================================
    # CLEAR
    # ========================================================

    elif command in {
        "clear",
        "/clear",
        "reset",
        "/reset",
        "start over",
        "new chat",
    }:

        st.session_state.thread_id = (create_thread())
        st.session_state.messages = []
        st.rerun()


    # ========================================================
    # HISTORY
    # ========================================================

    elif command in {
        "history",
        "/history",
        "show history",
        "conversation history",
    }:

        history = get_history(st.session_state.thread_id)

        if not history:
            answer = ("No conversation history yet.")
        else:
            history_parts = []
            for item in history:
                role = (
                    "You"
                    if item["role"] == "user"
                    else "AI"
                )

                history_parts.append(
                    f"**{role}:** "
                    f"{item['content']}"
                )

            answer = ("\n\n".join(history_parts))


    # ========================================================
    # NORMAL CHAT / RAG
    # ========================================================

    else:

        with st.spinner("Thinking..."):
            try:
                answer = chat(st.session_state.thread_id,message_text,)
            except Exception as e:
                answer = (
                    "An error occurred while "
                    "processing your question.\n\n"
                    f"```text\n{e}\n```"
                )


    # ========================================================
    # DISPLAY AI RESPONSE
    # ========================================================

    st.session_state.messages.append({"role": "assistant", "content": answer,})
    st.rerun()
