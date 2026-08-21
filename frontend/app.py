import sys
from pathlib import Path

import streamlit as st


# ============================================================
# BACKEND PATH
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(ROOT_DIR),
)


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


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = create_thread()


if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversations" not in st.session_state:

    st.session_state.conversations = {}


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖ChatAI"
)

st.caption(
    "Ask questions. Get answers."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Chat Controls"
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

if st.button(
    "🆕 New Chat",
    use_container_width=True,
):

    new_thread_id = create_thread()

    st.session_state.thread_id = new_thread_id

    st.session_state.messages = []

    st.session_state.conversations[
        new_thread_id
    ] = {
        "title": "New Chat"
    }

    st.rerun()

    # --------------------------------------------------------
# NEW CHAT
# --------------------------------------------------------

if st.button(
    "🆕 New Chat",
    use_container_width=True,
):

    new_thread_id = create_thread()

    st.session_state.thread_id = new_thread_id

    st.session_state.messages = []

    st.session_state.conversations[
        new_thread_id
    ] = {
        "title": "New Chat"
    }

    st.rerun()


# --------------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------------

st.subheader(
    "📜 Conversation History"
)

if not st.session_state.conversations:

    st.caption(
        "No previous conversations"
    )

else:

    for thread_id, conversation in (
        st.session_state.conversations.items()
    ):

        if st.button(
            conversation["title"],
            key=f"conversation_{thread_id}",
            use_container_width=True,
        ):

            history = get_history(
                thread_id
            )

            st.session_state.thread_id = (
                thread_id
            )

            st.session_state.messages = (
                history
            )

            st.rerun()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True,
    ):

        st.session_state.thread_id = (
            create_thread()
        )

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # DOCUMENT SECTION
    # ========================================================

    st.header(
        "📚 Knowledge Base"
    )

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[
            "pdf",
            "txt",
            "md",
            "pptx"
        ],
    )


    if uploaded_file:

        documents_dir = (
            ROOT_DIR / "documents"
        )

        documents_dir.mkdir(
            exist_ok=True
        )

        file_path = (
            documents_dir
            / uploaded_file.name
        )

        file_path.write_bytes(
            uploaded_file.getbuffer()
        )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------

    if st.button(
        "🔄 Re-index Documents",
        use_container_width=True,
    ):

        with st.spinner(
            "Building knowledge base..."
        ):

            count = (
                rebuild_knowledge_base()
            )

        st.success(
            f"Knowledge base updated. "
            f"{count} chunks indexed."
        )


    st.divider()



    # ========================================================
    # COMMANDS
    # ========================================================

    st.header(
        "⌨️ Commands"
    )

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

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something...",
    accept_file=True,
    file_type=["pdf", "txt", "md", "pptx"],
)


if user_input:
    message_text = user_input.text.strip()

    # ========================================================
    # FILE UPLOAD
    # ========================================================


    if user_input.files:
      for uploaded_file in user_input.files:

        documents_dir = (ROOT_DIR / "documents")

        documents_dir.mkdir(exist_ok=True)

        file_path = (
            documents_dir / uploaded_file.name
            )

        file_path.write_bytes(
            uploaded_file.getbuffer()
        )

      with st.spinner("Indexing uploaded document..."):
        count = rebuild_knowledge_base()

      st.success(
          f"Document indexed successfully. "
          f"{count} chunks available."
      )

    if not message_text:
      st.rerun()

    command = (
        " ".join(
            message_text
            .lower()
            .split()
        )
    )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": message_text,
        }
    )

    # --------------------------------------------------------
    # SAVE CONVERSATION TITLE
    # --------------------------------------------------------

if (
    st.session_state.thread_id
    not in st.session_state.conversations
):

    st.session_state.conversations[
        st.session_state.thread_id
    ] = {
        "title": message_text[:40]
    }


    # ========================================================
    # EXIT
    # ========================================================

    if command in {
        "exit",
        "quit",
        "bye",
        "goodbye",
    }:

        answer = (
            "Goodbye! 👋"
        )


    # ========================================================
    # HELP
    # ========================================================

    elif command in {
        "help",
        "/help",
        "?",
    }:

        answer = """
### Available Commands

- `help` — Show commands
- `history` — Show conversation history
- `clear` — Start a new conversation
- `exit` — Close the conversation

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
    }:

        st.session_state.thread_id = (
            create_thread()
        )

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # HISTORY
    # ========================================================

    elif command in {
        "history",
        "/history",
    }:

        history = get_history(
            st.session_state.thread_id
        )

        if not history:

            answer = (
                "No conversation "
                "history yet."
            )

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

            answer = "\n\n".join(
                history_parts
            )


    # ========================================================
    # NORMAL CHAT / RAG
    # ========================================================

    else:

        with st.spinner(
            "Thinking..."
        ):

            answer = chat(
                st.session_state.thread_id,
                message_text,
            )


    # ========================================================
    # DISPLAY AI RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()
