"""
app.py
------
Streamlit front end for the YouTube Chatbot built with LangChain + FAISS + Gemini.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rp

load_dotenv()

st.set_page_config(
    page_title="Transcript — YouTube Chatbot",
    page_icon="▶",
    layout="centered",
)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0E1116;
        --bg-panel: #161B22;
        --bg-panel-2: #1C222B;
        --border: #2A313C;
        --text: #EDEFF2;
        --text-dim: #9AA4B2;
        --accent: #E8A33D;
        --user-bubble: #1F2937;
    }

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--bg); color: var(--text); }

    h1, h2, h3 { font-family: 'Newsreader', serif !important; letter-spacing: -0.01em; }

    #MainMenu, footer { visibility: hidden; }

    /* ---- Header ---- */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .app-header .mark {
        font-family: 'Newsreader', serif;
        font-style: italic;
        color: var(--accent);
        font-size: 2.1rem;
        line-height: 1;
    }
    .app-header h1 {
        font-size: 2.1rem;
        margin: 0;
        color: var(--text);
    }
    .app-tagline {
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-top: -0.3rem;
        margin-bottom: 1.8rem;
    }

    /* ---- Video card ---- */
    .video-card {
        display: flex;
        gap: 1rem;
        align-items: center;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.1rem;
    }
    .video-card img {
        width: 100px;
        border-radius: 6px;
        flex-shrink: 0;
    }
    .video-card .meta { min-width: 0; }
    .video-card .meta .title {
        font-weight: 600;
        font-size: 1rem;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .video-card .meta .channel {
        color: var(--text-dim);
        font-size: 0.85rem;
        margin-bottom: 0.45rem;
    }
    .pill {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-dim);
        border: 1px solid var(--border);
        background: var(--bg-panel-2);
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        margin-right: 0.35rem;
    }

    /* ---- Chat ---- */
    .stChatMessage { background: transparent !important; }
    div[data-testid="stChatMessageContent"] {
        border-radius: 10px;
    }
    .src-card {
        border: 1px solid var(--border);
        background: var(--bg-panel-2);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: var(--text-dim);
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.5;
    }
    .src-card .idx { color: var(--accent); font-weight: 600; }

    /* ---- Buttons ---- */
    .stButton > button, .stFormSubmitButton > button {
        background: var(--accent);
        color: #1a1200;
        border: none;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #f0b155;
        color: #1a1200;
    }

    .empty-state {
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        color: var(--text-dim);
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "session" not in st.session_state:
    st.session_state.session = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}
if "session_settings" not in st.session_state:
    st.session_state.session_settings = {}

DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

header_l, header_r = st.columns([5, 1])
with header_l:
    st.markdown('<div class="app-header"><span class="mark">▶</span><h1>Transcript</h1></div>', unsafe_allow_html=True)
with header_r:
    if st.session_state.session is not None:
        st.write("")  # vertical nudge to align with the title baseline
        if st.button("↻ New video", use_container_width=True):
            st.session_state.session = None
            st.session_state.chat_history = []
            st.rerun()

st.markdown('<div class="app-tagline">Ask questions about any YouTube video, answered from its own transcript.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Video loader
# --------------------------------------------------------------------------

if st.session_state.session is None:
    with st.form("load_video_form"):

        # Google API key is now loaded from .env
        api_key_input = os.environ.get("GOOGLE_API_KEY", "")

        video_input = st.text_input(
            "YouTube video ID or URL",
            placeholder="e.g. Gfr50f6ZBvo or https://www.youtube.com/watch?v=Gfr50f6ZBvo",
        )

        with st.expander("Advanced settings (optional)"):
            col1, col2 = st.columns(2)

            with col1:
                model_choice = st.selectbox(
                    "Gemini model",
                    DEFAULT_MODELS,
                    index=0
                )

                languages_raw = st.text_input(
                    "Transcript language code(s)",
                    value="en",
                    help="Comma-separated ISO codes, tried in order — e.g. 'en, en-US, hi'.",
                )

                top_k = st.slider(
                    "Chunks retrieved per question (k)",
                    2,
                    10,
                    4
                )

            with col2:
                temperature = st.slider(
                    "Temperature",
                    0.0,
                    1.0,
                    0.2,
                    0.1
                )

                chunk_size = st.slider(
                    "Chunk size (characters)",
                    300,
                    2000,
                    1000,
                    100
                )

                chunk_overlap = st.slider(
                    "Chunk overlap (characters)",
                    0,
                    500,
                    200,
                    50
                )

        submitted = st.form_submit_button(
            "Load video →",
            use_container_width=True
        )

    if submitted:
        if not api_key_input:
            st.error("GOOGLE_API_KEY not found in .env")
        elif not video_input:
            st.error("Paste a YouTube video ID or URL.")
        else:
            languages = [
                lang.strip()
                for lang in languages_raw.split(",")
                if lang.strip()
            ] or ["en"]

            steps = st.status(
                "Building the chatbot for this video…",
                expanded=True
            )

            try:
                steps.write("Fetching transcript…")

                session = rp.process_video(
                    raw_input=video_input,
                    google_api_key=api_key_input,
                    languages=languages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    llm_model=model_choice,
                    temperature=temperature,
                    top_k=top_k,
                )

                steps.write("Splitting into chunks and embedding…")

                steps.write(
                    f"Indexed {session.num_chunks} chunks "
                    f"({session.num_words:,} words)."
                )

                steps.update(
                    label="Ready!",
                    state="complete",
                    expanded=False
                )

                st.session_state.session = session
                st.session_state.chat_history = []
                st.session_state.session_settings = {
                    "model": model_choice
                }

                st.rerun()

            except rp.VideoProcessingError as exc:
                steps.update(
                    label="Couldn't load this video",
                    state="error",
                    expanded=True
                )

                st.error(str(exc))

    st.markdown(
        """
        <div class="empty-state">
        Paste a YouTube video ID or link above and hit <b>Load video</b>.<br>
        Once it's indexed, you can chat with it below.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Loaded video + chat
# --------------------------------------------------------------------------

else:
    session = st.session_state.session
    meta = session.metadata
    settings = st.session_state.session_settings

    st.markdown(
        f"""
        <div class="video-card">
            <img src="{meta['thumbnail']}" />
            <div class="meta">
                <div class="title">{meta['title']}</div>
                <div class="channel">{meta['author']}</div>
                <span class="pill">id: {session.video_id}</span>
                <span class="pill">{session.num_chunks} chunks</span>
                <span class="pill">{session.num_words:,} words</span>
                <span class="pill">{settings.get('model', '')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="▶" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"View {len(msg['sources'])} transcript sources"):
                    for i, doc in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f'<div class="src-card"><span class="idx">#{i}</span> '
                            f'{doc.page_content}</div>',
                            unsafe_allow_html=True,
                        )

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">Ask anything about this video — '
            'try "What is this video about?" to start.</div>',
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask a question about this video…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="▶"):
            with st.spinner("Reading the transcript…"):
                try:
                    answer, sources = rp.ask(session, question)
                except rp.VideoProcessingError as exc:
                    answer, sources = f"⚠️ {exc}", []
            st.markdown(answer)
            if sources:
                with st.expander(f"View {len(sources)} transcript sources"):
                    for i, doc in enumerate(sources, start=1):
                        st.markdown(
                            f'<div class="src-card"><span class="idx">#{i}</span> '
                            f'{doc.page_content}</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )