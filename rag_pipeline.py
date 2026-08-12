"""
rag_pipeline.py
----------------
The RAG pipeline behind the YouTube chatbot, factored out of the original
notebook into reusable functions:

    video URL/ID -> transcript -> chunks -> embeddings -> FAISS
                 -> retriever -> prompt -> Gemini -> answer

Everything here is UI-agnostic on purpose, so it can be driven from Streamlit,
a CLI, or tests without changes.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List, Sequence

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


class VideoProcessingError(Exception):
    """Any user-facing failure while turning a video into a chatbot."""


# --------------------------------------------------------------------------
# Video ID handling
# --------------------------------------------------------------------------

_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_URL_PATTERNS = [
    re.compile(r"(?:v=|\/embed\/|\/v\/|\/shorts\/|youtu\.be\/)([A-Za-z0-9_-]{11})"),
]


def extract_video_id(raw: str) -> str:
    """Accept a bare 11-character video ID or any common YouTube URL shape."""
    raw = raw.strip()
    if not raw:
        raise VideoProcessingError("Please enter a YouTube video ID or URL.")

    if _ID_RE.fullmatch(raw):
        return raw

    for pattern in _URL_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1)

    raise VideoProcessingError(
        "That doesn't look like a YouTube video ID or URL. Paste either the "
        "11-character ID (e.g. Gfr50f6ZBvo) or the full video link."
    )


# --------------------------------------------------------------------------
# Metadata (title / channel / thumbnail) — no API key required
# --------------------------------------------------------------------------

def fetch_video_metadata(video_id: str) -> dict:
    fallback = {
        "title": "YouTube video",
        "author": "Unknown channel",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title") or fallback["title"],
                "author": data.get("author_name") or fallback["author"],
                "thumbnail": data.get("thumbnail_url") or fallback["thumbnail"],
            }
    except requests.RequestException:
        pass
    return fallback


# --------------------------------------------------------------------------
# Transcript retrieval
# --------------------------------------------------------------------------

def fetch_transcript(video_id: str, languages: Sequence[str]) -> str:
    ytt = YouTubeTranscriptApi()
    try:
        transcript = ytt.fetch(video_id, languages=list(languages))
    except TranscriptsDisabled as exc:
        raise VideoProcessingError(
            "Captions are disabled for this video, so there's no transcript to read."
        ) from exc
    except NoTranscriptFound as exc:
        raise VideoProcessingError(
            f"No transcript is available in {list(languages)} for this video. "
            "Try adding another language code in Settings."
        ) from exc
    except VideoUnavailable as exc:
        raise VideoProcessingError(
            "That video is unavailable — it may be private, deleted, or region-locked."
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        raise VideoProcessingError(f"Couldn't retrieve a transcript: {exc}") from exc

    text = " ".join(snippet.text for snippet in transcript)
    if not text.strip():
        raise VideoProcessingError("The transcript for this video came back empty.")
    return text


# --------------------------------------------------------------------------
# Chunking + embeddings + vector store
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=2)
def _load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    # Cached process-wide so the ~90MB model is only loaded once per model name.
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(
    transcript_text: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[FAISS, int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript_text])
    if not chunks:
        raise VideoProcessingError("Couldn't split the transcript into chunks.")
    embeddings = _load_embeddings(embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


# --------------------------------------------------------------------------
# Prompt + chain
# --------------------------------------------------------------------------

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant answering questions about a YouTube video using
only the transcript excerpts you're given below.

Answer ONLY using the provided transcript context.

If the answer is not found in the context, respond with:
"I don't know based on the provided transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@dataclass
class VideoSession:
    """Everything the UI needs to keep around for a loaded video."""

    video_id: str
    metadata: dict
    num_chunks: int
    num_words: int
    retriever: object
    chain: object


def process_video(
    raw_input: str,
    google_api_key: str,
    languages: Sequence[str] = ("en",),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    top_k: int = 2,
) -> VideoSession:
    """Run the full pipeline for a video and return a ready-to-query session."""
    video_id = extract_video_id(raw_input)
    metadata = fetch_video_metadata(video_id)
    transcript_text = fetch_transcript(video_id, languages)
    vector_store, num_chunks = build_vector_store(
        transcript_text, chunk_size, chunk_overlap, embedding_model
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    try:
        llm = ChatGoogleGenerativeAI(
            model=llm_model, temperature=temperature, google_api_key=google_api_key
        )
    except Exception as exc:  # noqa: BLE001 - surface any auth/config error to the UI
        raise VideoProcessingError(f"Couldn't initialize the Gemini model: {exc}") from exc

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
    )
    chain = parallel_chain | PROMPT | llm | StrOutputParser()

    return VideoSession(
        video_id=video_id,
        metadata=metadata,
        num_chunks=num_chunks,
        num_words=len(transcript_text.split()),
        retriever=retriever,
        chain=chain,
    )


def ask(session: VideoSession, question: str) -> tuple[str, List]:
    """Answer a question against an already-loaded video, returning sources too."""
    if not question or not question.strip():
        raise VideoProcessingError("Please enter a question.")
    try:
        answer = session.chain.invoke(question)
    except Exception as exc:  # noqa: BLE001 - surface API errors (quota, bad key, etc.)
        raise VideoProcessingError(f"The model call failed: {exc}") from exc
    sources = session.retriever.invoke(question)
    return answer, sources