# Transcript — YouTube Video Chatbot

A Streamlit UI for the RAG pipeline from `YouTube_Chatbot_using_LangChain.ipynb`.
Paste a YouTube video ID or link, and chat with the video using only its own
transcript as context.

**Pipeline:** transcript (`youtube-transcript-api`) → chunks
(`RecursiveCharacterTextSplitter`) → embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
→ vector store (`FAISS`) → similarity search → prompt → Gemini
(`langchain-google-genai`) → answer, all wired together with LangChain
Runnables — exactly the notebook's logic, refactored into `rag_pipeline.py`.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your Google API key

Get a free key at https://aistudio.google.com/apikey, then either:

- paste it into the **Google API key** field in the app's sidebar, or
- copy `.env.example` to `.env` and set `GOOGLE_API_KEY=...` there.

The key is only ever kept in memory for your session — it's never written to disk.

## 3. Run

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Using the app

1. Paste a YouTube video ID (e.g. `Gfr50f6ZBvo`) or a full URL into the box
   and click **Load video**. The app fetches the transcript, splits it into
   chunks, embeds them, and builds a FAISS index — this takes a few seconds.
2. Ask questions in the chat box at the bottom. Each answer is grounded only
   in the transcript; if it's not in there, the assistant says so instead of
   guessing.
3. Expand **View sources** under any answer to see the exact transcript
   chunks that were retrieved for it.
4. Use **Settings** in the sidebar to change the Gemini model, temperature,
   transcript language(s), chunk size/overlap, and how many chunks are
   retrieved per question. Click **Load a different video** to start over.

## Notes & troubleshooting

- **"Captions are disabled"** — the video's owner has turned off captions;
  there's no transcript to build a chatbot from.
- **"No transcript found in [...]"** — the video doesn't have a transcript in
  the language code(s) you set. Try adding the video's actual language (e.g.
  `hi` for Hindi) in Settings → Retrieval & chunking.
- **First load is slower** — the embedding model (~90MB) downloads once per
  machine and is cached for every video after that.
- Gemini model names change over time; if the selected model errors out,
  check https://ai.google.dev/gemini-api/docs/models for the current list
  and try another one from the sidebar dropdown.

## Project structure

```
app.py             Streamlit UI
rag_pipeline.py    Transcript fetch, chunking, embeddings, FAISS, chain (UI-agnostic)
requirements.txt
.env.example
.streamlit/config.toml   Dark theme matching the UI
```
