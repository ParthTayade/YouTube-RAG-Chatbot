# 🎥 YouTube RAG Chatbot

An AI-powered YouTube video chatbot that lets users ask questions about a
YouTube video and receive answers grounded exclusively in the video's
transcript.

The application combines **LangChain**, **FAISS**, **Hugging Face
embeddings**, **YouTube Transcript API**, and **Google Gemini** to implement
a complete Retrieval-Augmented Generation (RAG) pipeline.

Instead of sending the entire transcript to the language model, the system
splits the transcript into smaller chunks, converts them into embeddings,
retrieves the most relevant chunks for each question, and provides those
chunks to Gemini as context.

---

## 🎬 Live Demo

<!-- ============================================================
     ADD YOUR PROJECT DEMO VIDEO HERE
     
     Option 1 — YouTube:
     Replace YOUR_VIDEO_ID with your YouTube video's ID.

     Option 2 — GitHub:
     Upload the video to GitHub and replace the placeholder below
     with the GitHub-generated video URL.
     ============================================================ -->

**▶️ [Watch the Live Project Demo](YOUR_VIDEO_LINK_HERE)**

<!-- Example:
[![YouTube RAG Chatbot Demo](screenshots/demo-thumbnail.png)](YOUR_VIDEO_LINK_HERE)
-->

---

## 📸 Project Screenshots

The following screenshots demonstrate the application's interface,
video processing workflow, conversational question answering, and
retrieved transcript sources.

### 1. Application Home Screen

<!-- ============================================================
     ADD SCREENSHOT HERE
     Example:
     ![Application Home Screen](screenshots/home.png)
     ============================================================ -->

![Application Home Screen](screenshots/home.png)


### 2. Loading a YouTube Video

<!-- ============================================================
     ADD SCREENSHOT HERE
     Replace the filename with your actual screenshot filename.
     ============================================================ -->

![Loading YouTube Video](screenshots/video-loading.png)


### 3. Video Loaded Successfully

<!-- ============================================================
     ADD SCREENSHOT HERE
     ============================================================ -->

![Loaded Video](screenshots/video-loaded.png)


### 4. Asking Questions About the Video

<!-- ============================================================
     ADD SCREENSHOT HERE
     ============================================================ -->

![Chat with YouTube Video](screenshots/chat.png)


### 5. Retrieved Transcript Sources

<!-- ============================================================
     ADD SCREENSHOT HERE
     This screenshot should ideally show the "View transcript
     sources" section underneath an answer.
     ============================================================ -->

![Retrieved Transcript Sources](screenshots/sources.png)

---

## ✨ Key Features

- 🎥 Accepts a YouTube video ID or full YouTube URL
- 📝 Automatically retrieves the video's transcript
- ✂️ Splits transcripts into configurable text chunks
- 🧠 Generates embeddings using Hugging Face
- 🔎 Stores embeddings in a FAISS vector database
- 🎯 Retrieves the most relevant transcript chunks for each question
- 🤖 Uses Google Gemini for grounded answer generation
- 🔗 Uses LangChain Runnables to build the RAG chain
- 📚 Displays the transcript sources retrieved for each answer
- 🌐 Fetches video title, channel, and thumbnail metadata
- ⚙️ Provides configurable retrieval and generation settings
- 🔐 Loads the Google API key from `.env` rather than exposing it in the UI
- 💬 Maintains conversational history during the current session
- ⚠️ Provides user-friendly errors for unavailable videos, missing
  transcripts, disabled captions, and model/API failures

---

## 🧠 How It Works

The application follows a complete Retrieval-Augmented Generation workflow:

```text
                    YouTube Video
                          │
                          ▼
                 Video ID / URL
                          │
                          ▼
                Transcript Retrieval
                          │
                          ▼
                  Text Chunking
                          │
                          ▼
              Hugging Face Embeddings
                          │
                          ▼
                    FAISS Index
                          │
                          ▼
                    User Question
                          │
                          ▼
                Similarity Retrieval
                          │
                          ▼
             Relevant Transcript Chunks
                          │
                          ▼
                  Prompt + Context
                          │
                          ▼
                  Google Gemini
                          │
                          ▼
                  Grounded Answer
                          │
                          ▼
             Retrieved Sources Displayed
