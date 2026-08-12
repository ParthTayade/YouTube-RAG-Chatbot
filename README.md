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

**▶️ [Watch the Live Project Demo]("")**


## 📸 Project Screenshots

The following screenshots demonstrate the application's interface,
video processing workflow, conversational question answering, and
retrieved transcript sources.

### 1. Application Home Screen

<img width="100%" alt="Application Home Screen" src="https://github.com/user-attachments/assets/0bc8db7b-c627-4fe0-8a9a-8b12301afcc4" />

### 2. Video Loaded Successfully

<img width="100%" alt="Video Loaded Successfully" src="https://github.com/user-attachments/assets/349cbee6-eb66-4424-98c6-3d684384c958" />

### 3. Asking Questions About the Video

<img width="100%" alt="Asking Questions About the Video" src="https://github.com/user-attachments/assets/395a4f8b-e79e-4287-85f7-1392e3435992" />

<img width="100%" alt="Chatbot Answer" src="https://github.com/user-attachments/assets/6e659693-da27-45d4-bc6a-4a764cb9e946" />

### 4. Retrieved Transcript Sources

<img width="100%" alt="Retrieved Transcript Sources" src="https://github.com/user-attachments/assets/26a72c8a-11a5-4409-a0b0-78e66af86983" />

<img width="100%" alt="Transcript Source Details" src="https://github.com/user-attachments/assets/b124847c-9080-4c73-aea2-6403c23f8f0b" />


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
