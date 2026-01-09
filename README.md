# 🎙️ AI Enabled Real-Time Sales Call Assistant

**Project Title:** AI Enabled Real Time AI Sales Call Assistant for Enhanced Conversation Strategies  
**Domain:** Artificial Intelligence / Machine Learning / NLP  
**Internship:** Infosys Springboard (Capstone Project)

---

## 📝 Project Overview
This project is a high-performance **Real-Time AI Assistant** designed for sales professionals. It listens to live customer calls, transcribes the conversation, and uses Large Language Models (LLMs) to provide instant feedback on customer sentiment and intent. 

The goal is to empower sales representatives with "AI-driven empathy" and conversation strategies to improve customer satisfaction and conversion rates.

---

## ✨ Key Features
* **Live Transcription:** Converts speech to text with high accuracy using **OpenAI's Whisper**.
* **Intent Recognition:** Automatically identifies if the customer is inquiring, complaining, or greeting.
* **Sentiment Analysis:** Detects the emotional tone (Positive, Negative, Neutral) of the speaker.
* **Smart Coaching Tips:** Displays real-time sales advice (e.g., "Acknowledge & Empathize") based on the conversation flow.
* **Distributed Architecture:** Uses a **FastAPI** backend to handle complex AI computations independently from the frontend.



---

## 🛠️ Tech Stack
* **Frontend:** Streamlit (For the Interactive Dashboard)
* **Backend:** FastAPI (For High-performance API handling)
* **Speech-to-Text:** Faster-Whisper (Open-source STT)
* **LLM Engine:** TinyLlama (via Ollama)
* **Communication:** REST API, Pydantic, JSON
* **Audio Processing:** SoundDevice, SoundFile

---

## 📂 Repository Structure
```text
├── app.py              # Streamlit Web Application (UI)
├── server_api.py       # FastAPI Server (AI Logic & Inference)
├── test.py             # Script to verify backend connectivity
├── requirements.txt    # List of dependencies
└── README.md           # Project Documentation
```
---
## ⚙️ How it Works
**Audio Input**: The salesperson speaks into the microphone via the Streamlit interface.
**Processing**: The audio file is sent to the FastAPI server.
**Inference:**
- Whisper model transcribes the audio into text.
- TinyLlama analyzes the text to determine the Intent and Sentiment.
- Feedback: The server sends back a JSON response containing the analysis and a custom sales tip.
- UI Update: Results are displayed instantly on the dashboard.
