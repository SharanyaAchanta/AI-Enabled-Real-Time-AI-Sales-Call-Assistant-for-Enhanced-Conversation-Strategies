import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Sales Copilot", page_icon="🚀", layout="wide")

st.title("🚀 Real-Time AI Sales Assistant")
st.markdown("""
    This application listens to your live sales calls, transcribes them in real-time, 
    and provides AI-powered suggestions based on the customer's intent.
""")

# --- 2. SESSION STATE MANAGEMENT ---
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'last_suggestion' not in st.session_state:
    st.session_state.last_suggestion = "Waiting for conversation..."
if 'run_active' not in st.session_state:
    st.session_state.run_active = False

# --- 3. MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def init_models():
    # Using 'tiny' for speed on standard laptops
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    llm = ChatOllama(model="tinyllama", temperature=0)
    return whisper, llm

whisper_model, llm_model = init_models()

# --- 4. AI LOGIC SETUP ---
class AnalysisResult(BaseModel):
    intent: str
    sentiment: str

parser = PydanticOutputParser(pydantic_object=AnalysisResult)
prompt = ChatPromptTemplate.from_template(
    "Identify the intent and sentiment of this sales dialogue.\n"
    "Dialogue: {text}\n"
    "{format_instructions}"
)
chain = prompt | llm_model | parser

# --- 5. AUDIO & BACKGROUND PROCESSING ---
audio_queue = queue.Queue()
SAMPLE_RATE = 16000
CHUNK_DURATION = 3 # Process audio every 3 seconds

def audio_input_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def speech_to_advice_worker():
    """Background thread to process audio without freezing the UI."""
    audio_buffer = np.zeros((0, 1))
    
    while st.session_state.run_active:
        while not audio_queue.empty():
            audio_buffer = np.vstack([audio_buffer, audio_queue.get()])

        if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            # Extract chunk and reset buffer
            chunk = audio_buffer[:SAMPLE_RATE * CHUNK_DURATION].flatten().astype(np.float32)
            audio_buffer = audio_buffer[SAMPLE_RATE * CHUNK_DURATION:]

            # Whisper Transcription
            segments, _ = whisper_model.transcribe(chunk, vad_filter=True)
            text = " ".join(seg.text for seg in segments).strip()

            if text:
                st.session_state.transcript_history.append(text)
                try:
                    # AI Analysis
                    ai_response = chain.invoke({
                        "text": text,
                        "format_instructions": parser.get_format_instructions()
                    })
                    
                    # Recommendation Mapping
                    strategy_map = {
                        "pricing_objection": "💡 Suggestion: Focus on long-term ROI and value over price.",
                        "complaint": "💡 Suggestion: Listen actively, empathize, and offer a solution.",
                        "greeting": "💡 Suggestion: Build rapport and set a positive tone.",
                        "inquiry": "💡 Suggestion: Provide specific features and benefits.",
                        "close": "💡 Suggestion: Ask for the next steps or the final commitment."
                    }
                    st.session_state.last_suggestion = strategy_map.get(
                        ai_response.intent, "✅ Tip: Keep the conversation natural."
                    )
                except:
                    pass
                st.rerun()

# --- 6. USER INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Transcription")
    chat_box = st.container(border=True, height=450)
    if not st.session_state.transcript_history:
        chat_box.write("*No audio detected yet...*")
    else:
        for msg in st.session_state.transcript_history[-12:]:
            chat_box.write(f"🎧 {msg}")

with col2:
    st.subheader("AI Feedback")
    st.info(st.session_state.last_suggestion)
    
    st.divider()
    
    # CONTROL BUTTONS
    if not st.session_state.run_active:
        if st.button("🔴 Start Listening", use_container_width=True):
            st.session_state.run_active = True
            threading.Thread(target=speech_to_advice_worker, daemon=True).start()
            st.rerun()
    else:
        if st.button("⏹️ Stop Assistant", use_container_width=True):
            st.session_state.run_active = False
            st.rerun()

# This part keeps the audio stream open in the main thread
if st.session_state.run_active:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_input_callback):
        while st.session_state.run_active:
            time.sleep(1)