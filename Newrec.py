import sounddevice as sd
import queue
import numpy as np
from faster_whisper import WhisperModel
import threading
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. LLM Setup (TinyLlama is good, keep temperature 0)
class LiveIntent(BaseModel):
    intent: str
    sentiment: str

llm = ChatOllama(model="tinyllama", temperature=0)
parser = PydanticOutputParser(pydantic_object=LiveIntent)
prompt = ChatPromptTemplate.from_template(
    "Extract intent and sentiment.\nText: {text}\n{format_instructions}"
)
chain = prompt | llm | parser

# Audio Settings
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3  
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def process_audio():
    """Background thread to handle transcription and LLM"""
    buffer = np.zeros((0, 1))
    
    while True:
        # Collect audio samples
        while not audio_queue.empty():
            buffer = np.vstack([buffer, audio_queue.get()])

        if len(buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
            chunk = buffer[:SAMPLE_RATE * BUFFER_SECONDS].flatten().astype(np.float32)
            buffer = buffer[SAMPLE_RATE * BUFFER_SECONDS:]

            # Transcribe (Directly from memory, no temp files needed)
            segments, _ = model.transcribe(chunk, vad_filter=True)
            text = " ".join(seg.text for seg in segments).strip()

            if text:
                print(f"Transcript: {text}")
                try:
                    # AI Analysis
                    res = chain.invoke({
                        "text": text,
                        "format_instructions": parser.get_format_instructions()
                    })
                    
                    # Simple Logic for Tips
                    tips = {
                        "pricing_objection": "Tip: Explain the value, don't ask discount!",
                        "complaint": "Tip: Say sorry and tell then that you will help.",
                        "greeting": "Tip: Be enthusiastic!"
                    }
                    print(f"AI Suggestion: {tips.get(res.intent, 'Keep going!')}")
                except Exception as e:
                    print("AI is little slow, logic is being skipped...")

# Start Background Thread
threading.Thread(target=process_audio, daemon=True).start()

# Start Recording
print("Listening... (Press Ctrl+C to stop)")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while True:
        pass