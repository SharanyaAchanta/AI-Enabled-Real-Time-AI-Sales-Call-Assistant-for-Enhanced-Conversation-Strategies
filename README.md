📂 Project Structure
app.py - Interactive Streamlit Dashboard.

server_api.py - FastAPI server handling AI model inference.

requirements.txt - List of necessary Python libraries.

⚙️ How it Works
Audio Input: The salesperson speaks into the microphone via the Streamlit interface.

Processing: The audio file is sent to the FastAPI server.

Inference:

Whisper model transcribes the audio into text.

TinyLlama analyzes the text to determine the Intent and Sentiment.

Feedback: The server sends back a JSON response containing the analysis and a custom sales tip.

UI Update: Results are displayed instantly on the dashboard.
