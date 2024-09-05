import streamlit as st
import tempfile
import os
from transformers import pipeline
from pydub import AudioSegment
import numpy as np

# Initialize the transcription pipeline
@st.cache_resource
def load_transcriber():
    return pipeline("automatic-speech-recognition", model="openai/whisper-base")

transcriber = load_transcriber()

# Load and preprocess audio file
def load_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        return samples, sample_rate
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None

# Transcribe function
def transcribe_audio(audio_file):
    try:
        waveform, sample_rate = load_audio(audio_file)
        if waveform is None or sample_rate is None:
            return "Error: Unable to load audio file."
        
        print(f"Audio loaded. Shape: {waveform.shape}, Sample rate: {sample_rate}")
        
        result = transcriber(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return f"Error during transcription: {str(e)}"

# Streamlit app
st.title("Audio Transcription App")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Transcribe audio
            transcription = transcribe_audio(tmp_file_path)
            
            # Remove temporary file
            os.unlink(tmp_file_path)
        
        if not transcription.startswith("Error"):
            st.success("Transcription Complete!")
            st.write(transcription)
        else:
            st.error(transcription)

st.markdown("---")
st.write("Powered by Whisper ASR model")