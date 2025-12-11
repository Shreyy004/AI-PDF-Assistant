# modules/voice_utils.py
import speech_recognition as sr
from gtts import gTTS
import tempfile
import streamlit as st
import re

def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        return f"Error: {str(e)}"

def clean_text_for_tts(text):
    # Remove markdown formatting like **bold**, *italic*, __underline__, etc.
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
    text = re.sub(r'__(.*?)__', r'\1', text)      # __underline__
    
    # Remove leading bullets or symbols from each line
    text = re.sub(r'^\s*[\*\-•]+\s*', '', text, flags=re.MULTILINE)

    # Remove leftover markdown/control characters
    text = re.sub(r'[•*_~`>#]', '', text)

    # Collapse multiple spaces into one
    # text = re.sub(r'\s+', ' ', text)

    return text.strip()

def speak_text(text, lang='en'):
    try:
        cleaned = clean_text_for_tts(text)
        tts = gTTS(cleaned, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio_bytes = open(tmp.name, 'rb').read()
            st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
