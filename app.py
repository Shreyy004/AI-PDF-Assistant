import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from modules.pdf_handler import extract_text_from_pdfs
from modules.chat_engine import create_vectorstore, get_conversation_chain
from modules.voice_utils import transcribe_audio, speak_text
from modules.translate_utils import translate_text

# Set Streamlit page config and title
st.set_page_config(page_title="Smart PDF Chatbot", layout="wide")
st.title("📚 PDF Chatbot with Voice & Memory")

# Initialize session state
for key in ["chat_chain", "chat_history", "spoken_input", "uploaded_text"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# Supported Languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te"
}

# Sidebar
pdf_files = st.sidebar.file_uploader("📂 Upload PDF files", accept_multiple_files=True)
lang_full = st.sidebar.selectbox("🌍 Translate responses to", list(LANGUAGES.keys()), index=0)
lang_code = LANGUAGES[lang_full]

# Page links added here 👇
st.sidebar.divider()
st.sidebar.page_link("pages/1_Summarization.py", label="Go to Summarization", icon="📝")
st.sidebar.page_link("pages/2_Evaluation.py", label="Go to Evaluation", icon="📈")
st.sidebar.page_link("pages/2_Practice.py", label="Practice Mode", icon="🧠")

# Main App Functionality
if st.sidebar.button("🔄 Process PDFs"):
    with st.spinner("Reading and indexing documents..."):
        text = extract_text_from_pdfs(pdf_files)
        st.session_state.uploaded_text = text
        st.session_state.chat_chain = get_conversation_chain(create_vectorstore(text))
        st.session_state.chat_history = []
    st.sidebar.success("✅ Chatbot is ready!")

for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)
        if sender == "🤖 Bot":
            if st.button("🔊 Read Out Loud", key=f"btn_{hash(msg)}"):
                speak_text(msg, lang=lang_code)

chat_col, mic_col = st.columns([9, 1])
with mic_col:
    if st.button("🎙️", help="Click to speak your question"):
        spoken = transcribe_audio()
        st.session_state.spoken_input = spoken
        st.success(f"You said: {spoken}")

with chat_col:
    typed_input = st.chat_input("Type your question or click 🎙️ to speak", key="chatbox")

user_input = st.session_state.spoken_input or typed_input

if user_input and st.session_state.chat_chain:
    st.session_state.chat_history.append(("🧑 You", user_input))
    with st.spinner("Thinking..."):
        result = st.session_state.chat_chain({"question": user_input})
        answer = result['answer']
        if lang_code != "en":
            answer = translate_text(answer, target_lang=lang_code)
        st.session_state.chat_history.append(("🤖 Bot", answer))
    st.session_state.spoken_input = None
    st.rerun()
