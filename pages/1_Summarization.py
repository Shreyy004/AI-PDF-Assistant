import streamlit as st
from modules.summarizer import summarize_pdf

st.set_page_config(page_title="📄 PDF Summarization")
st.title("📝 PDF Summarization")

# Sidebar navigation
st.sidebar.page_link("app.py", label="⬅️ Back to Chatbot", icon="🏠")
pdf_files = st.sidebar.file_uploader("📂 Upload PDF files to summarize", accept_multiple_files=True)

if not pdf_files:
    st.info("Please upload PDF files using the sidebar to generate summaries.")
else:
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    for pdf in pdf_files:
        st.markdown(f"### 📘 {pdf.name}")
        if pdf.name not in st.session_state.summaries:
            try:
                with st.spinner("Summarizing..."):
                    summary = summarize_pdf(pdf)
                st.session_state.summaries[pdf.name] = summary
            except Exception as e:
                st.error(f"❌ Failed to summarize {pdf.name}: {e}")
                continue  # Skip download if failed

        # Display the summary
        summary = st.session_state.summaries[pdf.name]
        st.markdown("**Summary:**")
        st.success(summary)

        # Download button
        file_name = pdf.name.replace(".pdf", "_summary.txt")
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=file_name,
            mime="text/plain"
        )
