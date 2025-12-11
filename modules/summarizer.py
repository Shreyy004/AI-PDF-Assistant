import os
import cohere
import re
from PyPDF2 import PdfReader

# Setup Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# === PDF Text Extractor ===
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# === Chunking by Sentence ===
def split_text_into_chunks(text, max_words=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count <= max_words:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# === Summarizer ===
def summarize_pdf(file):
    full_text = extract_text_from_pdf(file)
    chunks = split_text_into_chunks(full_text, max_words=500)

    print(f"🔹 Splitting into {len(chunks)} chunks...")

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            prompt = (
                "Summarize the following content **concisely**. Extract only the **most important, relevant facts or findings**. "
                "Avoid repetition and unnecessary detail. Be brief and clear.\n\n" + chunk
            )

            response = co.summarize(
                text=prompt,
                length="short",  # 'short' = concise
                format="paragraph",
                model="command-r-plus"
            )
            summaries.append(response.summary)
        except Exception as e:
            summaries.append(f"[Failed to summarize chunk {i+1}: {e}]")

    final_summary = "\n\n".join(summaries)
    return final_summary
