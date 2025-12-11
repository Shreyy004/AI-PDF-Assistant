# # modules/practice_utils.py
# import os
# import cohere
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from PyPDF2 import PdfReader

# # Initialize Cohere and embedder
# co = cohere.Client(os.getenv("COHERE_API_KEY"))
# embedder = SentenceTransformer("all-MiniLM-L6-v2")


# def extract_text_from_pdf(file):
#     """Extract full text from a PDF file."""
#     reader = PdfReader(file)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#     return text.strip()


# def generate_eval_questions(text, num_questions=10):
#     """Generate strictly content-based QA pairs using Cohere."""
#     prompt = f"""
# You are an academic AI assistant.

# Generate {num_questions} important, concept-focused, and content-specific question-answer pairs from the text below.
# Only ask questions based on the provided text. Avoid general knowledge or unrelated topics.

# TEXT:
# {text[:3000]}

# FORMAT:
# Q1: <question>
# A1: <ideal answer>
# Q2: <question>
# A2: <ideal answer>
# ...
# Only return the list of Qn/An pairs in the format shown above. No explanation needed.
# """

#     try:
#         response = co.generate(
#             model="command-r-plus",
#             prompt=prompt,
#             max_tokens=1200,
#             temperature=0.5
#         )
#         lines = response.generations[0].text.strip().split('\n')
#     except Exception as e:
#         print(f"❌ Cohere generation failed: {e}")
#         return []

#     questions = []
#     for i in range(0, len(lines), 2):
#         try:
#             q = lines[i].split(":", 1)[1].strip()
#             a = lines[i + 1].split(":", 1)[1].strip()
#             if q and a:
#                 questions.append({"question": q, "expected_answer": a})
#         except (IndexError, ValueError):
#             continue

#     return questions


# def evaluate_user_answers(eval_questions, user_answers):
#     """Evaluate answers based on cosine similarity with embedding."""
#     results = []
#     total_score = 0

#     for i, q in enumerate(eval_questions):
#         expected = q['expected_answer']
#         user_resp = user_answers[i] if i < len(user_answers) else ""

#         embeddings = embedder.encode([expected, user_resp])
#         score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

#         results.append({
#             "question": q["question"],
#             "expected_answer": expected,
#             "user_answer": user_resp,
#             "score": score
#         })

#         total_score += score

#     percentage_score = round((total_score / len(eval_questions)) * 100, 2)
#     feedback = generate_feedback(results)

#     return results, feedback, percentage_score


# def generate_feedback(results):
#     """Generate actionable feedback based on weak answers."""
#     weak_areas = [res["question"] for res in results if res["score"] < 0.5]

#     if not weak_areas:
#         return "✅ Excellent work! You answered all questions accurately. Keep practicing to retain the knowledge."

#     return (
#         "⚠️ Focus on the following areas where your answers were weaker:\n\n"
#         + "\n".join(f"• {q}" for q in weak_areas)
#         + "\n\n📌 Consider revisiting those topics in your PDF for better understanding."
#     )

import os
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Load sentence transformer for answer evaluation
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# PDF Text Extraction Utility
# -----------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text.strip()

# -----------------------------
# Ollama Availability Check
# -----------------------------
def is_ollama_available():
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        return True
    except Exception:
        return False

# -----------------------------
# Generate Questions using Ollama (phi)
# -----------------------------
def generate_eval_questions(text, num_questions=5):
    if not is_ollama_available():
        raise RuntimeError("❌ Ollama is not available. Please install and run Ollama with a model like 'phi'.")

    prompt = f"""
You are an academic assistant.

Your task is to generate exactly {num_questions} content-specific question-answer pairs from the given text.

✅ Only create questions based on the text provided.
❌ Do NOT use external knowledge or add explanations.
❗ Ensure the response has exactly {num_questions} question-answer pairs.

TEXT:
{text[:3000]}

FORMAT:
Q1: <question>
A1: <ideal answer>
Q2: <question>
A2: <ideal answer>
...
Only return the Q/A pairs. No intro, no explanation.
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "phi"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        output = result.stdout.decode("utf-8").strip().splitlines()
    except subprocess.CalledProcessError as e:
        print("❌ Ollama generation failed:", e.stderr.decode())
        return []

    # Parse QnA pairs from output
    questions = []
    i = 0
    while i < len(output) - 1:
        if output[i].startswith("Q") and output[i + 1].startswith("A"):
            try:
                q = output[i].split(":", 1)[1].strip()
                a = output[i + 1].split(":", 1)[1].strip()
                if q and a:
                    questions.append({"question": q, "expected_answer": a})
                i += 2
            except Exception:
                i += 1
        else:
            i += 1

    # Enforce exactly `num_questions`
    if len(questions) > num_questions:
        questions = questions[:num_questions]

    # Retry once if insufficient
    if len(questions) < num_questions:
        print(f"⚠️ Only {len(questions)} questions generated. Retrying...")
        return generate_eval_questions(text, num_questions)

    return questions

# -----------------------------
# Evaluate User Answers
# -----------------------------
def evaluate_user_answers(eval_questions, user_answers):
    results = []
    total_score = 0

    for i, q in enumerate(eval_questions):
        expected = q["expected_answer"]
        user_resp = user_answers[i] if i < len(user_answers) else ""

        embeddings = embedder.encode([expected, user_resp])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        results.append({
            "question": q["question"],
            "expected_answer": expected,
            "user_answer": user_resp,
            "score": score
        })

        total_score += score

    percentage_score = round((total_score / len(eval_questions)) * 100, 2)
    feedback = generate_feedback(results)

    return results, feedback, percentage_score

# -----------------------------
# Feedback Generator
# -----------------------------
def generate_feedback(results):
    weak_areas = [res["question"] for res in results if res["score"] < 0.5]

    if not weak_areas:
        return "✅ Excellent work! You answered all questions accurately. Keep practicing to retain the knowledge."

    return (
        "⚠️ Focus on the following areas where your answers were weaker:\n\n"
        + "\n".join(f"• {q}" for q in weak_areas)
        + "\n\n📌 Consider revisiting those topics in your PDF for better understanding."
    )
