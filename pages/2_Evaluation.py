import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from modules.evaluation_utils import generate_eval_questions, evaluate_responses

st.set_page_config(page_title="📊 Evaluation Dashboard", layout="wide")
st.title("🧪 Chatbot Evaluation Dashboard")

# Sidebar Navigation
st.sidebar.page_link("app.py", label="⬅️ Back to Chatbot", icon="🏠")

# Validate Required Session Data
if not st.session_state.get("uploaded_text") or not st.session_state.get("chat_chain"):
    st.warning("⚠️ Please upload and process a PDF in the main chatbot page first.")
    st.stop()

# Trigger Evaluation
if st.button("🔍 Generate Questions & Evaluate"):
    with st.spinner("Generating questions and evaluating..."):
        eval_questions = generate_eval_questions(st.session_state.uploaded_text)
        results = evaluate_responses(eval_questions, st.session_state.chat_chain)
        st.session_state["evaluation_results"] = results
    st.success("✅ Evaluation complete!")

# Results Rendering
if st.session_state.get("evaluation_results"):
    results = st.session_state["evaluation_results"]
    total = len(results)

    st.subheader("📊 Overall Scores per Question")

    questions = [f"Q{i+1}" for i in range(total)]
    cosine_scores = [r["cosine_score"] for r in results]
    bleu_scores = [r["bleu_score"] for r in results]
    rouge_scores = [r["rougeL_score"] for r in results]
    bert_scores = [r["bert_score"] for r in results]
    llm_scores = [r["llm_score"] if r["llm_score"] is not None else 0.0 for r in results]

    # Plotting grouped bar chart
    x = np.arange(total)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 2*width, cosine_scores, width, label='Cosine')
    ax.bar(x - width, bleu_scores, width, label='BLEU')
    ax.bar(x, rouge_scores, width, label='ROUGE-L')
    ax.bar(x + width, bert_scores, width, label='BERTScore')
    ax.bar(x + 2*width, llm_scores, width, label='LLM Score')

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation Scores by Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.legend()

    st.pyplot(fig)

    # Optional: Average score summary
    avg_scores = {
        "Cosine": round(np.mean(cosine_scores), 2),
        "BLEU": round(np.mean(bleu_scores), 2),
        "ROUGE-L": round(np.mean(rouge_scores), 2),
        "BERTScore": round(np.mean(bert_scores), 2),
        "LLM Score": round(np.mean([s for s in llm_scores if s is not None]), 2)
    }
    st.markdown("### 📈 Average Scores")
    for k, v in avg_scores.items():
        st.markdown(f"- **{k}**: {v}")

    # Expandable Detailed Report
    with st.expander("📋 Detailed Evaluation Report"):
        for i, r in enumerate(results):
            st.markdown(f"**Q{i+1}:** {r['question']}")
            st.markdown(f"**Expected:** {r['expected_answer']}")
            st.markdown(f"**Bot's Answer:** {r['bot_response']}")

            st.markdown(f"- 🔹 Cosine Score: **{r['cosine_score']}**")
            st.markdown(f"- 🔹 BLEU Score: **{r['bleu_score']}**")
            st.markdown(f"- 🔹 ROUGE-L Score: **{r['rougeL_score']}**")
            st.markdown(f"- 🔹 BERTScore: **{r['bert_score']}**")
            if r["llm_score"] is not None:
                st.markdown(f"- 🔹 LLM Score: **{r['llm_score']}**")
            st.markdown("---")
