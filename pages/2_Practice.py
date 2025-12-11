# pages/2_Practice.py
import streamlit as st
from modules.practice_utils import generate_eval_questions, evaluate_user_answers

st.set_page_config(page_title="📝 Practice Mode")

st.title("🧪 Practice & Self-Test Mode")
st.sidebar.page_link("app.py", label="⬅️ Back to Chatbot", icon="🏠")

# Check if uploaded PDF content exists
if not st.session_state.get("uploaded_text"):
    st.warning("⚠️ Please upload and process a PDF first in the main app.")
    st.stop()

# Initialize session states
if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []
if "practice_answers" not in st.session_state:
    st.session_state.practice_answers = {}

# Generate questions button
if st.button("🧠 Generate Practice Questions"):
    with st.spinner("Generating questions..."):
        try:
            st.session_state.practice_questions = generate_eval_questions(
                st.session_state.uploaded_text,
                num_questions=5  # default to 5 for faster feedback
            )
            st.session_state.practice_answers = {}  # reset previous answers
        except Exception as e:
            st.error(f"❌ Error generating questions: {e}")

# Show questions & input fields
if st.session_state.practice_questions:
    st.subheader("📘 Answer the following questions:")

    for i, q in enumerate(st.session_state.practice_questions):
        question_text = q['question'].split("A")[0].strip()  # remove any trailing A1: answer if included
        st.markdown(f"**Q{i+1}: {question_text}**")
        user_input = st.text_area(
            label="Your Answer:",
            key=f"user_answer_{i}",
            height=100
        )
        st.session_state.practice_answers[i] = user_input

    # Evaluation button
    if st.button("✅ Submit Answers for Evaluation"):
        with st.spinner("Evaluating your answers..."):
            results, feedback, score = evaluate_user_answers(
                st.session_state.practice_questions,
                list(st.session_state.practice_answers.values())
            )

        st.success(f"🎯 Your Score: **{score}%**")

        st.markdown("### 💡 Suggestions to Improve:")
        st.info(feedback)

        with st.expander("📋 View Detailed Evaluation"):
            for i, res in enumerate(results):
                st.markdown(f"**Q{i+1}:** {res['question']}")
                st.markdown(f"**Your Answer:** {res['user_answer']}")
                st.markdown(f"**Expected Answer:** {res['expected_answer']}")
                st.markdown(f"**Score:** {round(res['score'] * 100)}%")
                st.markdown("---")
