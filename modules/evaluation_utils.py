import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import cohere
import evaluate  # BLEU and ROUGE

# Cohere setup
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# ---------------- Question Generator using Cohere ----------------
def generate_eval_questions(text, num_questions=5):
    if not co:
        raise RuntimeError("❌ Cohere API key not configured.")

    prompt = f"""
You are a helpful teacher.

Generate {num_questions} important and content-specific question-answer pairs from the academic text below.

Only ask questions based strictly on the content. Avoid general knowledge.

TEXT:
{text[:3000]}

FORMAT:
Q1: <question>
A1: <ideal answer>
Q2: <question>
A2: <ideal answer>

Only return the list of Qn/An pairs in the format above. No explanation.
"""
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=800,
            temperature=0.4,
        )
        output = response.generations[0].text.strip().split("\n")
    except Exception as e:
        raise RuntimeError(f"❌ Cohere generation failed: {e}")

    questions = []
    for i in range(0, len(output), 2):
        try:
            q = output[i].split(":", 1)[1].strip()
            a = output[i + 1].split(":", 1)[1].strip()
            if q and a:
                questions.append({"question": q, "expected_answer": a})
        except Exception:
            continue

    return questions[:num_questions]

# ---------------- Evaluation Core ----------------
def evaluate_responses(eval_questions, chat_chain):
    results = []

    for q in eval_questions:
        question = q["question"]
        expected = q["expected_answer"]
        bot_answer = chat_chain.run(question)

        # Cosine similarity
        embeddings = embedder.encode([expected, bot_answer])
        cosine_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # BERTScore
        _, _, bertF1 = bert_score([bot_answer], [expected], lang="en", verbose=False)
        bert_score_val = float(bertF1[0])

        # BLEU score
        bleu_score_val = bleu_metric.compute(
            predictions=[bot_answer],
            references=[[expected]]
        )["bleu"]

        # ROUGE-L score
        rouge_score_val = rouge_metric.compute(
            predictions=[bot_answer],
            references=[expected]
        )["rougeL"]

        # Cohere LLM score
        llm_score = None
        if co:
            try:
                eval_prompt = f"""
You are an expert evaluator.

Compare the expected answer and the student's answer for this question. Return a score between 0 (worst) and 1 (perfect) based strictly on relevance, correctness, and completeness.

QUESTION:
{question}

EXPECTED:
{expected}

ANSWER:
{bot_answer}

Respond only with:
Score: <value between 0 and 1>
"""
                response = co.generate(
                    model="command-r-plus",
                    prompt=eval_prompt,
                    max_tokens=10,
                    temperature=0.2,
                )
                output = response.generations[0].text.strip()
                if "Score:" in output:
                    llm_score = float(output.split("Score:")[1].strip())
            except Exception as e:
                print("⚠️ Cohere LLM scoring failed:", e)

        results.append({
            "question": question,
            "expected_answer": expected,
            "bot_response": bot_answer,
            "cosine_score": round(cosine_score, 3),
            "bert_score": round(bert_score_val, 3),
            "bleu_score": round(bleu_score_val, 3),
            "rougeL_score": round(rouge_score_val, 3),
            "llm_score": round(llm_score, 3) if llm_score is not None else None,
        })

    return results
