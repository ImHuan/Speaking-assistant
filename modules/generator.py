from dotenv import load_dotenv
import os
from groq import Groq

# ==============================
# Load environment variables
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)


# ==============================
# Generator Function (with memory)
# ==============================
def generate_answer(query_text, documents, chat_history):
    """
    query_text: câu hỏi hiện tại của user
    documents: list các document (có .page_content)
    chat_history: list các tuple [(question, answer), ...]
    """

    # ===== Build context từ vector retrieval =====
    context_text = "\n\n".join([doc.page_content for doc in documents])

    # ===== Build conversation history =====
    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # ===== Final Prompt =====
    final_prompt = f"""
You are an IELTS Speaking examiner.
Use the conversation history and context below to answer naturally.
If the context is partially relevant, still try to answer fluently.

Conversation History:
{history_text}

Context:
{context_text}

Current Question:
{query_text}

Answer in English:
"""

    # ===== Call Groq API =====
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content