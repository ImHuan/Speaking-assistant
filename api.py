from fastapi import FastAPI
from pydantic import BaseModel
from modules.retrieval import get_relevant_chunks
from modules.generator import generate_answer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ==============================
# MEMORY LAYERS
# ==============================

# 🧠 Short-term memory cho LLM
conversation_memory = {}
MAX_HISTORY = 5

# 💾 Full history cho UI
full_history_store = {}


# ==============================
# REQUEST / RESPONSE MODELS
# ==============================

class QuestionRequest(BaseModel):
    question: str
    session_id: str


class QuestionResponse(BaseModel):
    answer: str


# ==============================
# ASK API
# ==============================

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    session_id = request.session_id
    user_question = request.question

    # ===== INIT SESSION =====
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    if session_id not in full_history_store:
        full_history_store[session_id] = []

    chat_history = conversation_memory[session_id]

    # ===== 1. RETRIEVE =====
    docs = get_relevant_chunks(user_question)

    # ===== 2. GENERATE =====
    answer = generate_answer(user_question, docs, chat_history)

    # ===== 3. UPDATE FULL HISTORY (KHÔNG CẮT) =====
    full_history_store[session_id].append((user_question, answer))

    # ===== 4. UPDATE CONVERSATION MEMORY (CÓ CẮT) =====
    chat_history.append((user_question, answer))
    conversation_memory[session_id] = chat_history[-MAX_HISTORY:]

    return {"answer": answer}


# ==============================
# GET FULL HISTORY FOR UI
# ==============================

@app.get("/history/{session_id}")
def get_history(session_id: str):

    if session_id not in full_history_store:
        return {"history": []}

    return {"history": full_history_store[session_id]}

# ==============================
# DELETE SESSION
# ==============================

@app.delete("/session/{session_id}")
def delete_session(session_id: str):

    # Xoá conversation memory
    if session_id in conversation_memory:
        del conversation_memory[session_id]

    # Xoá full history
    if session_id in full_history_store:
        del full_history_store[session_id]

    return {"message": "Session deleted"}