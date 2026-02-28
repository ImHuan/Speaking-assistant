from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

# load 1 lần duy nhất khi server start
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

def get_relevant_chunks(query: str, top_k: int = 4):
    results = db.similarity_search(query, k=top_k)
    return results