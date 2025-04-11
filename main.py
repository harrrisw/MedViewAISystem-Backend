from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load environmental variable
import os
from dotenv import load_dotenv

load_dotenv()  

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://med-view-ai-system-frontend.vercel.app", "https://med-view-ai-system-frontend.vercel.app/"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

class ChatRequest(BaseModel):
    device: str
    question: str

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "mdvd"
COLLECTION_NAME = "mdve"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Cache FAISS indexes per model
device_cache = {}

def load_device_data(model_name: str):
    if model_name in device_cache:
        return device_cache[model_name]

    doc = collection.find_one({"model": model_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Device not found in database")

    questions = []
    cleaned_questions = []
    answers = []

    for item in doc.get("questions", []):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            questions.append(q)
            # Remove model name (case-insensitive, word-boundary safe)
            q_cleaned = q.lower().replace(model_name.lower(), "").strip()
            cleaned_questions.append(q_cleaned)
            answers.append(a)

    if not cleaned_questions:
        raise HTTPException(status_code=404, detail="No questions available for this model")

    embeddings = model.encode(cleaned_questions, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    device_cache[model_name] = {
        "original_questions": questions,
        "cleaned_questions": cleaned_questions,
        "answers": answers,
        "index": index
    }

    return device_cache[model_name]

@app.post("/ask")
def ask_faq(req: ChatRequest):
    entry = load_device_data(req.device)

    try:
        user_query = req.question.lower().replace(req.device.lower(), "").strip()
        user_embedding = model.encode([user_query])[0]
    except Exception as embed_err:
        raise HTTPException(status_code=500, detail="Failed to process your question.")

    try: 
        D, I = entry["index"].search(np.array([user_embedding]), k=1)
        best_idx = I[0][0]
        distance = D[0][0]
        similarity = 1 - distance / 4  # tweakable
    except Exception as search_err:
        raise HTTPException(status_code=500, detail="Failed to find a similar question.")

    if similarity > 0.7:
        return {
            "answer": entry["answers"][best_idx],
            "similarity": float(round(similarity, 2))
        }
    else:
        return {
            "answer": f"Sorry, I couldn’t find a good answer for your {req.device} question.",
            "similarity": float(round(similarity, 2))
        }