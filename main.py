from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://med-view-ai-system-frontend.vercel.app/"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class ChatRequest(BaseModel):
    device: str
    question: str

# device -> {'questions': [...], 'answers': [...], 'index': FAISS}
device_data = {}

@app.get("/ping")
def ping():
    global model, device_data
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

        with open("faqs.json") as f:
            all_faqs = json.load(f)

        grouped = {}
        for item in all_faqs:
            device = item["device"].lower()
            grouped.setdefault(device, []).append(item)

        for device, faqs in grouped.items():
            questions = [f["question"] for f in faqs]
            answers = [f["answer"] for f in faqs]
            embeddings = model.encode(questions, convert_to_numpy=True)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            device_data[device.lower()] = {
                "questions": questions,
                "answers": answers,
                "index": index
            }

    return {"status": "ok", "model_loaded": model is not None}

@app.post("/ask")
def ask_faq(req: ChatRequest):
    if model is None:
        ping()
        
    device = req.device.lower()
    if device not in device_data:
        raise HTTPException(status_code=404, detail="Device not found")

    entry = device_data[device]
    user_embedding = model.encode([req.question])[0]
    D, I = entry["index"].search(np.array([user_embedding]), k=1)
    best_idx = I[0][0]
    distance = D[0][0]
    similarity = 1 - distance / 4  # Rough cosine-like approximation

    if similarity > 0.7:
        return {
            "answer": entry["answers"][best_idx],
            "similarity": float(round(similarity, 2))
        }
    else:
        return {
            "answer": f"Sorry, I couldn’t find a good answer for your {device} question.",
            "similarity": float(round(similarity, 2))
        }
