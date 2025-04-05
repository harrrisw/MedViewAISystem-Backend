from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class ChatRequest(BaseModel):
    device: str
    question: str

# device -> {'questions': [...], 'answers': [...], 'index': FAISS}
device_data = {}

with open("faqs.json") as f:
    all_faqs = json.load(f)

# Group by device
grouped = {}
for item in all_faqs:
    device = item["device"].lower()
    grouped.setdefault(device, []).append(item)

# Create index per device
for device, faqs in grouped.items():
    questions = [f["question"] for f in faqs]
    answers = [f["answer"] for f in faqs]
    embeddings = model.encode(questions, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    device_data[device] = {
        "questions": questions,
        "answers": answers,
        "index": index
    }

@app.post("/ask")
def ask_faq(req: ChatRequest):
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
            "similarity": round(similarity, 2)
        }
    else:
        return {
            "answer": f"Sorry, I couldn’t find a good answer for your {device} question.",
            "similarity": round(similarity, 2)
        }
