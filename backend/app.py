import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Add it inside your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "vector.index"
CHUNKS_FILE = "chunks.pkl"

index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as file:
    chunks = pickle.load(file)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://za17384.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


def search_context(question, top_k=4):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for i in indices[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])

    return "\n\n".join(results)


@app.get("/")
def home():
    return {
        "message": "Personal AI chatbot backend is running."
    }


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        return {"answer": "Please ask a question."}

    context = search_context(question)

    prompt = f"""
You are Zeeshan Ali's personal portfolio chatbot.

Your job is to answer questions about Zeeshan using ONLY the information provided below.

Rules:
- Be friendly and professional.
- Keep answers short unless the user asks for detail.
- If the answer is not available in the information, say:
  "I don't have that information yet."
- Do not make up projects, skills, schools, awards, or experience.
- Answer in first person if appropriate, as if representing Zeeshan.

Available Information:
{context}

User Question:
{question}
"""

    response = gemini_model.generate_content(prompt)

    answer = response.text if response.text else "I don't have that information yet."

    return {"answer": answer}